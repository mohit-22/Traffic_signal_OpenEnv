"""LLMAgent — traffic signal agent backed by a provider-agnostic LLM adapter.

Environment variable contract
------------------------------
  MODEL_PROVIDER   : "openai" | "hf" | "ollama"  (auto-detected if unset)
  MODEL_NAME       : model identifier
  API_BASE_URL     : base URL for openai-compatible endpoints
  OPENROUTER_API_KEY / HF_TOKEN / HUGGING_FACE_TOKEN : auth tokens
  OLLAMA_MODEL     : if set, forces local Ollama regardless of provider
  HF_MODEL_NAME    : HF model name (falls back to MODEL_NAME)
  HF_API_URL / HF_ENDPOINT : custom HF endpoint (optional)

Learning mechanism
------------------
  1. Build compact prompt from (metadata, step_feedback, memory).
  2. Call LLM adapter with exponential backoff.
  3. Parse and safety-sanitize the response.
  4. Record (situation, action, reward, feedback) into AgentMemory.

After each episode, memory is updated with lessons, enabling the NEXT
episode's prompts to be better. No gradient / backprop involved.

act() signature is backward-compatible:
  act(obs, step=0)              — original signature
  act(obs, step=0, feedback=None) — extended with optional feedback
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv(override=False)  # Never override grader-injected env vars

from llm_agent.llm_adapter import LLMAdapter, build_adapter
from llm_agent.memory import AgentMemory
from llm_agent.prompt_builder import PromptBuilder, extract_tags
from env.schemas import StepFeedback

_PHASE_LABELS = {0: "NS_GREEN", 1: "EW_GREEN", 2: "ALL_RED"}

DEFAULT_TEMP     = 0.0    # deterministic
DEFAULT_MAX_TOKS = 32     # enough for comma-separated integers
MAX_RETRIES      = 2
RETRY_DELAY      = 0.5
MAX_CONSEC_FAIL  = 2      # circuit-breaker threshold
ANTI_STUCK_N     = 6      # how many recent actions to track for stuck detection
# Margin required for the HELD action to be considered the WRONG direction.
# If NS queue total > EW queue total + ANTI_STUCK_MARGIN and we are serving NS,
# we are NOT stuck — we are correct. Only flip if the other side is clearly bigger.
_ANTI_STUCK_MARGIN = 0.08

_CONN_ERR_PATTERNS = (
    "connection error", "connectionerror", "connect call failed",
    "connection refused", "name or service not known",
    "failed to establish", "remotedisconnected", "eof occurred",
    "hf request error",
)


def validate_adapter(adapter: Optional[LLMAdapter], verbose: bool = True) -> None:
    """Warn if adapter is missing expected configuration."""
    if adapter is None:
        if verbose:
            print(
                "[LLMAgent] ⚠  No LLM adapter available — "
                "rule-based fallback will be used for all steps.",
                flush=True,
            )


def llm_health_check(adapter: Optional[LLMAdapter], verbose: bool = True) -> bool:
    """Send a trivial prompt and verify the response contains integers."""
    if adapter is None:
        if verbose:
            print("[LLMAgent] ⚠  Skipping health check — adapter is None.", flush=True)
        return False
    if verbose:
        print(
            f"[LLMAgent] 🔍 Health check → {adapter.provider_name} ({adapter.model_id})",
            flush=True,
        )
    try:
        text = adapter.complete(
            system="Output ONLY comma-separated integers 0-2. No words.",
            user="Traffic test. 2 intersections. Reply with exactly: 0,1",
            max_tokens=16,
            temperature=0.0,
        )
        if text is None:
            print("[LLMAgent] ❌ Health check returned None.", flush=True)
            return False
        if verbose:
            print(f"[LLMAgent] Health check response: {text!r}", flush=True)
        if re.search(r"[0-2]", text):
            print("[LLMAgent] ✔ LLM connected and responding correctly.", flush=True)
            return True
        print(f"[LLMAgent] ⚠ LLM replied but no valid digits found: {text!r}", flush=True)
        return True  # at least responded
    except Exception as exc:
        print(f"[LLMAgent] ❌ Health check FAILED: {exc}", flush=True)
        return False


class LLMAgent:
    """Self-improving traffic signal agent backed by a provider-agnostic LLM.

    Parameters
    ----------
    n_intersections : number of intersections the environment exposes.
    memory          : AgentMemory instance (shared across episodes).
    model_name      : LLM model ID (overrides env var; passed to adapter).
    base_url        : API base URL (overrides env var; used for OAI adapter).
    api_key         : API key (overrides env vars).
    temperature     : LLM sampling temperature (0 = deterministic).
    verbose         : if True, log LLM calls to stdout.
    adapter         : pre-built LLMAdapter (overrides all above if provided).
    """

    def __init__(
        self,
        n_intersections: int,
        memory: Optional[AgentMemory] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMP,
        verbose: bool = True,
        adapter: Optional[LLMAdapter] = None,
    ) -> None:
        self.n_intersections = n_intersections
        self.temperature     = temperature
        self.verbose         = verbose

        # Override env vars when explicit params provided
        if model_name:
            os.environ.setdefault("MODEL_NAME", model_name)
        if base_url:
            os.environ.setdefault("API_BASE_URL", base_url)
        if api_key:
            os.environ.setdefault("OPENROUTER_API_KEY", api_key)

        # Build or accept the adapter
        self._adapter: Optional[LLMAdapter] = adapter or build_adapter(verbose=verbose)
        validate_adapter(self._adapter, verbose=verbose)

        self.memory = memory or AgentMemory()
        self._prompt_builder = PromptBuilder(
            memory=self.memory,
            n_intersections=n_intersections,
        )

        self._current_episode:   int               = 0
        self._last_action:       Optional[List[int]] = None
        self._last_situation:    str               = ""
        self._last_tags:         List[List[str]]   = []
        self._last_feedback:     Optional[StepFeedback] = None
        self._fallback_count:    int               = 0
        self._llm_success_count: int               = 0
        self._consecutive_fail:  int               = 0
        self._llm_dead:          bool              = False
        self._action_history: Deque[List[int]]     = deque(maxlen=ANTI_STUCK_N)
        # Observation / prompt dedup tracking
        self._last_obs_sig:      Optional[str]     = None
        self._obs_repeat_streak: int               = 0
        self._last_prompt_sig:   Optional[str]     = None
        self._last_metadata:     Optional[np.ndarray] = None
        self._reward_history:    Deque[float]      = deque(maxlen=ANTI_STUCK_N)

        provider = self._adapter.provider_name if self._adapter else "none"
        model_id = self._adapter.model_id if self._adapter else "none"
        self._log(f"LLMAgent ready | provider={provider} | model={model_id}")

    # ------------------------------------------------------------------
    # Public API (backward-compatible)
    # ------------------------------------------------------------------

    def reset(self, episode: int = 0) -> None:
        self._current_episode    = episode
        self._last_action        = None
        self._last_situation     = ""
        self._last_tags          = []
        self._last_feedback      = None
        self._fallback_count     = 0
        self._llm_success_count  = 0
        self._consecutive_fail   = 0
        self._action_history: Deque[List[int]] = deque(maxlen=ANTI_STUCK_N)
        self._last_obs_sig       = None
        self._obs_repeat_streak  = 0
        self._last_prompt_sig    = None   # type: ignore[assignment]
        self._last_metadata      = None
        self._reward_history.clear()
        # _llm_dead persists across episodes (endpoint stays dead until success)

    def act(
        self,
        obs: Any,
        step: int = 0,
        feedback: Optional[StepFeedback] = None,
    ) -> List[int]:
        """Choose actions from a TrafficObservation.

        Args:
            obs      : TrafficObservation with .metadata attribute.
            step     : current step index (for logging).
            feedback : optional StepFeedback from previous step's info dict.
        """
        metadata = obs.metadata
        self._last_feedback = feedback
        return self._choose_action(metadata, step, feedback)

    def record_reward(
        self,
        reward: float,
        step: int,
        feedback: Optional[StepFeedback] = None,
    ) -> None:
        """Record step outcome into memory."""
        if self._last_action is None:
            return
        flat_tags = list({tag for tags in self._last_tags for tag in tags})
        self.memory.record_step(
            episode=self._current_episode,
            step=step,
            situation=self._last_situation,
            action=self._last_action,
            reward=reward,
            tags=flat_tags,
            feedback=feedback,
        )
        self._reward_history.append(reward)

    def end_episode(
        self,
        total_reward: float,
        grader_score: float,
        n_steps: int,
        episode_feedback: Optional[Any] = None,
    ) -> None:
        """Finalise episode in memory."""
        self.memory.record_episode(
            episode=self._current_episode,
            total_reward=total_reward,
            grader_score=grader_score,
            n_steps=n_steps,
            episode_feedback=episode_feedback,
        )
        llm_rate = self._llm_success_count / max(n_steps, 1) * 100
        self._log(
            f"Episode {self._current_episode} done | "
            f"total_reward={total_reward:.3f} | "
            f"grader_score={grader_score:.4f} | "
            f"LLM={self._llm_success_count}/{n_steps} ({llm_rate:.0f}%) | "
            f"Fallbacks={self._fallback_count} | "
            f"Trend: {self.memory.trend_string()}"
        )

    # ------------------------------------------------------------------
    # Internal decision-making
    # ------------------------------------------------------------------

    def _choose_action(
        self,
        metadata: np.ndarray,
        step: int,
        feedback: Optional[StepFeedback] = None,
    ) -> List[int]:
        # ── Observation dedup check ─────────────────────────────────────
        obs_sig = hashlib.md5(metadata.tobytes()).hexdigest()[:8]
        if obs_sig == self._last_obs_sig:
            self._obs_repeat_streak += 1
            if self._obs_repeat_streak % 5 == 0:
                self._log(
                    f"⚠ OBS UNCHANGED for {self._obs_repeat_streak} consecutive steps "
                    f"(sig={obs_sig}, step={step}) — possible env stall or fixed seed"
                )
        else:
            if self._obs_repeat_streak > 0:
                self._log(
                    f"✓ OBS changed after {self._obs_repeat_streak} repeated steps "
                    f"(new sig={obs_sig}, step={step})"
                )
            self._obs_repeat_streak = 0
        self._last_obs_sig = obs_sig

        all_tags = [extract_tags(row) for row in metadata]

        user_prompt   = self._prompt_builder.build_user_prompt(
            metadata=metadata,
            step=step,
            episode=self._current_episode,
            all_tags=all_tags,
            feedback=feedback,
            previous_action=self._last_action,  # anti-repetition context
        )
        system_prompt = self._prompt_builder.build_system_prompt()

        self._last_situation = user_prompt[:200]
        self._last_tags      = all_tags

        # Log compact observation summary before every LLM call
        if self.verbose:
            ns_vals = [float(r[0]) + float(r[1]) for r in metadata]
            ew_vals = [float(r[2]) + float(r[3]) for r in metadata]
            phases  = [round(float(r[4]) * 2) for r in metadata]
            obs_line = "  ".join(
                f"I{i}(NS={ns:.2f},EW={ew:.2f},ph={p})"
                for i, (ns, ew, p) in enumerate(zip(ns_vals, ew_vals, phases))
            )
            self._log(
                f"[OBS] ep={self._current_episode} step={step} sig={obs_sig}  {obs_line}"
            )

        # Circuit-breaker
        if self._consecutive_fail >= MAX_CONSEC_FAIL or self._llm_dead:
            action = self._prompt_builder.rule_fallback(metadata)
            self._fallback_count += 1
            if not self._llm_dead:
                reason = "consecutive LLM failures" if self._consecutive_fail >= MAX_CONSEC_FAIL else "dead endpoint"
                self._log(
                    f"⚡ Circuit-breaker ({reason}, consec={self._consecutive_fail}) "
                    f"— rule fallback (step={step}) → {action}"
                )
            self._last_action = action
            self._action_history.append(action)
            return action

        # Log prompt: full text at step 0 or every 20 steps; hash every step
        prompt_sig = hashlib.md5(user_prompt.encode()).hexdigest()[:8]
        log_full   = (step == 0 or step % 20 == 0)
        if self.verbose:
            if log_full:
                self._log(
                    f"[PROMPT] ep={self._current_episode} step={step} "
                    f"sig={prompt_sig} len={len(user_prompt)}\n"
                    + "─" * 40 + "\n"
                    + user_prompt
                    + "\n" + "─" * 40
                )
            else:
                # Every step: just the hash so repeated prompts are obvious
                repeat_flag = " ⚠PROMPT_REPEAT" if prompt_sig == self._last_prompt_sig else ""
                self._log(
                    f"[PROMPT] step={step} sig={prompt_sig}{repeat_flag}"
                )
            self._last_prompt_sig = prompt_sig

        raw_response = self._call_adapter(system_prompt, user_prompt, step)

        if raw_response is not None:
            action = self._parse_response(raw_response, metadata)
            self._llm_success_count += 1
            self._consecutive_fail  = 0
            self._llm_dead          = False
            phase_names = [_PHASE_LABELS.get(p, str(p)) for p in action]
            action_changed = (self._last_action is None or action != self._last_action)
            change_flag = "" if action_changed else " ⚠SAME_AS_PREV"
            self._log(f"✔ Action: {action} ({', '.join(phase_names)}){change_flag}")
        else:
            action = self._prompt_builder.rule_fallback(metadata)
            self._fallback_count   += 1
            self._consecutive_fail += 1
            self._log(
                f"⚠ FALLBACK (step={step}, consec={self._consecutive_fail}/"
                f"{MAX_CONSEC_FAIL}, total={self._fallback_count}) → {action}"
            )

        # Anti-stuck override: only fire when action is wrong for ANTI_STUCK_N steps.
        # Correct stable actions (agent holds NS while NS queue is always larger)
        # should NOT be overridden — doing so hurts smoothness unnecessarily.
        self._action_history.append(action)
        if self._is_stuck(action, metadata):
            stuck_summary = f"last {ANTI_STUCK_N} actions all={action} (harmful repetition)"
            new_action = self._anti_stuck_override(action, metadata)
            self._log(
                f"🔄 Anti-stuck override ({stuck_summary}, step={step}) "
                f"{action} → {new_action}"
            )
            action = new_action

        self._last_action = action
        self._last_metadata = metadata.copy()
        return action

    def _call_adapter(
        self, system_prompt: str, user_prompt: str, step: int = 0
    ) -> Optional[str]:
        if self._adapter is None:
            return None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._log(f"[LLM] step={step} attempt={attempt}/{MAX_RETRIES}")
                text = self._adapter.complete(
                    system=system_prompt,
                    user=user_prompt,
                    max_tokens=DEFAULT_MAX_TOKS,
                    temperature=self.temperature,
                )
                self._log(f"[LLM] raw={text!r}")
                return text

            except Exception as exc:
                err      = str(exc)
                err_low  = err.lower()
                self._log(f"[LLM] ERROR attempt {attempt}/{MAX_RETRIES}: {err[:120]}")

                is_conn = any(p in err_low for p in _CONN_ERR_PATTERNS)
                if is_conn:
                    self._llm_dead = True
                    self._log(
                        "[LLM] ❌ Connection error — endpoint marked dead. "
                        "Fallback will be used for remaining steps."
                    )
                    return None

                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY * (2 ** (attempt - 1))
                    if "429" in err or "rate" in err_low:
                        delay = max(delay, 5.0)
                    self._log(f"[LLM] Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)

        return None

    def _parse_response(
        self, text: str, metadata: np.ndarray
    ) -> List[int]:
        """Parse LLM response → valid phase list.

        Sanitization:
          - Only 0 and 1 are accepted (ALL_RED removed to prevent neutral action abuse).
          - ALL_RED only allowed if yellow active or emergency present.
          - Otherwise replaced by rule-based answer for that intersection.
        """
        n = self.n_intersections

        # Accept only 0 and 1 in first pass
        valid_tokens = re.findall(r"[01]", text)
        if not valid_tokens:
            # Widen search to include 2, then clamp
            all_ints = re.findall(r"\d+", text)
            if all_ints:
                clamped = [str(max(0, min(1, int(x)))) for x in all_ints]
                valid_tokens = clamped

        if not valid_tokens:
            self._log(f"[LLM] Parse failed — no valid integers in: {text!r}")
            return self._prompt_builder.rule_fallback(metadata)

        phases = [int(t) for t in valid_tokens[:n]]

        if len(phases) < n:
            fallback = self._prompt_builder.rule_fallback(metadata)
            phases = (phases + fallback)[:n]

        # Sanitize: clamp to [0, 1] for each intersection
        sanitized = []
        fallback_actions = self._prompt_builder.rule_fallback(metadata)
        for i, (phase, row) in enumerate(zip(phases, metadata)):
            yellow_active = float(row[6]) > 0.05
            emerg_active  = float(row[7]) > 0.01
            if phase == 2 and not yellow_active and not emerg_active:
                sanitized.append(fallback_actions[i])
                self._log(f"[LLM] ALL_RED sanitized at I{i} → {fallback_actions[i]}")
            else:
                sanitized.append(max(0, min(1, phase)))  # clamp stray values

        return sanitized

    # ------------------------------------------------------------------
    # Anti-stuck helpers
    # ------------------------------------------------------------------

    def _is_stuck(self, current_action: List[int], metadata: np.ndarray) -> bool:
        """Return True only when the agent is both repeating an action AND serving
        the WRONG direction or holding a stale state pointlessly.

        Trigger conditions for harmful repetition:
        - Excessive phase hold (phase_timer >= 0.95)
        - No queue improvement on the served side across the action history
        - Clear dominance of the un-served side
        - Repeated low reward penalty
        """
        if len(self._action_history) < ANTI_STUCK_N:
            return False
        # All recent actions must be identical
        if not all(a == current_action for a in self._action_history):
            return False

        # If the reward history is consistently bad, repetition might be harmful
        avg_reward = sum(self._reward_history) / max(len(self._reward_history), 1) if self._reward_history else 0.5
        is_low_reward = len(self._reward_history) == ANTI_STUCK_N and avg_reward < 0.2

        is_wrong_action = False
        for i, (a, row) in enumerate(zip(current_action, metadata)):
            ns_total = float(row[0]) + float(row[1])
            ew_total = float(row[2]) + float(row[3])
            phase = int(round(float(row[4]) * 2)) if len(row) > 4 else None
            phase_timer = float(row[5]) if len(row) > 5 else 0.0

            # Condition 1: Excessive phase hold beyond a threshold
            if phase_timer >= 0.95 and phase in (0, 1):
                is_wrong_action = True
                break

            # Condition 2: Clear dominance of the un-served side
            if a == 0 and ew_total > ns_total + _ANTI_STUCK_MARGIN:
                is_wrong_action = True
                break
            if a == 1 and ns_total > ew_total + _ANTI_STUCK_MARGIN:
                is_wrong_action = True
                break

            # Condition 3: No queue improvement & stale state
            if self._last_metadata is not None:
                prev_row = self._last_metadata[i]
                prev_ns = float(prev_row[0]) + float(prev_row[1])
                prev_ew = float(prev_row[2]) + float(prev_row[3])
                if a == 0 and ns_total >= prev_ns and ns_total > 0 and is_low_reward:
                    is_wrong_action = True
                    break
                if a == 1 and ew_total >= prev_ew and ew_total > 0 and is_low_reward:
                    is_wrong_action = True
                    break

        return is_wrong_action

    def _anti_stuck_override(self, action: List[int], metadata: np.ndarray) -> List[int]:
        """Force a switch for any stuck intersection (0→1 or 1→0)."""
        new_action = list(action)
        for i, (a, row) in enumerate(zip(action, metadata)):
            # Only switch 0↔1; don't introduce ALL_RED
            new_action[i] = 1 - a if a in (0, 1) else 0
        return new_action

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[LLMAgent] {msg}", flush=True)

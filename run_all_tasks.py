#!/usr/bin/env python3
"""
run_all_tasks.py — Evaluate the LLM traffic agent across EASY, MEDIUM, and HARD tasks.

This script is the single entry point for running all three difficulty levels
sequentially and reporting per-task scores with diagnostics.

Usage:
    python run_all_tasks.py [--episodes N] [--seed SEED] [--fallback-only] [--quiet]

Diagnostics added (non-invasive):
    - Per-step observation hash to detect repeated states
    - Per-step observation summary (NS_queue, EW_queue, phase, timer)
    - Logs when action is unchanged from previous step
    - Logs anti-stuck override with reason
    - Logs seed used for each episode reset
    - Separate per-task scores printed at end
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(override=False)  # Never override grader-injected env vars

import numpy as np

from baseline.rule_based_agent import RuleBasedAgent
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader
from tasks.task_easy import make_env as make_easy
from tasks.task_medium import make_env as make_medium
from tasks.task_hard import make_env as make_hard


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY = {
    "easy":   (make_easy,   EasyGrader),
    "medium": (make_medium, MediumGrader),
    "hard":   (make_hard,   HardGrader),
}

_DIFFICULTY_BASELINES: Dict[str, Optional[float]] = {
    "easy":   None,
    "medium": None,
    "hard":   None,
}


# ---------------------------------------------------------------------------
# Observation diagnostics helpers
# ---------------------------------------------------------------------------

def _obs_signature(metadata: np.ndarray) -> str:
    """Compute a short hash signature of the observation metadata."""
    raw = metadata.tobytes()
    return hashlib.md5(raw).hexdigest()[:8]


def _obs_summary(metadata: np.ndarray) -> str:
    """One-line human-readable observation summary per intersection."""
    parts = []
    for i, row in enumerate(metadata):
        q0, q1, q2, q3 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        ns = q0 + q1
        ew = q2 + q3
        phase_raw = float(row[4])
        timer_raw = float(row[5])
        yellow = float(row[6])
        emerg = float(row[7])
        spill = float(row[10])

        phase_idx = min(2, max(0, round(phase_raw * 2)))
        phase_names = ["NS_GREEN", "EW_GREEN", "ALL_RED"]
        pname = phase_names[phase_idx]

        flags = []
        if yellow > 0.05:
            flags.append("YLW")
        if emerg > 0.01:
            flags.append("EMRG")
        if spill > 0.5:
            flags.append("SPILL")

        flag_str = "/" + "|".join(flags) if flags else ""
        parts.append(
            f"I{i}[NS={ns:.2f} EW={ew:.2f} ph={pname} t={timer_raw:.2f}{flag_str}]"
        )
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Rule-based baseline runner
# ---------------------------------------------------------------------------

def _run_baseline(task_id: str, seed: int, verbose: bool = True) -> float:
    """Run one episode with rule-based agent. Returns grader score."""
    make_fn, grader_cls = _TASK_REGISTRY[task_id]
    env = make_fn(seed=seed)
    grader = grader_cls()
    agent = RuleBasedAgent(
        n_intersections=env.n_intersections,
        min_phase_steps=env.cfg.sim.phase_duration_min,
        max_phase_steps=env.cfg.sim.phase_duration_max,
    )
    obs = env.reset(seed=seed)
    agent.reset()
    done = False
    while not done:
        ta = agent.act(obs)
        obs, _, done, _ = env.step(ta.phase_indices)
    score = grader.grade(env.trajectory)
    if verbose:
        print(f"  [Baseline] task={task_id} seed={seed} score={score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Diagnostic episode runner
# ---------------------------------------------------------------------------

def run_episode_with_diagnostics(
    task_id: str,
    episode: int,
    seed: int,
    agent=None,
    grader=None,
    verbose: bool = True,
    quiet: bool = False,
) -> Tuple[float, float, int]:
    """
    Run one episode with full diagnostic logging.

    Returns (grader_score, total_reward, n_steps).
    """
    make_fn, grader_cls = _TASK_REGISTRY[task_id]
    env = make_fn(seed=seed)
    if grader is None:
        grader = grader_cls()

    # Rule-based fallback if no LLM agent
    rb_agent = None
    if agent is None:
        rb_agent = RuleBasedAgent(
            n_intersections=env.n_intersections,
            min_phase_steps=env.cfg.sim.phase_duration_min,
            max_phase_steps=env.cfg.sim.phase_duration_max,
        )

    obs = env.reset(seed=seed)
    if agent is not None:
        agent.reset(episode=episode)
    if rb_agent is not None:
        rb_agent.reset()

    # --- Log reset state ---
    init_meta = obs.metadata
    sig = _obs_signature(init_meta)
    print(
        f"\n{'─'*60}",
        flush=True,
    )
    print(
        f"  [EPISODE {episode}] task={task_id.upper()} seed={seed}",
        flush=True,
    )
    print(
        f"  Reset state sig={sig}  {_obs_summary(init_meta)}",
        flush=True,
    )

    done = False
    step = 0
    total_reward = 0.0
    prev_sig = sig
    prev_action: Optional[List[int]] = None
    repeated_obs_count = 0
    repeated_action_count = 0
    anti_stuck_count = 0  # we track via action vs. LLM suggestion divergence

    while not done:
        # Choose action
        if agent is not None:
            action = agent.act(obs, step=step)
        else:
            ta = rb_agent.act(obs)
            action = ta.phase_indices

        obs, reward, done, info = env.step(action)
        if agent is not None:
            agent.record_reward(reward, step=step)

        total_reward += reward
        step += 1

        # --- Per-step diagnostics ---
        cur_meta = obs.metadata
        cur_sig  = _obs_signature(cur_meta)

        obs_changed   = (cur_sig != prev_sig)
        action_changed = (prev_action is not None and action != prev_action)

        if not obs_changed:
            repeated_obs_count += 1
        if prev_action is not None and not action_changed:
            repeated_action_count += 1

        if not quiet and verbose:
            # Compact per-step log
            obs_flag = "" if obs_changed else " ⚠OBS_REPEAT"
            act_flag = "" if (prev_action is None or action_changed) else " ⚠ACT_REPEAT"
            print(
                f"    step={step:>3}  action={action}  reward={reward:+.4f}"
                f"  sig={cur_sig}{obs_flag}{act_flag}",
                flush=True,
            )
            if not obs_changed and step % 10 == 0:
                print(
                    f"    ⚠ OBS UNCHANGED for step {step}: {_obs_summary(cur_meta)}",
                    flush=True,
                )

        prev_sig    = cur_sig
        prev_action = action

    # Grade
    grader_score = grader.grade(env.trajectory)

    if agent is not None:
        agent.end_episode(
            total_reward=total_reward,
            grader_score=grader_score,
            n_steps=step,
        )

    print(
        f"  [EPISODE {episode}] done  steps={step}  "
        f"score={grader_score:.4f}  reward={total_reward:.3f}  "
        f"obs_repeats={repeated_obs_count}  act_repeats={repeated_action_count}",
        flush=True,
    )

    return grader_score, total_reward, step


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(
    task_id: str,
    n_episodes: int,
    base_seed: int,
    use_llm: bool,
    verbose: bool,
    quiet: bool,
    skip_baseline: bool,
) -> Dict:
    """
    Run full evaluation for one task difficulty.
    Returns a result dict with all scores and stats.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"  TASK: {task_id.upper()}   episodes={n_episodes}   seed_base={base_seed}", flush=True)
    print(f"{'='*60}", flush=True)

    # --- Baseline ---
    baseline_score: Optional[float] = None
    if not skip_baseline:
        print("  Running rule-based baseline...", flush=True)
        baseline_score = _run_baseline(task_id, seed=base_seed, verbose=verbose)
        _DIFFICULTY_BASELINES[task_id] = baseline_score
        print(f"  Baseline score: {baseline_score:.4f}", flush=True)
        print("─" * 60, flush=True)

    # --- LLM Agent setup ---
    agent = None
    if use_llm:
        try:
            from llm_agent.agent import LLMAgent
            from llm_agent.memory import AgentMemory
            from llm_agent.llm_adapter import build_adapter

            # Fresh memory per task — don't bleed lessons across difficulty levels
            memory  = AgentMemory()
            adapter = build_adapter(verbose=verbose)

            make_fn, _ = _TASK_REGISTRY[task_id]
            _tmp_env = make_fn(seed=base_seed)

            agent = LLMAgent(
                n_intersections=_tmp_env.n_intersections,
                memory=memory,
                verbose=verbose,
                adapter=adapter,
            )
        except Exception as exc:
            print(f"  [WARN] Could not build LLM agent: {exc}. Using rule-based fallback.", flush=True)

    # --- Episode loop ---
    scores: List[float] = []
    rewards: List[float] = []

    _, grader_cls = _TASK_REGISTRY[task_id]
    grader = grader_cls()

    for ep in range(1, n_episodes + 1):
        ep_seed = base_seed + ep   # vary seed per episode for diversity
        score, reward, n_steps = run_episode_with_diagnostics(
            task_id=task_id,
            episode=ep,
            seed=ep_seed,
            agent=agent,
            grader=grader,
            verbose=verbose,
            quiet=quiet,
        )
        scores.append(score)
        rewards.append(reward)

    # --- Per-task summary ---
    best_score  = max(scores)
    final_score = scores[-1]
    avg_score   = float(np.mean(scores))

    print(f"\n{'─'*60}", flush=True)
    print(f"  TASK SUMMARY: {task_id.upper()}", flush=True)
    print(f"  Episodes:      {n_episodes}", flush=True)
    print(f"  Per-episode:   {[round(s, 4) for s in scores]}", flush=True)
    print(f"  Best score:    {best_score:.4f}", flush=True)
    print(f"  Final score:   {final_score:.4f}", flush=True)
    print(f"  Avg score:     {avg_score:.4f}", flush=True)
    if baseline_score is not None:
        delta = final_score - baseline_score
        verdict = "BETTER" if delta >= 0 else "WORSE"
        print(
            f"  vs Baseline:   {delta:+.4f}  ({verdict})",
            flush=True,
        )
    print(f"{'─'*60}", flush=True)

    return {
        "task_id":      task_id,
        "n_episodes":   n_episodes,
        "scores":       scores,
        "best_score":   best_score,
        "final_score":  final_score,
        "avg_score":    avg_score,
        "baseline":     baseline_score,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run EASY, MEDIUM, and HARD traffic tasks with full diagnostics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=["easy", "medium", "hard"],
        default=["easy", "medium", "hard"],
        help="Which tasks to run (default: all three)",
    )
    p.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes per task (default: 3)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (episode i uses seed+i for variety, default: 42)",
    )
    p.add_argument(
        "--fallback-only", action="store_true",
        help="Skip LLM calls, run purely with rule-based agent",
    )
    p.add_argument(
        "--no-baseline", action="store_true",
        help="Skip rule-based baseline measurement",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step logs (only show episode summaries)",
    )
    p.add_argument(
        "--skip-health-check", action="store_true",
        help="Skip LLM connectivity check before running",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Determine if LLM should be used
    _grader_url = os.environ.get("API_BASE_URL", "").strip()
    _grader_key = os.environ.get("API_KEY", "").strip()
    use_llm = bool(_grader_url and _grader_key) or not args.fallback_only

    if args.fallback_only:
        print("🔄 FALLBACK-ONLY mode — no LLM calls.", flush=True)
        use_llm = False

    # LLM health check
    if use_llm and not args.skip_health_check:
        try:
            from llm_agent.agent import llm_health_check
            from llm_agent.llm_adapter import build_adapter
            adapter = build_adapter(verbose=True)
            ok = llm_health_check(adapter=adapter, verbose=True)
            if not ok:
                print(
                    "\n[WARN] LLM health check failed. Use --fallback-only to skip LLM.\n"
                    "       Continuing with rule-based fallback for this run.\n",
                    flush=True,
                )
                use_llm = False
        except Exception as exc:
            print(f"[WARN] Could not perform health check: {exc}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  MULTI-TASK EVALUATION", flush=True)
    print(f"  Tasks:     {args.tasks}", flush=True)
    print(f"  Episodes:  {args.episodes} per task", flush=True)
    print(f"  Seed base: {args.seed}", flush=True)
    print(f"  LLM:       {'YES' if use_llm else 'NO (rule-based fallback)'}", flush=True)
    print("=" * 60, flush=True)

    all_results: Dict[str, Dict] = {}
    t0 = time.time()

    for task_id in args.tasks:
        result = evaluate_task(
            task_id=task_id,
            n_episodes=args.episodes,
            base_seed=args.seed,
            use_llm=use_llm,
            verbose=not args.quiet,
            quiet=args.quiet,
            skip_baseline=args.no_baseline,
        )
        all_results[task_id] = result

    elapsed = time.time() - t0

    # --- Grand summary ---
    print("\n" + "=" * 60, flush=True)
    print("  FINAL RESULTS — ALL TASKS", flush=True)
    print("=" * 60, flush=True)
    print(
        f"  {'Task':<8}  {'Episodes':>8}  {'Best':>7}  {'Final':>7}  "
        f"{'Avg':>7}  {'Baseline':>9}  {'vs Base':>8}",
        flush=True,
    )
    print("  " + "─" * 60, flush=True)
    for tid, r in all_results.items():
        base_str = f"{r['baseline']:.4f}" if r["baseline"] is not None else "N/A"
        delta_str = ""
        if r["baseline"] is not None:
            delta = r["final_score"] - r["baseline"]
            delta_str = f"{delta:+.4f}"
        print(
            f"  {tid:<8}  {r['n_episodes']:>8}  {r['best_score']:>7.4f}  "
            f"{r['final_score']:>7.4f}  {r['avg_score']:>7.4f}  "
            f"{base_str:>9}  {delta_str:>8}",
            flush=True,
        )
    print(f"\n  Total runtime: {elapsed:.1f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

"""Aggregator — combines verifier, constraints, rubric, stability into final score.

The aggregator is the single source of truth for the score formula. It
preserves the exact same arithmetic as the original graders:

  raw     = W_PROCESS * process_score + W_OUTCOME * outcome_from_rubric
  capped  = soft_cap(raw, difficulty_ceil)    # medium / hard only
  final   = constraint_gate * capped

All scores are clamped to [0, 1]. If the verifier fails, final = 0.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from graders.verifier    import VerifierResult
from graders.constraints import ConstraintResult
from graders.rubric      import RubricMetrics
from graders.stability   import StabilityMetrics
from graders.explain     import Explanation, explain


# ---------------------------------------------------------------------------
# Result container (rich internal result; grade() just returns .final_score)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AggregationResult:
    """Full structured scoring result — for inspection, logging, testing."""
    verifier:      VerifierResult
    constraint:    ConstraintResult
    rubric:        RubricMetrics
    stability:     StabilityMetrics
    explanation:   Explanation
    process_score: float    # per-step action quality (0..1)
    raw_score:     float    # before gate and difficulty cap
    final_score:   float    # the value returned by grade()


# ---------------------------------------------------------------------------
# Weights (per-task, exactly matching existing graders)
# ---------------------------------------------------------------------------

# Weights kept in sync with easy_grader.py constants (W_PROCESS, W_TP, etc.)
# Sub-weights (w_tp + w_queue + w_improve + w_smooth) must sum to 1.0.
_EASY_WEIGHTS = dict(
    w_process=0.20,
    w_tp=0.55, w_queue=0.30, w_improve=0.10, w_smooth=0.05,
    w_spillback=0.0, w_fairness=0.0, w_emerg=0.0,
    difficulty_ceil=None,
)
_MEDIUM_WEIGHTS = dict(
    w_process=0.25,
    w_tp=0.38, w_queue=0.20, w_improve=0.0, w_smooth=0.06,
    w_spillback=0.26, w_fairness=0.10, w_emerg=0.0,
    difficulty_ceil=0.72,
)
_HARD_WEIGHTS = dict(
    w_process=0.35,
    w_tp=0.28, w_queue=0.0, w_improve=0.0, w_smooth=0.08,
    w_spillback=0.22, w_fairness=0.14, w_emerg=0.28,
    difficulty_ceil=0.58,
)

_TASK_WEIGHTS: Dict[str, Dict] = {
    "easy":   _EASY_WEIGHTS,
    "medium": _MEDIUM_WEIGHTS,
    "hard":   _HARD_WEIGHTS,
}


def _soft_cap(score: float, ceiling: float) -> float:
    if ceiling <= 0.0:
        return 0.0
    return ceiling * (1.0 - math.exp(-3.0 * score / ceiling))


def _clamp(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate(
    verifier:      VerifierResult,
    constraint:    ConstraintResult,
    rubric:        RubricMetrics,
    stability:     StabilityMetrics,
    process_score: float,
    task_id:       str,
) -> AggregationResult:
    """Combine all layers into a final AggregationResult.

    The .final_score field is identical to what grade() returns.
    """
    w = _TASK_WEIGHTS.get(task_id, _EASY_WEIGHTS)

    # --- Verifier gate ---
    if not verifier.passed:
        expl = explain(verifier, constraint, rubric, stability, 0.0, task_id)
        return AggregationResult(
            verifier=verifier,
            constraint=constraint,
            rubric=rubric,
            stability=stability,
            explanation=expl,
            process_score=process_score,
            raw_score=0.0,
            final_score=0.0,
        )

    # --- Outcome score from rubric ---
    outcome: float = (
        w["w_tp"]       * rubric.tp_score
        + w["w_queue"]    * rubric.queue_score
        + w["w_improve"]  * rubric.improvement_score
        + w["w_smooth"]   * rubric.smoothness_score
        + w["w_spillback"]* rubric.spillback_score
        + w["w_fairness"] * rubric.fairness_score
        + w["w_emerg"]    * rubric.emergency_quality_score
    )
    outcome = _clamp(outcome)

    # --- Raw combined score ---
    raw = w["w_process"] * process_score + (1.0 - w["w_process"]) * outcome
    raw = _clamp(raw)

    # --- Difficulty cap (medium / hard) ---
    ceil = w.get("difficulty_ceil")
    capped = _soft_cap(raw, ceil) if ceil is not None else raw

    # --- Constraint gate ---
    final = _clamp(constraint.gate * capped)

    expl = explain(verifier, constraint, rubric, stability, final, task_id)

    return AggregationResult(
        verifier=verifier,
        constraint=constraint,
        rubric=rubric,
        stability=stability,
        explanation=expl,
        process_score=process_score,
        raw_score=raw,
        final_score=final,
    )

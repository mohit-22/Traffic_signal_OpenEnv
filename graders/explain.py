"""Explainability reporter — deterministic, compact reason generation.

`explain()` takes all intermediate grading results and produces a human-readable
Explanation with a verdict, top contributing factors, violations, and a one-liner
recommendation. All logic is template-based: no LLM, no randomness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from graders.verifier    import VerifierResult
from graders.constraints import ConstraintResult
from graders.rubric      import RubricMetrics
from graders.stability   import StabilityMetrics


@dataclass(frozen=True)
class Explanation:
    score:          float
    verdict:        str              # "PASSED" | "FAILED_VERIFIER" | "PENALIZED" | "CLEAN"
    top_factors:    Tuple[str, ...]  # sorted by contribution, most important first
    violations:     Tuple[str, ...]  # constraint violation codes that fired
    recommendation: str              # one-liner actionable hint


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_verdict(
    verifier:    VerifierResult,
    constraint:  ConstraintResult,
    final_score: float,
) -> str:
    if not verifier.passed:
        return "FAILED_VERIFIER"
    if constraint.gate < 1.0:
        return "PENALIZED"
    if final_score >= 0.60:
        return "PASSED"
    return "CLEAN"


def _top_factors(
    rubric:    RubricMetrics,
    stability: StabilityMetrics,
    task_id:   str,
) -> List[Tuple[str, float]]:
    """Return (name, score) pairs sorted ascending (weakest first → most impactful to fix)."""
    factors: List[Tuple[str, float]] = []

    factors.append(("throughput",     rubric.tp_score))
    factors.append(("queue_pressure", rubric.queue_score))

    if task_id in ("medium", "hard"):
        factors.append(("spillback",  rubric.spillback_score))
        factors.append(("fairness",   rubric.fairness_score))

    if task_id == "hard":
        factors.append(("emergency_quality", rubric.emergency_quality_score))

    factors.append(("smoothness", rubric.smoothness_score))

    if stability.oscillation_rate > 0.30:
        factors.append(("oscillation_rate", 1.0 - stability.oscillation_rate))

    # Sort by score ascending — weakest first
    factors.sort(key=lambda x: x[1])
    return factors


def _recommendation(
    verifier:    VerifierResult,
    constraint:  ConstraintResult,
    rubric:      RubricMetrics,
    stability:   StabilityMetrics,
    task_id:     str,
) -> str:
    if not verifier.passed:
        return f"Fix degenerate policy: {verifier.reason}"

    viols = {v.code.value for v in constraint.violations}

    if "EMERGENCY_NEGLECT" in viols:
        return "Immediately serve emergency lanes — neglected events heavily penalise the score."
    if "ALL_RED_ABUSE" in viols:
        return "Stop choosing ALL_RED without justification — it blocks all traffic and triggers critical penalties."
    if "OSCILLATION_ABUSE" in viols:
        return "Hold phases longer — switching every step incurs heavy oscillation penalties."
    if "STARVATION_LOCK" in viols:
        return "Distribute green time across all intersections to prevent starvation."
    if "SPILLBACK_FLOOD" in viols:
        return "Reduce congestion at bottleneck intersections before it propagates grid-wide."

    # Rubric-driven hints
    if rubric.tp_score < 0.30:
        return "Increase throughput — serve dominant-queue lanes more aggressively."
    if rubric.emergency_quality_score < 0.50 and task_id == "hard":
        return "Respond faster to emergencies — aim for reaction within 4 steps for ambulances."
    if rubric.spillback_score < 0.40 and task_id in ("medium", "hard"):
        return "Address spillback propagation — coordinate neighbouring intersections to clear congestion."
    if rubric.fairness_score < 0.60 and task_id in ("medium", "hard"):
        return "Balance throughput across intersections — some are being starved of green time."
    if stability.oscillation_rate > 0.50:
        return "Reduce unnecessary phase switching for a more stable and higher-scoring policy."
    if stability.trend_slope < 0:
        return "Throughput is declining — avoid letting queues build unboundedly in later steps."

    return "Performance is within expected range; fine-tune congestion-pressure response."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain(
    verifier:    VerifierResult,
    constraint:  ConstraintResult,
    rubric:      RubricMetrics,
    stability:   StabilityMetrics,
    final_score: float,
    task_id:     str,
) -> Explanation:
    """Build a deterministic Explanation from all grading layers."""
    verdict = _classify_verdict(verifier, constraint, final_score)
    ranked  = _top_factors(rubric, stability, task_id)
    rec     = _recommendation(verifier, constraint, rubric, stability, task_id)

    top_factor_names = tuple(name for name, _ in ranked[:5])
    violation_codes  = tuple(v.code.value for v in constraint.violations)

    return Explanation(
        score=final_score,
        verdict=verdict,
        top_factors=top_factor_names,
        violations=violation_codes,
        recommendation=rec,
    )

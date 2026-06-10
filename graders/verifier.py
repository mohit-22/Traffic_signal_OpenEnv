"""Verifier layer — binary task-completion gating.

Separates "did the trajectory attempt and complete the task?" from
"how good was the trajectory quality?".

Rules
-----
* Verifier is binary: passed=True or passed=False.
* If passed=False the aggregator returns 0.0 (or a capped fallback).
* All checks are conservative: only fail on clearly degenerate runs
  that would score near-zero under the rubric anyway.
* All functions are pure (no side effects).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class VerifierResult:
    passed: bool
    reason: str          # human-readable; empty string when passed=True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _n_steps(trajectory: List[Dict]) -> int:
    return len(trajectory)


def _all_red_fraction(trajectory: List[Dict], n_intersections: int) -> float:
    """Fraction of (step × intersection) slots that were ALL_RED."""
    total = max(len(trajectory) * n_intersections, 1)
    count = 0
    for step in trajectory:
        for inter in step.get("state_snapshot", {}).get("intersections", []):
            if inter.get("phase", -1) == 2:
                count += 1
    return count / total


def _total_throughput(trajectory: List[Dict]) -> float:
    return sum(
        float(s.get("state_snapshot", {}).get("global_throughput", 0.0))
        for s in trajectory
    )


def _active_intersection_count(trajectory: List[Dict]) -> int:
    """Number of distinct intersection IDs that appear in the trajectory."""
    ids: set = set()
    for step in trajectory:
        for inter in step.get("state_snapshot", {}).get("intersections", []):
            ids.add(inter.get("id", 0))
    return len(ids)


def _emergency_neglect_fraction(trajectory: List[Dict]) -> float:
    """Fraction of logged emergency events that were NOT served."""
    last_snap = trajectory[-1].get("state_snapshot", {}) if trajectory else {}
    events: List[Dict] = last_snap.get("emergency_events", [])
    if not events:
        return 0.0
    neglected = sum(1 for ev in events if not ev.get("served", False))
    return neglected / max(len(events), 1)


# ---------------------------------------------------------------------------
# Per-task verifiers
# ---------------------------------------------------------------------------

def verify_easy(trajectory: List[Dict[str, Any]]) -> VerifierResult:
    """Verifier for Task 1 (Easy — single intersection, normal traffic).

    Fails only on clearly degenerate runs:
    1. Episode ran fewer than 5 steps (too short to grade meaningfully).
    2. ALL_RED was chosen for >95% of (step × intersection) slots
       (agent did literally nothing productive).
    3. Total throughput is exactly 0 for the entire episode AND the
       episode ran at least 10 steps (trivially broken policy).
    """
    n = _n_steps(trajectory)
    if n < 5:
        return VerifierResult(False, f"Episode too short: {n} steps (minimum 5)")

    ar = _all_red_fraction(trajectory, n_intersections=1)
    if ar > 0.95:
        return VerifierResult(
            False,
            f"Degenerate trajectory: ALL_RED fraction={ar:.2%} (threshold 95%)"
        )

    tp = _total_throughput(trajectory)
    if tp == 0.0 and n >= 10:
        return VerifierResult(
            False,
            "Zero total throughput over ≥10 steps — trivially broken policy"
        )

    return VerifierResult(True, "")


def verify_medium(trajectory: List[Dict[str, Any]]) -> VerifierResult:
    """Verifier for Task 2 (Medium — 4-intersection grid).

    Fails on:
    1. Episode ran fewer than 5 steps.
    2. ALL_RED >95% across all intersections.
    3. Fewer than 2 distinct intersections present in snapshots
       (indicates a badly misconfigured run, not a policy failure).
    4. Zero throughput for ≥15 steps (stricter: grid has more capacity).
    """
    n = _n_steps(trajectory)
    if n < 5:
        return VerifierResult(False, f"Episode too short: {n} steps (minimum 5)")

    ar = _all_red_fraction(trajectory, n_intersections=4)
    if ar > 0.95:
        return VerifierResult(
            False,
            f"Degenerate trajectory: ALL_RED fraction={ar:.2%} (threshold 95%)"
        )

    active = _active_intersection_count(trajectory)
    if active < 2:
        return VerifierResult(
            False,
            f"Only {active} intersection(s) in snapshots — expected ≥2 for medium task"
        )

    tp = _total_throughput(trajectory)
    if tp == 0.0 and n >= 15:
        return VerifierResult(
            False,
            "Zero total throughput over ≥15 steps on a 4-intersection grid"
        )

    return VerifierResult(True, "")


def verify_hard(trajectory: List[Dict[str, Any]]) -> VerifierResult:
    """Verifier for Task 3 (Hard — emergency + partial obs + weather).

    Fails on:
    1. Episode ran fewer than 5 steps.
    2. ALL_RED >95% across all intersections.
    3. 100% of documented emergency events were neglected AND there
       were ≥3 such events (single-event noise is tolerated).
    """
    n = _n_steps(trajectory)
    if n < 5:
        return VerifierResult(False, f"Episode too short: {n} steps (minimum 5)")

    ar = _all_red_fraction(trajectory, n_intersections=4)
    if ar > 0.95:
        return VerifierResult(
            False,
            f"Degenerate trajectory: ALL_RED fraction={ar:.2%} (threshold 95%)"
        )

    # Check emergency log if present
    last_snap = trajectory[-1].get("state_snapshot", {}) if trajectory else {}
    events: List[Dict] = last_snap.get("emergency_events", [])
    if len(events) >= 3:
        neglect_frac = _emergency_neglect_fraction(trajectory)
        if neglect_frac >= 1.0:
            return VerifierResult(
                False,
                f"All {len(events)} emergency events neglected — "
                "agent never served an emergency vehicle"
            )

    return VerifierResult(True, "")

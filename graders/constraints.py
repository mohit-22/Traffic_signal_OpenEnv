"""Constraint layer — hard rule violations with structured codes and gates.

Each constraint maps to a ViolationCode. Violations produce a gate_factor
in (0, 1] which multiplies into the final score. The combined gate is the
product of all individual gate_factors, floored at MIN_GATE.

The thresholds here are copied VERBATIM from the existing grader gate logic
so that existing trajectories produce the same final score.

All functions are pure (no side effects, no randomness).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Violation taxonomy
# ---------------------------------------------------------------------------

class ViolationCode(str, Enum):
    EMERGENCY_NEGLECT  = "EMERGENCY_NEGLECT"   # emerg. event expired unserved
    ALL_RED_ABUSE      = "ALL_RED_ABUSE"        # pathological ALL_RED fraction
    PHASE_LOCK         = "PHASE_LOCK"           # >92% same phase (lock-in)
    OSCILLATION_ABUSE  = "OSCILLATION_ABUSE"   # >70% switch rate
    STARVATION_LOCK    = "STARVATION_LOCK"      # intersection(s) phase-starved
    SPILLBACK_FLOOD    = "SPILLBACK_FLOOD"      # mean spillback >75%
    NO_OP_LOOP         = "NO_OP_LOOP"           # reserved; not yet triggered


SEVERITY_GATE: Dict[str, float] = {
    "critical": 0.20,
    "major":    0.55,
    "minor":    0.80,
}

MIN_GATE = 0.05  # absolute floor for combined gate


@dataclass(frozen=True)
class ConstraintViolation:
    code: ViolationCode
    severity: str           # "critical" | "major" | "minor"
    detail: str             # human-readable description
    gate_factor: float      # in (0, 1]; 1.0 = no penalty


@dataclass(frozen=True)
class ConstraintResult:
    violations: Tuple[ConstraintViolation, ...]
    gate: float             # combined multiplicative gate, in [MIN_GATE, 1.0]
    summary: str            # comma-joined violation codes, or "OK"


# ---------------------------------------------------------------------------
# Raw metric helpers (pure)
# ---------------------------------------------------------------------------

def _all_red_rate(trajectory: List[Dict], n_intersections: int) -> float:
    total = max(len(trajectory) * n_intersections, 1)
    count = 0
    for step in trajectory:
        for inter in step.get("state_snapshot", {}).get("intersections", []):
            if inter.get("phase", -1) == 2:
                count += 1
    return count / total


def _oscillation_rate(trajectory: List[Dict]) -> float:
    if len(trajectory) < 2:
        return 0.0
    switches = 0
    prev = None
    for step in trajectory:
        phases = tuple(
            i.get("phase", -1)
            for i in step.get("state_snapshot", {}).get("intersections", [])
        )
        if prev is not None and phases != prev:
            switches += 1
        prev = phases
    return switches / max(len(trajectory) - 1, 1)


def _dominant_phase_fraction(trajectory: List[Dict]) -> float:
    counts: Dict[int, int] = {}
    for step in trajectory:
        for inter in step.get("state_snapshot", {}).get("intersections", []):
            ph = inter.get("phase", -1)
            counts[ph] = counts.get(ph, 0) + 1
    total = max(sum(counts.values()), 1)
    return max(counts.values()) / total if counts else 0.0


def _starvation_fraction(trajectory: List[Dict], n_intersections: int) -> float:
    phase_counts: Dict[int, Dict[int, int]] = {}
    for step in trajectory:
        for inter in step.get("state_snapshot", {}).get("intersections", []):
            iid = inter.get("id", 0)
            ph  = inter.get("phase", -1)
            if iid not in phase_counts:
                phase_counts[iid] = {}
            phase_counts[iid][ph] = phase_counts[iid].get(ph, 0) + 1
    if not phase_counts:
        return 0.0
    starved = sum(
        1 for cnts in phase_counts.values()
        if max(cnts.values()) / max(sum(cnts.values()), 1) > 0.85
    )
    return starved / max(len(phase_counts), 1)


def _mean_spillback_rate(trajectory: List[Dict]) -> float:
    rates: List[float] = []
    for step in trajectory:
        inter_list = step.get("state_snapshot", {}).get("intersections", [])
        if not inter_list:
            continue
        spills = sum(1 for i in inter_list if i.get("spillback", 0) > 0)
        rates.append(spills / max(len(inter_list), 1))
    return statistics.mean(rates) if rates else 0.0


def _emergency_neglect_count(trajectory: List[Dict]) -> int:
    last_snap = trajectory[-1].get("state_snapshot", {}) if trajectory else {}
    events: List[Dict] = last_snap.get("emergency_events", [])
    return sum(1 for ev in events if not ev.get("served", False))


# ---------------------------------------------------------------------------
# Per-rule constraint checkers
# ---------------------------------------------------------------------------

def _check_all_red(
    all_red: float,
    thresholds: Tuple[float, float, float],
    gates: Tuple[float, float, float],
) -> Optional[ConstraintViolation]:
    """Generic ALL_RED check; thresholds = (critical_lo, major_lo, minor_lo)."""
    crit_lo, maj_lo, min_lo = thresholds
    crit_gate, maj_gate, min_gate = gates
    if all_red > crit_lo:
        detail = f"ALL_RED fraction={all_red:.2%} exceeds critical threshold {crit_lo:.0%}"
        return ConstraintViolation(ViolationCode.ALL_RED_ABUSE, "critical", detail, crit_gate)
    if all_red > maj_lo:
        detail = f"ALL_RED fraction={all_red:.2%} exceeds major threshold {maj_lo:.0%}"
        return ConstraintViolation(ViolationCode.ALL_RED_ABUSE, "major", detail, maj_gate)
    if all_red > min_lo:
        detail = f"ALL_RED fraction={all_red:.2%} (minor)"
        return ConstraintViolation(ViolationCode.ALL_RED_ABUSE, "minor", detail, min_gate)
    return None


def _check_oscillation(osc: float) -> Optional[ConstraintViolation]:
    if osc > 0.70:
        return ConstraintViolation(
            ViolationCode.OSCILLATION_ABUSE, "major",
            f"Phase switch rate={osc:.2%} (>70%)", 0.50
        )
    if osc > 0.50:
        return ConstraintViolation(
            ViolationCode.OSCILLATION_ABUSE, "minor",
            f"Phase switch rate={osc:.2%} (>50%)", 0.75
        )
    return None


def _check_phase_lock(dom_frac: float, threshold: float, gate: float) -> Optional[ConstraintViolation]:
    if dom_frac > threshold:
        return ConstraintViolation(
            ViolationCode.PHASE_LOCK, "major",
            f"Dominant phase fraction={dom_frac:.2%} (>{threshold:.0%})", gate
        )
    return None


def _check_starvation(
    starv: float, n_intersections: int, per_starved_penalty: float, floor: float
) -> Optional[ConstraintViolation]:
    if starv > 0.0:
        n_starved = int(round(starv * n_intersections))
        gf = max(floor, 1.0 - per_starved_penalty * n_starved)
        return ConstraintViolation(
            ViolationCode.STARVATION_LOCK, "major",
            f"{n_starved}/{n_intersections} intersection(s) phase-starved (>85% dominant phase)",
            gf,
        )
    return None


def _check_spillback(spill: float) -> Optional[ConstraintViolation]:
    if spill > 0.75:
        return ConstraintViolation(
            ViolationCode.SPILLBACK_FLOOD, "major",
            f"Mean spillback rate={spill:.2%} (>75%)", 0.60
        )
    if spill > 0.55:
        return ConstraintViolation(
            ViolationCode.SPILLBACK_FLOOD, "minor",
            f"Mean spillback rate={spill:.2%} (>55%)", 0.80
        )
    return None


def _check_emergency_neglect(neglect_count: int) -> Optional[ConstraintViolation]:
    if neglect_count > 0:
        gate = max(0.15, 1.0 - 0.40 * neglect_count)
        return ConstraintViolation(
            ViolationCode.EMERGENCY_NEGLECT, "critical",
            f"{neglect_count} emergency event(s) expired without service",
            gate,
        )
    return None


# ---------------------------------------------------------------------------
# Anti-exploit phase-lock (shared across tasks)
# ---------------------------------------------------------------------------

def _check_anti_exploit_phase_lock(dom_frac: float) -> Optional[ConstraintViolation]:
    if dom_frac > 0.92:
        return ConstraintViolation(
            ViolationCode.PHASE_LOCK, "major",
            f"Anti-exploit: dominant phase fraction={dom_frac:.2%} (>92%)", 0.55
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_constraints(
    trajectory: List[Dict[str, Any]],
    n_intersections: int,
    task_id: str,
) -> ConstraintResult:
    """Run all hard-rule checks for the given task.

    Returns a ConstraintResult whose `.gate` is already multiplied out and
    floored. The gate is conceptually: gate = product(v.gate_factor for v in violations).

    Thresholds are task-specific and match existing grader gate logic exactly.
    """
    if not trajectory:
        return ConstraintResult(violations=(), gate=1.0, summary="OK")

    all_red = _all_red_rate(trajectory, n_intersections)
    osc     = _oscillation_rate(trajectory)
    dom     = _dominant_phase_fraction(trajectory)
    starv   = _starvation_fraction(trajectory, n_intersections)
    spill   = _mean_spillback_rate(trajectory)

    viols: List[ConstraintViolation] = []

    if task_id == "easy":
        # --- Easy: single intersection ---
        v = _check_all_red(all_red, (0.60, 0.40, 0.20), (0.20, 0.55, 0.80))
        if v: viols.append(v)

        # Easy starvation: raised to 0.92 (natural balance for short episodes)
        if dom > 0.92:
            viols.append(ConstraintViolation(
                ViolationCode.PHASE_LOCK, "major",
                f"Dominant phase lock (easy): dom={dom:.2%}", 0.72
            ))

        v = _check_oscillation(osc)
        if v: viols.append(v)

        # Anti-exploit layer (matches _anti_exploit_penalty in BaseGrader)
        v = _check_anti_exploit_phase_lock(dom)
        if v: viols.append(v)

    elif task_id == "medium":
        # --- Medium: 4-intersection grid ---
        v = _check_all_red(all_red, (0.50, 0.30, 0.15), (0.20, 0.55, 0.80))
        if v: viols.append(v)

        v = _check_starvation(starv, n_intersections, 0.15, 0.40)
        if v: viols.append(v)

        v = _check_spillback(spill)
        if v: viols.append(v)

        v = _check_oscillation(osc)
        if v: viols.append(v)

        v = _check_anti_exploit_phase_lock(dom)
        if v: viols.append(v)

    elif task_id == "hard":
        # --- Hard: emergency + partial obs ---
        neglect = _emergency_neglect_count(trajectory)
        v = _check_emergency_neglect(neglect)
        if v: viols.append(v)

        v = _check_all_red(all_red, (0.50, 0.30, 0.15), (0.20, 0.55, 0.80))
        if v: viols.append(v)

        v = _check_starvation(starv, n_intersections, 0.18, 0.35)
        if v: viols.append(v)

        # Hard spillback: slightly different gates than medium
        if spill > 0.75:
            viols.append(ConstraintViolation(
                ViolationCode.SPILLBACK_FLOOD, "major",
                f"Mean spillback rate={spill:.2%} (>75%)", 0.65
            ))
        elif spill > 0.55:
            viols.append(ConstraintViolation(
                ViolationCode.SPILLBACK_FLOOD, "minor",
                f"Mean spillback rate={spill:.2%} (>55%)", 0.82
            ))

        v = _check_oscillation(osc)
        if v: viols.append(v)

        v = _check_anti_exploit_phase_lock(dom)
        if v: viols.append(v)

    else:
        # Unknown task — apply conservative defaults
        v = _check_all_red(all_red, (0.60, 0.40, 0.20), (0.20, 0.55, 0.80))
        if v: viols.append(v)
        v = _check_oscillation(osc)
        if v: viols.append(v)

    # Combine gate (product, floored)
    gate = 1.0
    for v in viols:
        gate *= v.gate_factor
    gate = float(max(MIN_GATE, min(1.0, gate)))

    summary = ", ".join(v.code.value for v in viols) if viols else "OK"
    return ConstraintResult(violations=tuple(viols), gate=gate, summary=summary)

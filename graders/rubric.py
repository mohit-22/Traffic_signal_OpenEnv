"""Rubric scorer — pure metric extraction functions shared by all graders.

Each `compute_rubric_*` function extracts calibrated metric scores from a
trajectory and returns a RubricMetrics dataclass. The aggregator then
applies per-task weights.

All functions are pure (no side effects).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RubricMetrics:
    """Normalised [0,1] scores for each rubric dimension."""
    tp_score:               float   # throughput (higher = better)
    queue_score:            float   # worst-queue pressure (higher = better)
    spillback_score:        float   # spillback avoidance (higher = better)
    fairness_score:         float   # Jain's fairness index (higher = better)
    smoothness_score:       float   # anti-oscillation (higher = better)
    improvement_score:      float   # late-vs-early throughput improvement
    emergency_quality_score: float  # emergency service quality (1.0 if no events)


# ---------------------------------------------------------------------------
# Shared pure normalisation helpers
# ---------------------------------------------------------------------------

def _get_bounds(
    calibration: Optional[Dict[str, Tuple[float, float]]],
    key: str,
    default_lo: float,
    default_hi: float,
) -> Tuple[float, float]:
    if calibration and key in calibration:
        lo, hi = calibration[key]
        if hi > lo:
            return lo, hi
    return default_lo, default_hi


def _normalise(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _invert(score: float) -> float:
    return 1.0 - max(0.0, min(1.0, score))


def _safe_mean(vals: List[float], default: float = 0.0) -> float:
    return statistics.mean(vals) if vals else default


def _robust_mean(vals: List[float], default: float = 0.0) -> float:
    """Trimmed mean with winsorization — identical to BaseGrader._robust_mean."""
    if not vals:
        return default
    # Winsorize
    if len(vals) >= 4:
        try:
            import numpy as np
            lo = float(np.percentile(vals, 5.0))
            hi = float(np.percentile(vals, 95.0))
            ws = [max(lo, min(hi, v)) for v in vals]
        except ImportError:
            ws = vals
    else:
        ws = vals
    # Trimmed mean (10%)
    n = len(ws)
    k = max(0, int(n * 0.10))
    sdata = sorted(ws)
    trimmed = sdata[k: n - k] if k > 0 else sdata
    return float(statistics.mean(trimmed)) if trimmed else float(statistics.mean(sdata))


def _jains_fairness(trajectory: List[Dict]) -> float:
    inter_totals: Dict[int, float] = {}
    for step in trajectory:
        for inter in step.get("state_snapshot", {}).get("intersections", []):
            iid = inter.get("id", 0)
            itp = float(inter.get("throughput", 0.0))
            inter_totals[iid] = inter_totals.get(iid, 0.0) + itp
    if len(inter_totals) <= 1:
        return 1.0
    vals = list(inter_totals.values())
    s  = sum(vals)
    sq = sum(v * v for v in vals)
    n  = len(vals)
    if sq == 0:
        return 0.5
    return float(min(1.0, (s * s) / (n * sq)))


# ---------------------------------------------------------------------------
# Emergency quality (shared between hard_grader and rubric)
# ---------------------------------------------------------------------------

# Response thresholds: (ideal_steps, bad_steps) per emergency type
_EMERG_THRESHOLDS: Dict[int, Tuple[int, int]] = {
    3: (4,  15),   # Ambulance
    2: (6,  20),   # Fire
    1: (8,  28),   # Police
}


def _resp_score(latency_steps: int, em_type: int) -> float:
    ideal, bad = _EMERG_THRESHOLDS.get(em_type, (8, 28))
    if latency_steps <= ideal:
        return 1.0
    if latency_steps >= bad:
        return 0.0
    t = (latency_steps - ideal) / max(bad - ideal, 1)
    return float(math.exp(-3.0 * t))


def _compute_emergency_quality(trajectory: List[Dict]) -> Tuple[float, int]:
    """Returns (mean_em_score, neglect_count). Identical to HardGrader logic."""
    last_snap = trajectory[-1].get("state_snapshot", {}) if trajectory else {}
    explicit_events: List[Dict] = last_snap.get("emergency_events", [])

    em_scores: List[float] = []
    neglect_count = 0

    if explicit_events:
        for ev in explicit_events:
            latency = int(ev.get("latency_steps", 999))
            etype   = int(ev.get("etype", 0))
            served  = bool(ev.get("served", False))
            if served:
                em_scores.append(_resp_score(latency, etype))
            else:
                em_scores.append(0.0)
                neglect_count += 1
    else:
        # Backward-compatible state-machine reconstruction
        em_state: Dict[str, Dict] = {}
        for step_data in trajectory:
            snap   = step_data.get("state_snapshot", {})
            s_step = int(snap.get("step", 0))
            for inter in snap.get("intersections", []):
                iid      = inter.get("id", 0)
                em_type  = int(inter.get("emergency", 0))
                em_lane  = int(inter.get("emergency_lane", -1))
                phase    = int(inter.get("phase", 2))
                key      = f"{iid}_em"
                green_lanes = {0: [0, 1], 1: [2, 3]}.get(phase, [])
                if em_type > 0:
                    if key not in em_state:
                        em_state[key] = {"arrival": s_step, "em_type": em_type, "done": False}
                    info = em_state[key]
                    if not info["done"] and em_lane in green_lanes:
                        em_scores.append(_resp_score(s_step - info["arrival"], em_type))
                        info["done"] = True
                elif key in em_state:
                    info = em_state.pop(key)
                    if not info["done"]:
                        em_scores.append(0.0)
                        neglect_count += 1
        for info in em_state.values():
            if not info["done"]:
                em_scores.append(0.0)
                neglect_count += 1

    quality = float(statistics.mean(em_scores)) if em_scores else 1.0
    return quality, neglect_count


# ---------------------------------------------------------------------------
# Per-task rubric computers
# ---------------------------------------------------------------------------

# Default calibration bounds (matching existing grader constants)
_EASY_DEFAULTS = dict(
    tp_lo=0.0, tp_hi=1.5,
    queue_lo=0.0, queue_hi=8.0,
    switch_hi=0.25,
)
_MEDIUM_DEFAULTS = dict(
    tp_lo=0.0, tp_hi=10.0,
    queue_lo=0.0, queue_hi=8.0,
    spillback_hi=0.75,
    switch_hi=0.30,
)
_HARD_DEFAULTS = dict(
    tp_lo=0.0, tp_hi=14.0,
    queue_lo=0.0, queue_hi=8.0,
    spillback_hi=0.80,
    switch_hi=0.30,
)


def compute_rubric_easy(
    trajectory: List[Dict[str, Any]],
    calibration: Optional[Dict[str, Tuple[float, float]]] = None,
) -> RubricMetrics:
    n_steps = len(trajectory)
    throughputs:  List[float] = []
    worst_queues: List[float] = []

    tp_lo, tp_hi     = _get_bounds(calibration, "tp",    _EASY_DEFAULTS["tp_lo"], _EASY_DEFAULTS["tp_hi"])
    q_lo,  q_hi      = _get_bounds(calibration, "queue", _EASY_DEFAULTS["queue_lo"], _EASY_DEFAULTS["queue_hi"])
    switch_hi        = _EASY_DEFAULTS["switch_hi"]

    for step in trajectory:
        snap = step.get("state_snapshot", {})
        throughputs.append(float(snap.get("global_throughput", 0.0)))
        lane_qs: List[float] = []
        for inter in snap.get("intersections", []):
            for q in inter.get("queues", []):
                lane_qs.append(float(q))
        worst_queues.append(max(lane_qs) if lane_qs else q_hi)

    tp_score    = _normalise(_robust_mean(throughputs, 0.0), tp_lo, tp_hi)
    queue_score = _invert(_normalise(_robust_mean(worst_queues, q_hi), q_lo, q_hi))

    # Improvement score: skip first n//3 warm-up steps; compare remaining windows.
    # Mirrors the easy_grader.py logic exactly so grade_detailed() matches grade().
    warmup    = max(n_steps // 3, 1)
    rest      = throughputs[warmup:]
    half2     = max(len(rest) // 2, 1)
    early_tp  = _safe_mean(rest[:half2]) if rest else 0.0
    late_tp   = _safe_mean(rest[half2:]) if len(rest) > half2 else early_tp
    improve_score = max(0.0, min(1.0,
        (late_tp - early_tp) / max(tp_hi - tp_lo, 1e-6) + 0.5
    ))

    # Smoothness: windowed over last min(5, n_steps) steps.
    # Mirrors easy_grader.py to keep grade_detailed() consistent with grade().
    window_size    = min(5, n_steps)
    window_steps   = trajectory[-window_size:]
    prev_ph        = None
    window_sw      = 0
    for sd in window_steps:
        ph_t = tuple(
            inter.get("phase", -1)
            for inter in sd.get("state_snapshot", {}).get("intersections", [])
        )
        if prev_ph is not None and ph_t != prev_ph:
            window_sw += 1
        prev_ph = ph_t
    window_rate  = window_sw / max(window_size - 1, 1)
    smooth_score = _invert(_normalise(window_rate, 0.0, switch_hi))

    return RubricMetrics(
        tp_score=tp_score,
        queue_score=queue_score,
        spillback_score=1.0,         # Easy has no spillback metric
        fairness_score=1.0,          # Easy: single intersection → trivially fair
        smoothness_score=smooth_score,
        improvement_score=improve_score,
        emergency_quality_score=1.0, # Easy has no emergencies
    )


def compute_rubric_medium(
    trajectory: List[Dict[str, Any]],
    calibration: Optional[Dict[str, Tuple[float, float]]] = None,
) -> RubricMetrics:
    n_steps     = len(trajectory)
    n_inters    = 4
    throughputs: List[float] = []
    worst_queues: List[float] = []

    tp_lo, tp_hi   = _get_bounds(calibration, "tp",       _MEDIUM_DEFAULTS["tp_lo"], _MEDIUM_DEFAULTS["tp_hi"])
    q_lo,  q_hi    = _get_bounds(calibration, "queue",    _MEDIUM_DEFAULTS["queue_lo"], _MEDIUM_DEFAULTS["queue_hi"])
    sp_lo, sp_hi   = _get_bounds(calibration, "spillback", 0.0, _MEDIUM_DEFAULTS["spillback_hi"])
    switch_hi      = _MEDIUM_DEFAULTS["switch_hi"]

    # Mean spillback rate (episode-level)
    spill_rates: List[float] = []
    for step in trajectory:
        snap   = step.get("state_snapshot", {})
        inters = snap.get("intersections", [])
        throughputs.append(float(snap.get("global_throughput", 0.0)))
        step_worst = 0.0
        step_spills = 0
        for inter in inters:
            qs = inter.get("queues", [])
            if qs:
                step_worst = max(step_worst, max(float(q) for q in qs))
            if inter.get("spillback", 0) > 0:
                step_spills += 1
        worst_queues.append(step_worst)
        spill_rates.append(step_spills / max(len(inters), 1))
    spill = _safe_mean(spill_rates)

    tp_score       = _normalise(_robust_mean(throughputs, 0.0), tp_lo, tp_hi)
    queue_score    = _invert(_normalise(_robust_mean(worst_queues, q_hi), q_lo, q_hi))
    spillback_score = _invert(_normalise(spill, sp_lo, sp_hi))
    fairness_score = _jains_fairness(trajectory)

    final_sw    = int(trajectory[-1].get("state_snapshot", {}).get("phase_switches", 0))
    switch_rate = final_sw / max(n_steps * n_inters, 1)
    smooth_score = _invert(_normalise(switch_rate, 0.0, switch_hi))

    return RubricMetrics(
        tp_score=tp_score,
        queue_score=queue_score,
        spillback_score=spillback_score,
        fairness_score=fairness_score,
        smoothness_score=smooth_score,
        improvement_score=0.5,       # Medium doesn't use improvement score
        emergency_quality_score=1.0, # Medium has no emergencies
    )


def compute_rubric_hard(
    trajectory: List[Dict[str, Any]],
    calibration: Optional[Dict[str, Tuple[float, float]]] = None,
) -> RubricMetrics:
    n_steps  = len(trajectory)
    n_inters = 4
    throughputs:  List[float] = []
    worst_queues: List[float] = []
    spill_rates:  List[float] = []

    tp_lo, tp_hi   = _get_bounds(calibration, "tp",       _HARD_DEFAULTS["tp_lo"], _HARD_DEFAULTS["tp_hi"])
    q_lo,  q_hi    = _get_bounds(calibration, "queue",    _HARD_DEFAULTS["queue_lo"], _HARD_DEFAULTS["queue_hi"])
    sp_lo, sp_hi   = _get_bounds(calibration, "spillback", 0.0, _HARD_DEFAULTS["spillback_hi"])
    switch_hi      = _HARD_DEFAULTS["switch_hi"]

    for step in trajectory:
        snap   = step.get("state_snapshot", {})
        inters = snap.get("intersections", [])
        throughputs.append(float(snap.get("global_throughput", 0.0)))
        step_worst  = 0.0
        step_spills = 0
        for inter in inters:
            qs = inter.get("queues", [])
            if qs:
                step_worst = max(step_worst, max(float(q) for q in qs))
            if inter.get("spillback", 0) > 0:
                step_spills += 1
        worst_queues.append(step_worst)
        spill_rates.append(step_spills / max(len(inters), 1))

    spill = _safe_mean(spill_rates)
    em_quality, _ = _compute_emergency_quality(trajectory)

    tp_score        = _normalise(_robust_mean(throughputs, 0.0), tp_lo, tp_hi)
    queue_score     = _invert(_normalise(_robust_mean(worst_queues, q_hi), q_lo, q_hi))
    spillback_score = _invert(_normalise(spill, sp_lo, sp_hi))
    fairness_score  = _jains_fairness(trajectory)

    final_sw    = int(trajectory[-1].get("state_snapshot", {}).get("phase_switches", 0))
    switch_rate = final_sw / max(n_steps * n_inters, 1)
    smooth_score = _invert(_normalise(switch_rate, 0.0, switch_hi))

    return RubricMetrics(
        tp_score=tp_score,
        queue_score=queue_score,
        spillback_score=spillback_score,
        fairness_score=fairness_score,
        smoothness_score=smooth_score,
        improvement_score=0.5,
        emergency_quality_score=em_quality,
    )

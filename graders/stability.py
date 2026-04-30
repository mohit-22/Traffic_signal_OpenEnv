"""Stability and robustness metrics for episode trajectories.

All functions are pure (no side effects, no randomness).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class StabilityMetrics:
    oscillation_rate:            float   # fraction of steps where phase changed
    phase_churn:                 float   # phase switches / (steps × n_intersections)
    reward_variance:             float   # variance of per-step rewards
    tp_variance:                 float   # variance of per-step throughput
    trend_slope:                 float   # linear trend of throughput (positive = improving)
    plateau_detected:            bool    # last 20% of steps have near-flat throughput
    emergency_reaction_delay:    float   # mean served emergency latency (steps); -1 if none


# ---------------------------------------------------------------------------
# Pure helper: linear trend slope (least-squares on index)
# ---------------------------------------------------------------------------

def _linear_slope(values: List[float]) -> float:
    """Slope of best-fit line through (i, values[i])."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = statistics.mean(values)
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


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


def _phase_churn(trajectory: List[Dict], n_intersections: int) -> float:
    if not trajectory:
        return 0.0
    final_sw = int(
        trajectory[-1].get("state_snapshot", {}).get("phase_switches", 0)
    )
    return final_sw / max(len(trajectory) * n_intersections, 1)


def _emergency_mean_delay(trajectory: List[Dict]) -> float:
    """Mean served emergency latency in steps. Returns -1.0 if no events."""
    last_snap = trajectory[-1].get("state_snapshot", {}) if trajectory else {}
    events: List[Dict] = last_snap.get("emergency_events", [])
    served = [ev.get("latency_steps", 0) for ev in events if ev.get("served", False)]
    return float(statistics.mean(served)) if served else -1.0


def _plateau_detected(tp_values: List[float], window_frac: float = 0.20) -> bool:
    """True if the last `window_frac` of steps have near-flat throughput (IQR < 5% range)."""
    if len(tp_values) < 5:
        return False
    window = max(2, int(len(tp_values) * window_frac))
    tail = tp_values[-window:]
    srt = sorted(tail)
    q1, q3 = srt[len(srt) // 4], srt[(3 * len(srt)) // 4]
    full_range = max(max(tp_values) - min(tp_values), 1e-9)
    return (q3 - q1) / full_range < 0.05


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_stability(
    trajectory: List[Dict[str, Any]],
    n_intersections: int,
) -> StabilityMetrics:
    """Compute all stability metrics for an episode trajectory."""
    if not trajectory:
        return StabilityMetrics(
            oscillation_rate=0.0,
            phase_churn=0.0,
            reward_variance=0.0,
            tp_variance=0.0,
            trend_slope=0.0,
            plateau_detected=False,
            emergency_reaction_delay=-1.0,
        )

    rewards = [float(s.get("reward", 0.0)) for s in trajectory]
    tp_vals = [
        float(s.get("state_snapshot", {}).get("global_throughput", 0.0))
        for s in trajectory
    ]

    osc    = _oscillation_rate(trajectory)
    churn  = _phase_churn(trajectory, n_intersections)
    r_var  = float(statistics.variance(rewards)) if len(rewards) >= 2 else 0.0
    t_var  = float(statistics.variance(tp_vals)) if len(tp_vals) >= 2 else 0.0
    slope  = _linear_slope(tp_vals)
    plat   = _plateau_detected(tp_vals)
    em_del = _emergency_mean_delay(trajectory)

    return StabilityMetrics(
        oscillation_rate=osc,
        phase_churn=churn,
        reward_variance=r_var,
        tp_variance=t_var,
        trend_slope=slope,
        plateau_detected=plat,
        emergency_reaction_delay=em_del,
    )

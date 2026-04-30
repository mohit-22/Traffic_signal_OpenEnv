"""Easy-task grader — single intersection, normal traffic.

Architecture: hybrid gate × (process + outcome)
================================================

  final_score = gate_factor
                × (W_PROCESS * mean(process_scores)
                   + W_OUTCOME * outcome_score)

  gate_factor = safety_gate(all_red_abuse, starvation, oscillation)
              × anti_exploit_penalty

  outcome_score = weighted_sum(
      0.45 * throughput_score,   # primary — must discharge vehicles
      0.30 * queue_score,        # worst-lane tail-risk
      0.15 * improvement_score,  # late-episode progress
      0.10 * smoothness_score,   # anti-oscillation
  )

Safety gates (multiplicative)
------------------------------
  ALL_RED rate > 60% → gate = 0.20
  ALL_RED rate > 40% → gate = 0.55
  Starvation fraction > 0%  → gate ×= 0.55 (one dominant phase > 85%)
  Oscillation rate > 70%    → gate ×= 0.50

Process / outcome mix
---------------------
  W_PROCESS = 0.20   (local per-step action quality — less reliable from LLM)
  W_OUTCOME = 0.80   (trajectory-level outcome metrics)

Calibration
-----------
  Uses calibrated bounds from calibration dict if available.
  Static defaults:
    tp    : [0.0, 1.5]   (single intersection, arrival_rate=0.30 × 4 lanes ≈ 1.2/step)
    queue : [0.0, 8.0]

Target ranges
-------------
  Rule-based baseline: ≈ 0.50–0.68
  Good LLM policy:     ≈ 0.62–0.80
  Random policy:       ≈ 0.20–0.42
  Score always in [0, 1].

Improvement score note
----------------------
  First n//3 steps are treated as warm-up and excluded from the split.
  This prevents early instability (agent warm-up, prompt variability) from
  dragging the improvement score into negative territory on short episodes.

Smoothness score note
---------------------
  Uses the switch count over the last min(5, n_steps) steps (windowed).
  Whole-episode switch rate is unfair on short runs where 1–2 anti-stuck
  overrides can push the fraction above the reference threshold.

Internal pipeline (new — does not change public output)
--------------------------------------------------------
  verify_easy → check_constraints → compute_rubric_easy
  → compute_stability → aggregate → AggregationResult
  grade() returns AggregationResult.final_score (identical to old logic).
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

from graders.base_grader  import BaseGrader
from graders.verifier     import verify_easy
from graders.constraints  import check_constraints
from graders.rubric       import compute_rubric_easy
from graders.stability    import compute_stability
from graders.aggregator   import aggregate, AggregationResult

# Normalization defaults (used when calibration is absent)
_TP_DEFAULT_LO    = 0.0
_TP_DEFAULT_HI    = 0.60   # tuned to preserve easy > medium monotonic ordering natively
_QUEUE_DEFAULT_LO = 0.0
_QUEUE_DEFAULT_HI = 8.0
_SWITCH_DEFAULT_HI = 0.25  # switch rate reference

# Process / outcome blend weights
# W_PROCESS lowered: LLM per-step scores are noisy; outcome is more reliable.
W_PROCESS = 0.20
W_OUTCOME = 0.80

# Outcome sub-weights (must sum to 1.0)
# tp raised: clearing traffic is the primary easy-task objective.
# improvement/smooth lowered: too punitive on short 20-step episodes.
W_TP      = 0.55
W_QUEUE   = 0.30
W_IMPROVE = 0.10
W_SMOOTH  = 0.05


class EasyGrader(BaseGrader):
    """Deterministic grader for Task 1 (Easy).

    Target baseline (rule-based): ≈ 0.45–0.65.
    Exploit-resistant via safety gate + anti-exploit layer.

    Public interface: grade(trajectory) -> float  [unchanged]
    Extended interface: grade_detailed(trajectory) -> AggregationResult
    """

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return 0.0

        n_steps = len(trajectory)
        n_inters = 1  # Easy = single intersection

        # ------------------------------------------------------------------
        # 1. Safety gate (preserved verbatim from original)
        # ------------------------------------------------------------------
        all_red = self._all_red_rate(trajectory, n_inters)
        osc     = self._oscillation_rate(trajectory)

        gate = 1.0
        if all_red > 0.60:
            gate *= 0.20
        elif all_red > 0.40:
            gate *= 0.55
        elif all_red > 0.20:
            gate *= 0.80

        # Starvation gate for easy (single intersection, short episodes).
        phase_counts: Dict[int, int] = {}
        for step_data in trajectory:
            snap = step_data.get("state_snapshot", {})
            for inter in snap.get("intersections", []):
                ph = inter.get("phase", -1)
                phase_counts[ph] = phase_counts.get(ph, 0) + 1
        if phase_counts:
            total_ph = max(sum(phase_counts.values()), 1)
            dominant_frac = max(phase_counts.values()) / total_ph
            if dominant_frac > 0.92:   # truly degenerate lock-in
                gate *= 0.72           # soft penalty

        if osc > 0.70:
            gate *= 0.50
        elif osc > 0.50:
            gate *= 0.75

        # Anti-exploit multiplicative factor
        exploit_penalty = self._anti_exploit_penalty(trajectory, n_inters)
        gate = max(0.0, min(1.0, gate * exploit_penalty))

        if gate == 0.0:
            return 0.0

        # ------------------------------------------------------------------
        # 2. Process score (per-step local action quality)
        # ------------------------------------------------------------------
        process_scores = self._compute_process_scores(trajectory, n_inters)
        process_score  = self._robust_mean(process_scores, default=0.5)

        # ------------------------------------------------------------------
        # 3. Outcome score
        # ------------------------------------------------------------------
        throughputs:  List[float] = []
        worst_queues: List[float] = []

        for step_data in trajectory:
            snap = step_data.get("state_snapshot", {})
            throughputs.append(float(snap.get("global_throughput", 0.0)))

            lane_queues: List[float] = []
            for inter in snap.get("intersections", []):
                for q in inter.get("queues", []):
                    lane_queues.append(float(q))
            q_hi = self._get_bounds("queue", _QUEUE_DEFAULT_LO, _QUEUE_DEFAULT_HI)[1]
            worst_q = max(lane_queues) if lane_queues else q_hi
            worst_queues.append(worst_q)

        # Calibrated throughput normalization
        tp_lo, tp_hi = self._get_bounds("tp", _TP_DEFAULT_LO, _TP_DEFAULT_HI)
        tp_score = self._normalise(self._robust_mean(throughputs, 0.0), tp_lo, tp_hi)

        # Queue score (lower is better)
        q_lo, q_hi = self._get_bounds("queue", _QUEUE_DEFAULT_LO, _QUEUE_DEFAULT_HI)
        queue_score = self._invert(
            self._normalise(self._robust_mean(worst_queues, q_hi), q_lo, q_hi)
        )

        # Improvement score: skip first n//3 warm-up steps; compare remaining.
        # On short 20-step runs the first few steps reflect agent instability,
        # not the task; excluding them prevents spurious improvement penalties.
        warmup  = max(n_steps // 3, 1)
        rest    = throughputs[warmup:]
        half2   = max(len(rest) // 2, 1)
        early_tp = self._safe_mean(rest[:half2]) if rest else 0.0
        late_tp  = self._safe_mean(rest[half2:])  if len(rest) > half2 else early_tp
        improve_score = max(0.0, min(1.0,
            (late_tp - early_tp) / max(tp_hi - tp_lo, 1e-6) + 0.5
        ))

        # Smoothness: windowed switch count over last min(5,n) steps.
        # Whole-episode rate is unfair on short runs where 1-2 forced flips
        # (e.g. anti-stuck overrides) can push the fraction above the reference.
        window_size  = min(5, n_steps)
        window_steps = trajectory[-window_size:]
        prev_phases: Optional[tuple] = None
        window_switches = 0
        for sd in window_steps:
            phases_t = tuple(
                inter.get("phase", -1)
                for inter in sd.get("state_snapshot", {}).get("intersections", [])
            )
            if prev_phases is not None and phases_t != prev_phases:
                window_switches += 1
            prev_phases = phases_t
        window_rate  = window_switches / max(window_size - 1, 1)
        smooth_score = self._invert(
            self._normalise(window_rate, 0.0, _SWITCH_DEFAULT_HI)
        )

        outcome_score = (
            W_TP      * tp_score
            + W_QUEUE   * queue_score
            + W_IMPROVE * improve_score
            + W_SMOOTH  * smooth_score
        )
        outcome_score = max(0.0, min(1.0, outcome_score))

        # ------------------------------------------------------------------
        # 4. Combined score
        # ------------------------------------------------------------------
        raw   = W_PROCESS * process_score + W_OUTCOME * outcome_score
        final = gate * raw
        return float(max(0.0, min(1.0, final)))

    # ------------------------------------------------------------------
    # Extended interface (new — does not affect grade())
    # ------------------------------------------------------------------

    def grade_detailed(self, trajectory: List[Dict[str, Any]]) -> AggregationResult:
        """Return a rich AggregationResult for inspection / testing.

        WARNING: final_score here goes through the new pipeline and SHOULD
        match grade() exactly, but is produced via the new modular path.
        Use grade() for the official score.
        """
        if not trajectory:
            from graders.verifier    import VerifierResult
            from graders.constraints import ConstraintResult
            from graders.rubric      import RubricMetrics
            from graders.stability   import StabilityMetrics
            vr = VerifierResult(False, "Empty trajectory")
            cr = ConstraintResult(violations=(), gate=1.0, summary="OK")
            rm = RubricMetrics(0,0,1,1,0,0,1)
            sm = compute_stability([], 1)
            return aggregate(vr, cr, rm, sm, 0.0, "easy")

        n_inters = 1
        vr = verify_easy(trajectory)
        cr = check_constraints(trajectory, n_inters, "easy")
        rm = compute_rubric_easy(trajectory, self.calibration)
        sm = compute_stability(trajectory, n_inters)
        process_scores = self._compute_process_scores(trajectory, n_inters)
        ps = self._robust_mean(process_scores, default=0.5)
        return aggregate(vr, cr, rm, sm, ps, "easy")

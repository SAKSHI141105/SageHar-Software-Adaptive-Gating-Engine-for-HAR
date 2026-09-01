"""
sage_gate.py
============
SAGE-HAR: Software Adaptive Gating Engine for Human Activity Recognition.

This module implements the core gating algorithm: a lightweight statistical
"gate" that decides, window by window, whether a full HAR deep-learning
model actually needs to run. When the wearer is resting, the gate stays
closed and the expensive model is skipped; when real motion shows up, the
gate opens and the model runs.

Only the Python standard library is used (dataclasses, enum, math), so this
file has zero dependencies and can run anywhere -- including directly on a
resource-constrained wearable -- without installing PyTorch or anything else.

Algorithm, in four steps:
  1. Signal Vector Magnitude (SVM):  collapse (x, y, z) into one number.
  2. Window variance:                how "jumpy" was the signal this window.
  3. Rest-filtered EWMA baseline:    a running estimate of what "resting"
                                      variance looks like for this user,
                                      updated ONLY while the gate is closed.
  4. Dual-threshold hysteresis:      a Schmitt trigger -- one threshold to
                                      turn the gate ON, a lower one to turn
                                      it back OFF -- so the gate doesn't
                                      flicker when the signal hovers near
                                      a single cutoff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

# One raw accelerometer reading: (x, y, z) in g's.
Sample = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Step 1: Signal Vector Magnitude (SVM)
# ---------------------------------------------------------------------------
def signal_vector_magnitude(x: float, y: float, z: float) -> float:
    """Collapse a 3-axis accelerometer reading into a single "how much
    motion is happening" number, independent of which way the device is
    oriented on the body.

        SVM = sqrt(x^2 + y^2 + z^2)
    """
    return math.sqrt(x * x + y * y + z * z)


def svm_series(window: Sequence[Sample]) -> List[float]:
    """Apply signal_vector_magnitude to every sample in a window."""
    return [signal_vector_magnitude(x, y, z) for x, y, z in window]


# ---------------------------------------------------------------------------
# Step 2: Window variance
# ---------------------------------------------------------------------------
def window_variance(values: Sequence[float]) -> float:
    """Population variance of a list of numbers.

    Variance is high when the SVM signal is bouncing around a lot (the
    person is moving) and low when it's basically flat (the person is at
    rest, since a resting accelerometer just reads a constant ~1g of
    gravity plus a little sensor noise).
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


# ---------------------------------------------------------------------------
# Step 3 & 4: Gate configuration, state, and decision logic
# ---------------------------------------------------------------------------
class GateStatus(Enum):
    INACTIVE = "INACTIVE"  # gate closed -> HAR model is SKIPPED this window
    ACTIVE = "ACTIVE"       # gate open   -> HAR model RUNS this window


@dataclass
class GateConfig:
    """Tunable knobs for the gate. The defaults below are reasonable
    starting points and can be tuned per-user or per-device."""

    alpha: float = 0.05          # EWMA smoothing factor for the resting baseline
    k_high: float = 3.0          # activation threshold   = k_high * var_rest
    k_low: float = 1.5           # deactivation threshold  = k_low  * var_rest
    eps_floor: float = 1e-6      # minimum allowed baseline (avoids ~0 thresholds)
    initial_var_rest: float = 1e-4  # starting guess for var_rest before any
                                     # windows have been observed. Must be seeded
                                     # near a *typical* resting variance -- seeding
                                     # it at eps_floor would make the very first
                                     # window look like a huge spike relative to
                                     # the baseline and falsely trip the gate.


@dataclass
class GateDecision:
    """A record of one window's pass through the gate. Useful for logging,
    plotting, and unit tests."""

    window_index: int
    variance: float
    var_rest: float
    eps_high: float
    eps_low: float
    status_before: GateStatus
    status_after: GateStatus
    run_model: bool

    @property
    def transitioned(self) -> bool:
        return self.status_before is not self.status_after


class SageGate:
    """Stateful gate. Call `.step(window)` once per incoming sensor window.

    The dual-threshold logic mirrors a Schmitt trigger from analog
    electronics -- the same trick used to stop a noisy signal from
    "chattering" back and forth across a single cutoff. Here:

      - while INACTIVE, the gate updates its resting baseline and opens
        only if variance climbs *above* eps_high;
      - while ACTIVE, the baseline is frozen (so a long workout doesn't
        get folded in as the new "resting normal"), and the gate closes
        only once variance drops *below* eps_low (a lower bar than
        eps_high, which is what creates the hysteresis gap).
    """

    def __init__(self, config: Optional[GateConfig] = None) -> None:
        self.config = config or GateConfig()
        self.status = GateStatus.INACTIVE
        self.var_rest = self.config.initial_var_rest
        self.window_index = 0
        self._warmed_up = False

    def reset(self) -> None:
        """Return the gate to its startup state, keeping the same config."""
        self.status = GateStatus.INACTIVE
        self.var_rest = self.config.initial_var_rest
        self.window_index = 0
        self._warmed_up = False

    def step(self, window: Sequence[Sample]) -> GateDecision:
        """Process one window of raw (x, y, z) samples."""
        variance = window_variance(svm_series(window))
        return self.step_variance(variance)

    def step_variance(self, variance: float) -> GateDecision:
        """Same as `.step()`, but takes an already-computed variance
        directly. This is what the JS dashboard's decision logic mirrors,
        since it computes variance client-side.
        """
        status_before = self.status
        cfg = self.config

        if status_before is GateStatus.INACTIVE:
            # Gate is CLOSED: this window is trustworthy "resting" data,
            # so fold it into the baseline BEFORE checking the threshold.
            #
            # On the very first window ever seen, `initial_var_rest` is just
            # a guess and may not match this particular sensor/user at all.
            # Rather than risk a false activation against a bad guess, we
            # "warm start" by snapping the baseline directly to the first
            # observed variance instead of blending it in gradually.
            if not self._warmed_up:
                self.var_rest = max(variance, cfg.eps_floor)
                self._warmed_up = True
            else:
                self.var_rest = (1 - cfg.alpha) * self.var_rest + cfg.alpha * variance

            baseline = max(self.var_rest, cfg.eps_floor)
            eps_high = cfg.k_high * baseline
            eps_low = cfg.k_low * baseline

            if variance > eps_high:
                self.status = GateStatus.ACTIVE
        else:
            # Gate is OPEN: baseline and thresholds stay frozen.
            baseline = max(self.var_rest, cfg.eps_floor)
            eps_high = cfg.k_high * baseline
            eps_low = cfg.k_low * baseline

            if variance < eps_low:
                self.status = GateStatus.INACTIVE

        decision = GateDecision(
            window_index=self.window_index,
            variance=variance,
            var_rest=self.var_rest,
            eps_high=eps_high,
            eps_low=eps_low,
            status_before=status_before,
            status_after=self.status,
            run_model=self.status is GateStatus.ACTIVE,
        )
        self.window_index += 1
        return decision

    def step_batch(self, windows: Sequence[Sequence[Sample]]) -> List[GateDecision]:
        """Convenience helper: run the gate over many windows in order."""
        return [self.step(w) for w in windows]


# ---------------------------------------------------------------------------
# Smoke test -- run this file directly to see the gate react to a burst of
# motion sandwiched between two resting periods.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    random.seed(0)
    gate = SageGate()

    def resting_window(n: int = 40) -> List[Sample]:
        return [
            (random.gauss(0, 0.02), random.gauss(0, 0.02), 1.0 + random.gauss(0, 0.02))
            for _ in range(n)
        ]

    def moving_window(n: int = 40) -> List[Sample]:
        return [
            (random.gauss(0, 1.5), random.gauss(0, 1.5), 1.0 + random.gauss(0, 1.5))
            for _ in range(n)
        ]

    demo_windows = (
        [resting_window() for _ in range(20)]
        + [moving_window() for _ in range(10)]
        + [resting_window() for _ in range(20)]
    )

    print(f"{'win':>4}  {'variance':>10}  {'var_rest':>10}  {'eps_high':>10}  {'status':<9}")
    for d in gate.step_batch(demo_windows):
        marker = "  <-- TRANSITION" if d.transitioned else ""
        print(f"{d.window_index:4d}  {d.variance:10.5f}  {d.var_rest:10.5f}  "
              f"{d.eps_high:10.5f}  {d.status_after.value:<9}{marker}")

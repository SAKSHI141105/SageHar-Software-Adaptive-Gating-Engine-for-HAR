"""Tests for sage_gate.py -- the core gate engine."""

import math
import random

import pytest

from sage_gate import (
    GateConfig,
    GateStatus,
    SageGate,
    signal_vector_magnitude,
    window_variance,
)


def resting_window(rng, n=40, std=0.02):
    return [(rng.gauss(0, std), rng.gauss(0, std), 1.0 + rng.gauss(0, std)) for _ in range(n)]


def moving_window(rng, n=40, std=1.5):
    return [(rng.gauss(0, std), rng.gauss(0, std), 1.0 + rng.gauss(0, std)) for _ in range(n)]


class TestSignalVectorMagnitude:
    def test_known_3_4_0_triangle(self):
        assert signal_vector_magnitude(3, 4, 0) == pytest.approx(5.0)

    def test_zero_is_zero(self):
        assert signal_vector_magnitude(0, 0, 0) == 0.0

    def test_always_non_negative(self):
        assert signal_vector_magnitude(-3, -4, 0) == pytest.approx(5.0)


class TestWindowVariance:
    def test_known_values(self):
        # population variance of [1,2,3,4]: mean=2.5, var=1.25
        assert window_variance([1, 2, 3, 4]) == pytest.approx(1.25)

    def test_constant_series_has_zero_variance(self):
        assert window_variance([5, 5, 5, 5]) == pytest.approx(0.0)

    def test_empty_series_is_zero(self):
        assert window_variance([]) == 0.0

    def test_single_sample_is_zero(self):
        assert window_variance([42.0]) == 0.0


class TestSageGateWarmStart:
    """Regression tests for the seed bug: an earlier version seeded var_rest
    near zero, which made window 0 look like a huge spike and falsely
    activated the gate immediately."""

    def test_stays_inactive_through_typical_rest(self):
        rng = random.Random(0)
        gate = SageGate()
        for _ in range(30):
            decision = gate.step(resting_window(rng))
        assert decision.status_after is GateStatus.INACTIVE

    def test_first_window_never_forces_activation(self):
        rng = random.Random(1)
        gate = SageGate()
        decision = gate.step(resting_window(rng))
        assert decision.status_after is GateStatus.INACTIVE
        assert decision.transitioned is False


class TestSageGateHysteresis:
    def test_activates_on_motion_burst(self):
        rng = random.Random(2)
        gate = SageGate()
        for _ in range(20):
            gate.step(resting_window(rng))
        decision = gate.step(moving_window(rng))
        assert decision.status_after is GateStatus.ACTIVE
        assert decision.run_model is True

    def test_deactivates_after_motion_stops(self):
        rng = random.Random(3)
        gate = SageGate()
        for _ in range(20):
            gate.step(resting_window(rng))
        for _ in range(10):
            gate.step(moving_window(rng))
        assert gate.status is GateStatus.ACTIVE

        decision = None
        for _ in range(30):
            decision = gate.step(resting_window(rng))
            if decision.status_after is GateStatus.INACTIVE:
                break
        assert decision.status_after is GateStatus.INACTIVE

    def test_var_rest_frozen_while_active(self):
        rng = random.Random(4)
        gate = SageGate()
        for _ in range(20):
            gate.step(resting_window(rng))
        gate.step(moving_window(rng))
        assert gate.status is GateStatus.ACTIVE
        var_rest_snapshot = gate.var_rest

        for _ in range(5):
            gate.step(moving_window(rng))
            assert gate.var_rest == var_rest_snapshot


class TestGateConfig:
    def test_defaults_form_valid_hysteresis_band(self):
        cfg = GateConfig()
        assert cfg.k_high > cfg.k_low > 0


class TestSageGateReset:
    def test_reset_returns_to_inactive_and_clears_warmup(self):
        rng = random.Random(5)
        gate = SageGate()
        for _ in range(20):
            gate.step(resting_window(rng))
        gate.step(moving_window(rng))
        assert gate.status is GateStatus.ACTIVE

        gate.reset()
        assert gate.status is GateStatus.INACTIVE
        assert gate.window_index == 0
        assert gate._warmed_up is False

    def test_step_batch_matches_sequential_step_calls(self):
        rng_a = random.Random(6)
        windows = [resting_window(rng_a) for _ in range(10)] + [moving_window(rng_a) for _ in range(5)]

        gate_batch = SageGate()
        batch_decisions = gate_batch.step_batch(windows)

        gate_seq = SageGate()
        seq_decisions = [gate_seq.step(w) for w in windows]

        assert [d.status_after for d in batch_decisions] == [d.status_after for d in seq_decisions]
        assert [d.variance for d in batch_decisions] == pytest.approx(
            [d.variance for d in seq_decisions]
        )

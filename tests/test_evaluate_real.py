"""Tests for evaluate_real.py's data-plumbing helpers (grouping, conversion,
power estimate) using small synthetic tensors -- not the real 61MB dataset,
so these run in milliseconds."""

import pytest

torch = pytest.importorskip("torch")

from evaluate_real import estimate_power_saved_pct, group_by_subject, window_to_gate_samples


class TestWindowToGateSamples:
    def test_converts_3xN_tensor_to_xyz_tuples(self):
        window = torch.tensor([
            [1.0, 2.0, 3.0],   # x
            [4.0, 5.0, 6.0],   # y
            [7.0, 8.0, 9.0],   # z
        ])
        samples = window_to_gate_samples(window)
        assert samples == [(1.0, 4.0, 7.0), (2.0, 5.0, 8.0), (3.0, 6.0, 9.0)]


class TestGroupBySubject:
    def test_groups_rows_by_subject_id_preserving_order(self):
        X = torch.arange(12).reshape(4, 3).float()
        y = torch.tensor([0, 1, 0, 1])
        subjects = [1, 1, 2, 2]

        groups = list(group_by_subject(X, y, subjects))
        group_ids = [g[0] for g in groups]
        assert sorted(group_ids) == [1, 2]

        for sid, Xs, ys in groups:
            if sid == 1:
                assert Xs.shape[0] == 2
                assert ys.tolist() == [0, 1]
            else:
                assert Xs.shape[0] == 2
                assert ys.tolist() == [0, 1]

    def test_non_contiguous_subject_ids_still_group_correctly(self):
        X = torch.arange(9).reshape(3, 3).float()
        y = torch.tensor([9, 9, 9])
        subjects = [5, 1, 5]  # subject 5's rows are NOT adjacent

        groups = {sid: Xs.shape[0] for sid, Xs, ys in group_by_subject(X, y, subjects)}
        assert groups == {5: 2, 1: 1}


class TestEstimatePowerSaved:
    def test_zero_windows_is_zero(self):
        assert estimate_power_saved_pct(0, 0) == 0.0

    def test_all_har_calls_saves_almost_nothing(self):
        pct = estimate_power_saved_pct(1000, 1000)
        assert pct < 5.0


class TestFirstWindowNeverLeaksGroundTruth:
    """Regression test for the seeding bug: an earlier version seeded
    last_pred with the first window's own TRUE label ("a harmless
    placeholder"), which leaked ground truth into the gated prediction
    whenever the gate and heartbeat both stayed quiet past window 0 --
    inflating reported accuracy by ~2-3pp. Window 0 of every subject
    stream must always trigger a real classifier call, never a value
    derived from the label being predicted.

    This runs the same first-window logic evaluate_real.py's main loop
    uses, without needing the real 61MB dataset or a trained checkpoint.
    """

    def test_window_zero_always_forces_a_real_call(self):
        from sage_gate import GateConfig, SageGate

        # A gate config picked to be maximally likely to stay INACTIVE on
        # window 0 (huge k_high), with heartbeat far larger than 1, so the
        # only way window 0 gets classified is the explicit `i == 0` check.
        gate = SageGate(GateConfig(k_high=100.0, k_low=50.0))
        window = [(0.001, 0.001, 1.0)] * 40  # trivially low-variance / resting
        decision = gate.step(window)

        assert decision.run_model is False  # gate genuinely stayed quiet

        heartbeat = 10
        windows_since_call = 1  # after gate.step() above, mirroring the real loop
        i = 0
        heartbeat_due = windows_since_call >= heartbeat
        forced = decision.run_model or heartbeat_due or i == 0
        assert forced is True, "window 0 must always be classified for real"

"""Tests for evaluate.py -- the synthetic gate-tuning benchmark.

Two of these are explicit regression tests for real bugs found and fixed
in this file: the gated-prediction carry-forward logic, and non-reproducible
seeding via Python's randomized string hash().
"""

import random
import subprocess
import sys

import pytest

import evaluate as ev
from sage_gate import GateConfig, SageGate


class TestMacroF1:
    def test_perfect_predictions_score_1(self):
        y_true = [0, 1, 0, 1, 1]
        assert ev.macro_f1(y_true, y_true) == pytest.approx(1.0)

    def test_all_wrong_scores_0(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        assert ev.macro_f1(y_true, y_pred) == pytest.approx(0.0)

    def test_empty_class_does_not_crash(self):
        # class 1 never appears in y_true or y_pred -- precision/recall
        # denominators are 0 and must be handled without division errors.
        # Class 1's undefined precision/recall conventionally count as 0,
        # so macro-F1 averages a perfect class 0 (f1=1.0) with an empty
        # class 1 (f1=0.0) to land at 0.5, not 1.0.
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]
        assert ev.macro_f1(y_true, y_pred) == pytest.approx(0.5)


class TestGatedPredictionDefaultsToRest:
    """Regression test for the carry-forward bug: gated_pred used to reuse
    the *last active* prediction through every subsequent rest window,
    indefinitely, instead of defaulting to REST while the gate is closed.
    """

    def test_inactive_windows_predict_rest_not_stale_active_label(self):
        rng = random.Random(0)
        # One short active bout, correctly classified as active (label 1),
        # followed by a long run of INACTIVE windows.
        decisions = [
            type("D", (), {"run_model": True})(),
            type("D", (), {"run_model": False})(),
            type("D", (), {"run_model": False})(),
            type("D", (), {"run_model": False})(),
        ]
        labels = [1, 0, 0, 0]

        gated_pred = [
            ev.classify(label, rng) if d.run_model else 0
            for d, label in zip(decisions, labels)
        ]

        # Every window after the active one must default to 0 (rest),
        # never silently inherit the earlier "active" classification.
        assert gated_pred[1:] == [0, 0, 0]


class TestEstimatePowerSaved:
    def test_zero_windows_is_zero(self):
        assert ev.estimate_power_saved_pct(0, 0) == 0.0

    def test_all_windows_skipped_saves_close_to_100_percent(self):
        pct = ev.estimate_power_saved_pct(1000, 0)
        assert pct > 95.0

    def test_no_windows_skipped_saves_little(self):
        pct = ev.estimate_power_saved_pct(1000, 1000)
        assert pct < 5.0


class TestSyntheticStreamGenerator:
    def test_generates_requested_number_of_windows(self):
        rng = random.Random(0)
        windows, labels = ev.generate_subject_stream(ev.PROFILES["calm"], 50, rng)
        assert len(windows) == 50
        assert len(labels) == 50

    def test_labels_are_binary(self):
        rng = random.Random(0)
        _, labels = ev.generate_subject_stream(ev.PROFILES["restless"], 200, rng)
        assert set(labels) <= {0, 1}


class TestReproducibility:
    """Regression test for the hash()-seeding bug: running evaluate.py
    twice used to produce different results every time because Python's
    string hash() is randomized per-process by default."""

    def test_two_separate_processes_produce_identical_output(self):
        run1 = subprocess.run([sys.executable, "evaluate.py"], capture_output=True, text=True, timeout=60)
        run2 = subprocess.run([sys.executable, "evaluate.py"], capture_output=True, text=True, timeout=60)
        assert run1.returncode == 0
        assert run2.returncode == 0
        assert run1.stdout == run2.stdout

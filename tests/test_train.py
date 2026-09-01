"""Tests for train.py's data loading and metric helpers.

Full training is exercised manually (`python train.py`) rather than in the
test suite -- it takes real wall-clock time (~15-30s) and downloads/uses a
61MB dataset, which isn't a fit for a test suite that should run in seconds.
What we *can* and do test here without touching the real dataset: the
metric math, and the UCI-HAR text-format parser against small fixture data.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from train import _load_signal_file, macro_f1

UCI_HAR_PRESENT = Path("data/raw/uci_har/train/Inertial Signals/total_acc_x_train.txt").exists()


class TestMacroF1MultiClass:
    def test_perfect_predictions_score_1(self):
        y_true = [0, 1, 2, 3, 4, 5]
        assert macro_f1(y_true, y_true, num_classes=6) == pytest.approx(1.0)

    def test_all_wrong_scores_0(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        assert macro_f1(y_true, y_pred, num_classes=2) == pytest.approx(0.0)

    def test_missing_class_does_not_crash(self):
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]
        assert macro_f1(y_true, y_pred, num_classes=3) == pytest.approx(1.0 / 3)


class TestLoadSignalFile:
    def test_parses_whitespace_separated_rows(self, tmp_path):
        f = tmp_path / "signal.txt"
        f.write_text("  1.0 2.0 3.0\n  4.5 5.5 6.5\n")
        tensor = _load_signal_file(f)
        assert tensor.shape == (2, 3)
        assert tensor[0].tolist() == pytest.approx([1.0, 2.0, 3.0])
        assert tensor[1].tolist() == pytest.approx([4.5, 5.5, 6.5])


@pytest.mark.skipif(not UCI_HAR_PRESENT, reason="UCI-HAR dataset not present in data/raw/uci_har")
class TestLoadSplitWithRealData:
    def test_train_and_test_splits_load_with_expected_shapes(self):
        from train import load_split

        X_train, y_train = load_split("train")
        X_test, y_test = load_split("test")

        assert X_train.shape[1:] == (3, 128)
        assert X_test.shape[1:] == (3, 128)
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert set(y_train.tolist()) <= {0, 1, 2, 3, 4, 5}

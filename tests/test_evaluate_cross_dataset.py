"""Tests for evaluate_cross_dataset.py's label-mapping and loading logic."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from evaluate_cross_dataset import DATA_DIR, FOLDER_PREFIX_TO_LABEL, trial_prefix
from train import ACTIVITY_NAMES

MOTIONSENSE_PRESENT = DATA_DIR.exists()


class TestTrialPrefix:
    def test_extracts_prefix_before_underscore(self):
        assert trial_prefix("wlk_15") == "wlk"
        assert trial_prefix("dws_2") == "dws"
        assert trial_prefix("ups_11") == "ups"


class TestFolderPrefixToLabelMapping:
    def test_maps_to_correct_uci_har_indices(self):
        assert ACTIVITY_NAMES[FOLDER_PREFIX_TO_LABEL["wlk"]] == "WALKING"
        assert ACTIVITY_NAMES[FOLDER_PREFIX_TO_LABEL["ups"]] == "WALKING_UPSTAIRS"
        assert ACTIVITY_NAMES[FOLDER_PREFIX_TO_LABEL["dws"]] == "WALKING_DOWNSTAIRS"
        assert ACTIVITY_NAMES[FOLDER_PREFIX_TO_LABEL["sit"]] == "SITTING"
        assert ACTIVITY_NAMES[FOLDER_PREFIX_TO_LABEL["std"]] == "STANDING"

    def test_jogging_and_laying_are_not_mapped(self):
        # No UCI-HAR equivalent for MotionSense's "jog", and MotionSense has
        # no LAYING trials at all -- both must stay out of the evaluated set
        # rather than being forced into a misleading fake mapping.
        assert "jog" not in FOLDER_PREFIX_TO_LABEL
        assert ACTIVITY_NAMES.index("LAYING") not in FOLDER_PREFIX_TO_LABEL.values()


@pytest.mark.skipif(not MOTIONSENSE_PRESENT, reason="MotionSense dataset not present in data/raw/motionsense")
class TestLoadMotionsenseWindows:
    def test_windows_have_expected_shape_and_only_mapped_labels(self):
        from evaluate_cross_dataset import EVALUATED_CLASSES, load_motionsense_windows

        X, y = load_motionsense_windows()
        assert X.shape[1:] == (3, 128)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] > 0
        assert set(y.tolist()) <= set(EVALUATED_CLASSES)

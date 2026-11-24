import numpy as np
import pandas as pd
import pytest

from evaluate_extension import RULE_COLUMNS, counts, frozen_cutoff, gap_interval


def test_freeze_uses_only_calibration_order_statistic_and_keeps_ties():
    scores = np.array([.8, .1, .3, .2, .4, .5, .6, .7, .8, .8])
    cutoff = frozen_cutoff(scores)
    assert cutoff == .8
    assert int((scores <= cutoff).sum()) == 10
    assert (np.array([.79, .8, .81]) <= cutoff).tolist() == [True, True, False]
    with pytest.raises(ValueError, match="finite scores"):
        frozen_cutoff(np.array([np.nan]))


def test_replay_counts_and_bootstrap_gap_use_all_patients_as_denominators():
    first, second = RULE_COLUMNS
    rows = pd.DataFrame({
        "label": [0, 0, 1, 1], "predicted": [0, 1, 0, 1], "error": [0, 1, 1, 0],
        first + "_accept": [True, True, True, False], second + "_accept": [True] * 4,
    })
    assert counts(rows, first) == {
        "n": 4, "accepted": 3, "n_malignant": 2, "accepted_malignant": 1,
        "accepted_errors": 2, "accepted_false_negatives": 1,
    }
    assert gap_interval([pd.concat([rows] * 10, ignore_index=True)])[second] == [0., 0.]

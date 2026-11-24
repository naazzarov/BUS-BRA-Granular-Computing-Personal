import numpy as np
import pandas as pd
import pytest

from evaluate import assert_disjoint, augrc, fixed_coverage, summarize


def test_augrc_equals_integral_of_empirical_generalized_risk():
    errors = np.array([1, 0, 0, 1])
    risk = np.array([0.3, 0.2, 0.7, 0.9])
    ranked = errors[np.argsort(risk)]
    curve = np.r_[0, ranked.cumsum() / len(errors)]
    assert augrc(errors, risk) == pytest.approx(np.trapz(curve, dx=1/len(errors)))


def test_tied_scores_use_expected_rank():
    errors = np.array([1, 0, 1])
    risk = np.array([.1, .1, .5])
    first = augrc(errors, [.1, .11, .5])
    second = augrc(errors, [.11, .1, .5])
    assert augrc(errors, risk) == pytest.approx((first + second) / 2)


def test_fixed_coverage_resolves_boundary_by_expected_counts():
    rows = pd.DataFrame({"label":[1, 1, 0, 0],
                         "predicted":[0, 1, 0, 1],
                         "error":[1, 0, 0, 1],
                         "risk":[.1, .1, .1, .7]})
    m = fixed_coverage(rows, "risk", .5)
    assert m["accepted"] == 2
    assert m["accepted_errors"] == pytest.approx(2/3)
    assert m["accepted_false_negatives"] == pytest.approx(2/3)


def test_patient_reduction_and_disjointness():
    groups = np.array(["a", "a", "b"])
    y = np.array([0, 0, 1])
    m = summarize(groups, y, np.array([.2, .6, .8]))
    assert m.loc["a", "mean_score"] == pytest.approx(.4)
    assert m.loc["a", "gap"] == pytest.approx(.4)
    assert m.loc["b", "n_images"] == 1
    with pytest.raises(ValueError, match="both sides"):
        assert_disjoint(np.array([0]), np.array([1, 2]), groups)

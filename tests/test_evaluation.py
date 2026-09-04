from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import f_oneway

from core.evaluate import anova_p, variance_reduction


def test_variance_reduction_matches_current_weighted_within_zone_formula():
    outcome = pd.Series([1.0, 2.0, 9.0, 10.0])
    labels = np.array([0, 0, 1, 1])
    global_variance = np.var(outcome, ddof=1)
    expected = (global_variance - 0.5) / global_variance

    assert variance_reduction(outcome, labels) == pytest.approx(expected, rel=1e-12)


def test_variance_reduction_treats_singleton_zone_variance_as_zero():
    outcome = pd.Series([1.0, 2.0, 10.0])
    labels = np.array([0, 0, 1])
    global_variance = np.var(outcome, ddof=1)
    expected_within = np.average([0.5, 0.0], weights=[2, 1])

    assert variance_reduction(outcome, labels) == pytest.approx(
        (global_variance - expected_within) / global_variance,
        rel=1e-12,
    )


def test_variance_reduction_is_zero_for_constant_outcome():
    assert variance_reduction(pd.Series([4.0, 4.0, 4.0, 4.0]), [0, 0, 1, 1]) == 0.0


def test_anova_matches_scipy_one_way_result():
    outcome = pd.Series([1.0, 1.5, 2.0, 8.0, 8.5, 9.0, 15.0, 15.5, 16.0])
    labels = np.repeat([0, 1, 2], 3)
    expected = f_oneway(outcome[:3], outcome[3:6], outcome[6:]).pvalue

    assert anova_p(outcome, labels) == pytest.approx(expected, rel=1e-10, abs=1e-15)

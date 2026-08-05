import numpy as np
import pandas as pd
import pytest

from geofarmai.evaluation import anova_p, variance_reduction


def test_variance_reduction_known_case():
    y = pd.Series([1.0, 2.0, 9.0, 10.0])
    labels = np.array([0, 0, 1, 1])
    expected = (np.var(y, ddof=1) - 0.5) / np.var(y, ddof=1)
    assert variance_reduction(y, labels) == expected


def test_anova_detects_separated_groups_and_is_label_permutation_invariant():
    y = pd.Series([1.0, 1.2, 0.9, 8.8, 9.0, 9.2])
    labels = np.array([0, 0, 0, 1, 1, 1])
    p_value = anova_p(y, labels)
    assert p_value < 0.01
    assert anova_p(y, 1 - labels) == pytest.approx(p_value)


def test_constant_yield_variance_reduction_is_zero():
    assert variance_reduction(pd.Series([4.0, 4.0, 4.0, 4.0]), [0, 0, 1, 1]) == 0.0

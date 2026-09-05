import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm


def variance_reduction(values, labels):
    """Return weighted within-group variance reduction for numeric values."""

    values = pd.Series(values)
    gvar = np.var(values, ddof=1)
    zones = pd.DataFrame({'values': values, 'z': labels}).groupby('z')['values'].var(ddof=1)
    w = pd.Series(labels).value_counts().sort_index()
    zvar = np.average(zones.fillna(0).values, weights=w.values)
    return float((gvar - zvar) / gvar) if gvar > 0 else 0.0


def anova_p(values, labels):
    """Return the one-way ANOVA p-value for arbitrary numeric outcomes."""

    values = pd.Series(values)
    df = pd.DataFrame({'values': values, 'z': pd.Categorical(labels)})
    model = ols('values ~ C(z)', data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    return float(table['PR(>F)'].iloc[0])

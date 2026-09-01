import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm


def variance_reduction(y: pd.Series, labels):
    gvar = np.var(y, ddof=1)
    zones = pd.DataFrame({'y': y, 'z': labels}).groupby('z')['y'].var(ddof=1)
    w = pd.Series(labels).value_counts().sort_index()
    zvar = np.average(zones.fillna(0).values, weights=w.values)
    return float((gvar - zvar) / gvar) if gvar > 0 else 0.0


def anova_p(y: pd.Series, labels):
    df = pd.DataFrame({'y': y, 'z': pd.Categorical(labels)})
    model = ols('y ~ C(z)', data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    return float(table['PR(>F)'].iloc[0])

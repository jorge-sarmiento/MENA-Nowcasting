"""
filters.py
----------
Time-series decomposition utilities.

Implements the Hodrick-Prescott (1997) filter to separate cyclical
fluctuations from the underlying trend component.

KNOWN LIMITATION — TWO-SIDED FILTER:
    The HP filter is a two-sided (symmetric) smoother: at each point t,
    the estimated trend uses observations both before AND after t.
    This means the cyclical component at time t implicitly contains
    information from future observations t+1, t+2, ...

    In this pipeline the HP filter is used ONLY to construct the target
    proxy for the CycleAligner (utils.py), which is estimated on the
    first min_train*3 months of the training window only. It is NOT used
    in any model prediction or OOS evaluation. The lookahead bias from
    the two-sided filter therefore affects only the lead-lag selection
    step (which lags each predictor), not the actual GDP forecasts.

    This is documented as a limitation: the selected leads may be
    slightly optimistic because the HP cycle target was constructed with
    some future information. The practical impact is limited because:
        (a) The CycleAligner uses simple correlation, not model fitting.
        (b) The leads are constrained to [1, 6] months (MAX_LEAD_MONTHS).
        (c) The effect diminishes as the training window grows.

    Alternative: Hamilton (2018) proposes a regression-based filter that
    avoids the two-sided problem and endpoint distortion.
    Ref: Hamilton, J.D. (2018). Why You Should Never Use the
         Hodrick-Prescott Filter. Review of Economics and Statistics,
         100(5), 831-843. DOI:10.1162/rest_a_00706

    We retain the HP filter here for comparability with central bank
    nowcasting literature (ECB, Fed, BIS all use HP-based cycles for
    leading indicator selection).

References:
    - Hodrick, R. & Prescott, E. (1997). Postwar US Business Cycles:
      An Empirical Investigation. Journal of Money, Credit and Banking,
      29(1), 1-16. DOI:10.2307/2953682
    - Ravn, M. & Uhlig, H. (2002). On Adjusting the Hodrick-Prescott
      Filter for the Frequency of Observations. Review of Economics and
      Statistics, 84(2), 371-376. DOI:10.1162/003465302317411604
      (Source of lambda=14400 monthly, 1600 quarterly)
    - Hamilton, J.D. (2018). Why You Should Never Use the
      Hodrick-Prescott Filter. Review of Economics and Statistics,
      100(5), 831-843. DOI:10.1162/rest_a_00706
"""

import pandas as pd
import statsmodels.api as sm

from src.config import HP_LAMBDA_MONTHLY, HP_LAMBDA_QUARTERLY


def extract_cycle(series: pd.Series, frequency: str = "monthly") -> tuple:
    """
    Decomposes a time series into cyclical and trend components
    using the HP filter.

    SCOPE OF USE IN THIS PIPELINE:
        Used exclusively as the cycle target for CycleAligner in main.py,
        estimated on the initial training window only. Not used for
        forecasting or OOS evaluation. See module docstring for the
        two-sided limitation and its bounded impact.

    Parameters
    ----------
    series : pd.Series
        Input time series. Must not contain NaN values.
    frequency : str
        Either 'monthly' (lambda=14400, Ravn & Uhlig 2002) or
        'quarterly' (lambda=1600, Hodrick & Prescott 1997).

    Returns
    -------
    tuple : (pd.Series, pd.Series)
        Cyclical component and trend component.
    """
    lam = HP_LAMBDA_MONTHLY if frequency == "monthly" else HP_LAMBDA_QUARTERLY
    cycle, trend = sm.tsa.filters.hpfilter(series.dropna(), lamb=lam)
    return cycle, trend
"""
utils.py
--------
Econometric utilities for pre-estimation diagnostics.

Contains:
    - StationarityTester: ADF unit root testing for input validation.
    - CycleAligner: Optimizes the lead-lag structure of each predictor
      relative to a target business cycle proxy.

METHODOLOGY NOTE — CycleAligner and Expanding-Window Evaluation:
    The CycleAligner estimates the optimal lead (in months) for each
    predictor by maximizing its absolute correlation with a cycle proxy
    over a fixed initial training window. These leads are then held
    CONSTANT throughout the expanding-window OOS evaluation.

    This design choice requires explicit justification:

    OPTION A — Re-estimate leads at each OOS step (fully recursive):
        Pros: Theoretically cleaner. Leads adapt as more data arrives.
        Cons: (1) Computationally expensive — O(T^2) vs O(T).
              (2) Lead estimates are noisy on small samples (early OOS steps
                  have only ~12-20 quarters = 36-60 months to estimate leads).
              (3) Changing leads mid-evaluation breaks interpretability of the
                  coefficient time series.

    OPTION B — Fix leads on initial training window (implemented here):
        Pros: (1) Consistent with the "pre-specified" approach in central
                  bank nowcasting practice (ECB, Norges Bank, SNB all use
                  fixed lead structures estimated on historical data before
                  the evaluation period).
              (2) Avoids estimation noise from short windows.
              (3) Equivalent to a researcher specifying leads a priori
                  based on economic theory, which is the most transparent
                  approach.
        Cons: If lead structure changes over time (e.g., post-COVID), the
              fixed leads may become suboptimal. This is a known limitation.

    RESOLUTION — documented a priori lead rationale:
        The leads estimated by CycleAligner should be interpreted as
        data-driven confirmation of economically motivated a priori leads:
            - Equity prices: lead by 1-3 months (standard in Stock & Watson 2003)
            - Oil prices: lead by 1-2 months (Kilian 2009)
            - Banking/credit: lead by 0-2 months (contemporaneous credit channel)
        MAX_LEAD_MONTHS=6 (from config) is consistent with the standard
        "6-month leading indicator" horizon used by OECD Composite Leading Indicators.
        Ref: OECD (2012). OECD System of Composite Leading Indicators.
             https://www.oecd.org/sdd/leading-indicators/

    RESIDUAL LIMITATION — HP filter used as target:
        The cycle proxy is computed via the HP filter (two-sided smoother).
        See filters.py for full documentation of this limitation and its
        bounded impact on lead selection.

References:
    - Dickey, D. & Fuller, W. (1979). Distribution of the Estimators
      for Autoregressive Time Series with a Unit Root. JASA, 74(366),
      427-431. DOI:10.2307/2286348
    - Stock, J. & Watson, M. (2003). Forecasting Output and Inflation:
      The Role of Asset Prices. Journal of Economic Literature, 41(3),
      788-829. DOI:10.1257/jel.41.3.788
    - Kilian, L. (2009). Not All Oil Price Shocks Are Alike.
      American Economic Review, 99(3), 1053-1069. DOI:10.1257/aer.99.3.1053
    - OECD (2012). OECD System of Composite Leading Indicators.
      OECD Statistics Directorate.
      https://www.oecd.org/sdd/leading-indicators/
"""

import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.config import MAX_LEAD_MONTHS

logger = logging.getLogger(__name__)


class StationarityTester:
    """
    Validates that input series are stationary before model estimation.

    Non-stationary series violate the assumptions of OLS and PCA.
    The Augmented Dickey-Fuller test is applied with the null hypothesis
    of a unit root; rejection at the specified significance level
    confirms stationarity.
    """

    @staticmethod
    def is_stationary(series: pd.Series, significance: float = 0.05) -> bool:
        """
        Returns True if the ADF test rejects the unit root null.

        Parameters
        ----------
        series : pd.Series
            Time series to test. NaN values are dropped automatically.
        significance : float
            p-value threshold for rejection.
        """
        clean = series.dropna()
        if len(clean) < 20:
            logger.warning(
                "Insufficient observations (%d) for ADF test on '%s'.",
                len(clean),
                series.name,
            )
            return False

        adf_stat, p_value, _, _, _, _ = adfuller(clean, autolag="AIC")
        is_stat = p_value < significance
        logger.debug(
            "ADF test for '%s': stat=%.3f, p=%.4f -> %s",
            series.name,
            adf_stat,
            p_value,
            "stationary" if is_stat else "non-stationary",
        )
        return is_stat


class CycleAligner:
    """
    Identifies the optimal lead for each predictor relative to a
    target business cycle proxy.

    For each feature X_j, the aligner tests whether X_j(t-k) for
    k = 1, ..., max_lead maximizes the absolute correlation with
    the target Y(t). The selected lag k* represents how many months
    X_j leads the real economy.

    DESIGN: Leads are estimated ONCE on the initial training window
    and then applied FIXED to the full dataset. See module docstring
    for the methodological justification of this choice vs. fully
    recursive re-estimation.

    Parameters
    ----------
    target : pd.Series
        The business cycle proxy (e.g., HP-filtered equity index or
        HP cycle of Brent Oil) estimated on training data only.
        NOTE: If the target is HP-filtered, it contains a two-sided
        lookahead bias. See filters.py for documentation.
    """

    def __init__(self, target: pd.Series):
        self.target = target
        self.lag_map = {}
        self.correlation_map = {}

    def align(
        self, features: pd.DataFrame, max_lead: int = None
    ) -> pd.DataFrame:
        """
        Shifts each feature backward by its optimal lead.

        The optimal lead is the integer k in [1, max_lead] that
        maximises |corr(target_t, X_{t-k})|. Dummy variables (regime
        indicators) are kept at lag 0 — they are contemporaneous by
        construction (e.g., OPEC_Cuts is known at time t).

        FIXED LEADS: the lag_map estimated here is applied to the
        entire sample (training + OOS) in main.py. This is intentional
        and documented — see module docstring for justification.

        Parameters
        ----------
        features : pd.DataFrame
            Monthly indicator matrix (training window only).
        max_lead : int, optional
            Maximum months to test. Defaults to MAX_LEAD_MONTHS=6.

        Returns
        -------
        pd.DataFrame
            Features shifted by their optimal leads on the training window.
        """
        if max_lead is None:
            max_lead = MAX_LEAD_MONTHS

        aligned = pd.DataFrame(index=features.index)

        header = f"{'Variable':<25} {'Optimal Lead':>14} {'|Corr|':>8}"
        logger.info(header)
        logger.info("-" * len(header))

        for col in features.columns:
            # Dummy variables: contemporaneous by definition.
            # OPEC cut decisions and fiscal impulse dates are known at t.
            if "Cuts" in col or "Impulse" in col:
                aligned[col] = features[col]
                self.lag_map[col] = 0
                self.correlation_map[col] = np.nan
                continue

            best_abs_corr = -1.0
            best_lag = 0

            for lag in range(1, max_lead + 1):
                shifted = features[col].shift(lag)
                corr = self.target.corr(shifted)
                if abs(corr) > best_abs_corr:
                    best_abs_corr = abs(corr)
                    best_lag = lag

            self.lag_map[col] = best_lag
            self.correlation_map[col] = best_abs_corr
            aligned[col] = features[col].shift(best_lag)

            logger.info(
                "%-25s %10d mo %8.4f", col, best_lag, best_abs_corr
            )

        aligned = aligned.dropna()
        logger.info(
            "Alignment complete: %d observations retained. "
            "Leads fixed — will be applied unchanged to OOS period.",
            len(aligned)
        )
        return aligned

    def get_lead_summary(self) -> pd.DataFrame:
        """
        Returns a summary table of optimal leads and correlations.
        Includes a note on the fixed-lead methodology.
        """
        records = []
        for col in self.lag_map:
            records.append(
                {
                    "Variable": col,
                    "Lead_Months": self.lag_map[col],
                    "Abs_Correlation": self.correlation_map.get(col, np.nan),
                    "Note": (
                        "Fixed on initial training window — see utils.py docstring"
                        if self.lag_map[col] > 0 else "Contemporaneous (dummy)"
                    ),
                }
            )
        return pd.DataFrame(records).sort_values(
            "Lead_Months", ascending=False
        )

import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.config import MAX_LEAD_MONTHS

logger = logging.getLogger(__name__)


class StationarityTester:
    """
    Validates that input series are stationary before model estimation.

    Non-stationary series violate the assumptions of OLS and PCA.
    The Augmented Dickey-Fuller test is applied with the null hypothesis
    of a unit root; rejection at the specified significance level
    confirms stationarity.
    """

    @staticmethod
    def is_stationary(series: pd.Series, significance: float = 0.05) -> bool:
        """
        Returns True if the ADF test rejects the unit root null.

        Parameters
        ----------
        series : pd.Series
            Time series to test. NaN values are dropped automatically.
        significance : float
            p-value threshold for rejection.
        """
        clean = series.dropna()
        if len(clean) < 20:
            logger.warning(
                "Insufficient observations (%d) for ADF test on '%s'.",
                len(clean),
                series.name,
            )
            return False

        adf_stat, p_value, _, _, _, _ = adfuller(clean, autolag="AIC")
        is_stat = p_value < significance
        logger.debug(
            "ADF test for '%s': stat=%.3f, p=%.4f -> %s",
            series.name,
            adf_stat,
            p_value,
            "stationary" if is_stat else "non-stationary",
        )
        return is_stat


class CycleAligner:
    """
    Identifies the optimal lead for each predictor variable relative
    to a target business cycle proxy.

    For each feature X_j, the aligner tests whether X_j(t-k) for
    k = 1, ..., max_lead maximizes the absolute correlation with
    the target Y(t). The selected lag k* represents the number of
    months by which X_j leads the real economy.

    This is the core mechanism for constructing a leading
    indicator system, as opposed to a coincident nowcast.

    Parameters
    ----------
    target : pd.Series
        The business cycle proxy (e.g., HP-filtered GDP or broad
        equity index) against which lead structure is optimized.
    """

    def __init__(self, target: pd.Series):
        self.target = target
        self.lag_map = {}
        self.correlation_map = {}

    def align(
        self, features: pd.DataFrame, max_lead: int = None
    ) -> pd.DataFrame:
        """
        Shifts each feature backward by its optimal lead and returns
        the aligned DataFrame.

        Parameters
        ----------
        features : pd.DataFrame
            Monthly indicator matrix.
        max_lead : int, optional
            Maximum number of months to test. Defaults to
            MAX_LEAD_MONTHS from config.

        Returns
        -------
        pd.DataFrame
            Features shifted by their optimal leads. Rows with
            insufficient history are dropped.
        """
        if max_lead is None:
            max_lead = MAX_LEAD_MONTHS

        aligned = pd.DataFrame(index=features.index)

        header = f"{'Variable':<25} {'Optimal Lead':>14} {'|Corr|':>8}"
        logger.info(header)
        logger.info("-" * len(header))

        for col in features.columns:
            # Skip dummy variables (they are contemporaneous by definition)
            if "Cuts" in col or "Impulse" in col:
                aligned[col] = features[col]
                self.lag_map[col] = 0
                self.correlation_map[col] = np.nan
                continue

            best_abs_corr = -1.0
            best_lag = 0

            for lag in range(1, max_lead + 1):
                shifted = features[col].shift(lag)
                corr = self.target.corr(shifted)
                if abs(corr) > best_abs_corr:
                    best_abs_corr = abs(corr)
                    best_lag = lag

            self.lag_map[col] = best_lag
            self.correlation_map[col] = best_abs_corr
            aligned[col] = features[col].shift(best_lag)

            logger.info(
                "%-25s %10d mo %8.4f", col, best_lag, best_abs_corr
            )

        aligned = aligned.dropna()
        logger.info(
            "Alignment complete: %d observations retained.", len(aligned)
        )
        return aligned

    def get_lead_summary(self) -> pd.DataFrame:
        """
        Returns a summary table of optimal leads and correlations,
        suitable for inclusion in output reports.
        """
        records = []
        for col in self.lag_map:
            records.append(
                {
                    "Variable": col,
                    "Lead_Months": self.lag_map[col],
                    "Abs_Correlation": self.correlation_map.get(col, np.nan),
                }
            )
        return pd.DataFrame(records).sort_values(
            "Lead_Months", ascending=False
        )
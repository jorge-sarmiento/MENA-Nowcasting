"""
features.py
-----------
Feature engineering for the MENA Nowcasting Engine.

Transforms raw stationary indicators into a richer signal set:
    1. Momentum: 3m and 6m price momentum (non-overlapping, standard definition)
    2. Volatility: Rolling standard deviation (risk regime proxy)
    3. Spreads: Cross-sectoral divergence signals

Applied AFTER stationarity transforms (YoY log-differences) and BEFORE
the CycleAligner.

NOTE ON SAMPLE LOSS — documented explicitly:
    The pipeline accumulates NaN rows at several stages:
        1. YoY log-diff (lag=12):     loses first 12 monthly observations
        2. Momentum windows (max=6):  loses 6 more monthly observations
        3. Volatility window (6):     loses 6 more monthly observations
        4. dropna() here:             takes the intersection of all above
    Net effect: ~18-24 months lost from the beginning of the raw series.
    For data starting 2010-01, the effective feature start is ~2012-Q1.
    This is documented in the README as "194 obs since 2010" for raw data
    vs. shorter effective estimation window.

References:
    - Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and
      Selling Losers: Implications for Stock Market Efficiency.
      Journal of Finance, 48(1), 65-91. DOI:10.1111/j.1540-6261.1993.tb04702.x
      (Standard definition of price momentum as non-overlapping returns)
    - Stock, J. & Watson, M. (2003). Forecasting Output and Inflation:
      The Role of Asset Prices. Journal of Economic Literature, 41(3), 788-829.
    - Kilian, L. (2009). Not All Oil Price Shocks Are Alike: Disentangling
      Demand and Supply Shocks in the Crude Oil Market.
      American Economic Review, 99(3), 1053-1069.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def add_momentum(df, windows=(3, 6)):
    """
    Computes price momentum over 3-month and 6-month horizons.

    DEFINITION — non-overlapping returns (Jegadeesh & Titman, 1993):
        mom_k(t) = r(t) - r(t-k)

    where r(t) is the YoY log-return at time t (already computed upstream
    by transform_to_stationary). This measures the CHANGE in the annual
    return rate over the past k months — i.e., acceleration or deceleration
    of the signal — not a cumulative sum.

    WHY NOT rolling().sum()?
        The previous implementation used rolling(k).sum() over YoY returns.
        Since YoY returns overlap by 11 months with each other, summing k
        consecutive YoY returns does NOT produce k-month momentum. Instead
        it produces a highly autocorrelated series (11/12 observations are
        shared). This inflates t-statistics in linear models.

        The correct non-overlapping momentum is the simple difference:
            mom_k(t) = r_YoY(t) - r_YoY(t-k)
        which measures whether the annual growth rate has accelerated
        (positive) or decelerated (negative) over k months.

    Ref: Jegadeesh & Titman (1993). Journal of Finance 48(1), 65-91.
    Ref: Asness, C., Moskowitz, T. & Pedersen, L. (2013). Value and Momentum
         Everywhere. Journal of Finance, 68(3), 929-985.
         DOI:10.1111/jofi.12021 (Section II.A defines momentum as r(t)-r(t-k))
    """
    new = pd.DataFrame(index=df.index)
    for col in df.columns:
        if "Cuts" in col or "Impulse" in col or "GT_" in col:
            continue
        for w in windows:
            # mom_w(t) = r(t) - r(t-w): acceleration of annual return
            new[f"{col}_mom{w}"] = df[col].diff(w)
    return new


def add_volatility(df, window=6):
    """
    Rolling standard deviation over 6 months.

    Measures the dispersion of YoY log-returns, which proxies for
    macroeconomic uncertainty. High-volatility regimes correlate with
    GDP contractions in oil-dependent economies (Hamilton, 2009).

    Applied to already-stationary (YoY) series so units are in percentage
    points — interpretable as annualised return volatility.

    Ref: Hamilton, J.D. (2009). Causes and Consequences of the Oil Shock
         of 2007-08. Brookings Papers on Economic Activity, Spring 2009.
         https://www.brookings.edu/bpea-articles/causes-and-consequences-of-the-oil-shock-of-2007-08/
    """
    new = pd.DataFrame(index=df.index)
    for col in df.columns:
        if "Cuts" in col or "Impulse" in col or "GT_" in col:
            continue
        new[f"{col}_vol6"] = df[col].rolling(window).std()
    return new


def add_spreads(df, country):
    """
    Cross-sectoral spreads: divergence between oil, equity, and banking signals.

    Spreads are computed on the stationary (YoY) series, so they represent
    the differential in annual growth rates across sectors. A widening spread
    between oil returns and equity returns signals compositional shifts in GDP
    growth (oil vs. non-oil), which is the central narrative in GCC economics.

    Ref: Kilian, L. (2009). Not All Oil Price Shocks Are Alike.
         American Economic Review, 99(3), 1053-1069.
    """
    new = pd.DataFrame(index=df.index)

    oil = "Brent_Oil" if "Brent_Oil" in df.columns else None
    eq = "Equity_Broad" if "Equity_Broad" in df.columns else None
    bank = "Banking" if "Banking" in df.columns else None

    if oil and eq:
        new["Spread_Oil_Equity"] = df[oil] - df[eq]
    if oil and bank:
        new["Spread_Oil_Bank"] = df[oil] - df[bank]
    if country == "KSA" and "Aramco" in df.columns and eq:
        new["Spread_Aramco_Equity"] = df["Aramco"] - df[eq]
    if country == "UAE" and "Real_Estate" in df.columns and bank:
        new["Spread_RE_Bank"] = df["Real_Estate"] - df[bank]

    return new


def engineer_features(df, country):
    """
    Master function: applies momentum, volatility, and spread engineering.

    Returns the union of original + engineered features, dropping any row
    where more than half the columns are NaN. The effective start date of
    the output is ~18-24 months after the input start due to rolling windows
    (see module docstring for details).
    """
    mom = add_momentum(df)
    vol = add_volatility(df)
    spreads = add_spreads(df, country)

    combined = pd.concat([df, mom, vol, spreads], axis=1)

    # Drop rows where more than half the engineered columns are NaN.
    # Full dropna() would be too aggressive here — Aramco's short history
    # (post-2019 IPO) would eliminate all pre-2020 data.
    min_valid = max(int(combined.shape[1] * 0.5), df.shape[1])
    combined = combined.dropna(thresh=min_valid)

    n_new = combined.shape[1] - df.shape[1]
    logger.info(
        "Feature engineering: %d original + %d engineered = %d total "
        "(%d obs after dropna, effective start ~%s)",
        df.shape[1], n_new, combined.shape[1],
        len(combined),
        combined.index[0].strftime("%Y-%m") if len(combined) > 0 else "N/A"
    )
    return combined
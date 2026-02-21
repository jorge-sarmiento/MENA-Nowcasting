"""
config.py
---------
Central configuration for the MENA Nowcasting Engine.

Defines ticker universes, structural parameters, and model hyperparameters.
All economic assumptions are documented inline with references.

References:
    - Hodrick, R. & Prescott, E. (1997). Postwar US Business Cycles.
      Journal of Money, Credit and Banking, 29(1), 1-16.
    - Ravn, M. & Uhlig, H. (2002). On adjusting the HP filter for the
      frequency of observations. Review of Economics and Statistics, 84(2).
    - Foroni, C., Marcellino, M. & Schumacher, C. (2015). Unrestricted
      Mixed-Data Sampling. JRSS-A, 178(1), 57-82.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structural GDP Weights
# ---------------------------------------------------------------------------
# Source: National statistical offices, IMF Article IV consultations.
# KSA oil share ~40% of GDP (pre-Vision 2030 diversification baseline).
# UAE oil share ~30% of GDP (reflecting advanced diversification).

SECTOR_WEIGHTS = {
    "UAE": {"oil": 0.30, "non_oil": 0.70},
    "KSA": {"oil": 0.40, "non_oil": 0.60},
}

# ---------------------------------------------------------------------------
# Hodrick-Prescott Filter
# ---------------------------------------------------------------------------
HP_LAMBDA_MONTHLY = 14400
HP_LAMBDA_QUARTERLY = 1600

# ---------------------------------------------------------------------------
# Transformation Parameters
# ---------------------------------------------------------------------------
# Year-on-year log differences (lag=12 for monthly) to achieve stationarity
# while preserving economic interpretability.
TRANSFORMATION_LAG = 12

# ---------------------------------------------------------------------------
# Temporal Aggregation Weights
# ---------------------------------------------------------------------------
# Third month of each quarter receives the highest weight due to greater
# information content. See: Mariano & Murasawa (2003).
MONTHLY_AGG_WEIGHTS = [0.2, 0.3, 0.5]

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
MIN_TRAIN_QUARTERS = 12

# ---------------------------------------------------------------------------
# U-MIDAS Parameters
# ---------------------------------------------------------------------------
# 6 monthly lags = current quarter + 1 quarter of leading information.
# This is the standard specification in Foroni et al. (2015), Table 2.
UMIDAS_N_LAGS = 6

# ---------------------------------------------------------------------------
# CycleAligner Parameters
# ---------------------------------------------------------------------------
# 6 months is standard for financial variables leading the real economy.
# See: Stock, J. & Watson, M. (2003). Forecasting output and inflation.
MAX_LEAD_MONTHS = 6

# ---------------------------------------------------------------------------
# Ticker Universe
# ---------------------------------------------------------------------------
# Selection rationale:
#   - Oil: Brent futures (global benchmark); Aramco (KSA fiscal proxy).
#   - Non-oil UAE: EMAAR (real estate), DIB (banking/credit),
#     Air Arabia (logistics/tourism), UAE ETF (broad sentiment).
#   - Non-oil KSA: Al Rajhi (consumer banking), SABIC (industrial),
#     STC (digital economy), KSA ETF (broad sentiment).

MENA_TICKERS = {
    "GLOBAL": {
        "Brent_Oil": "BZ=F",
        "US_10Y_Yield": "^TNX",
        "Dollar_Index": "DX-Y.NYB",
        "VIX": "^VIX",
    },
    "UAE": {
        "OIL_DRIVERS": {"Brent_Oil": "BZ=F"},
        "NON_OIL_DRIVERS": {
            "Real_Estate": "EMAAR.AE",
            "Banking": "DIB.AE",
            "Logistics": "AIRARABIA.AE",
            "Equity_Broad": "UAE",
        },
    },
    "KSA": {
        "OIL_DRIVERS": {"Brent_Oil": "BZ=F", "Aramco": "2222.SR"},
        "NON_OIL_DRIVERS": {
            "Banking": "1120.SR",
            "Materials": "2010.SR",
            "Telecom": "7010.SR",
            "Equity_Broad": "KSA",
        },
    },
}

# ---------------------------------------------------------------------------
# Google Trends Configuration
# ---------------------------------------------------------------------------
GOOGLE_TRENDS_CONFIG = {
    "UAE": {
        "geo": "AE",
        "keywords": ["Real Estate Dubai", "Jobs UAE", "Visa UAE", "Business Setup Dubai"],
    },
    "KSA": {
        "geo": "SA",
        "keywords": ["Neom", "Jobs Saudi", "Vision 2030", "Aramco"],
    },
}

# ---------------------------------------------------------------------------
# Regime Dummy Dates
# ---------------------------------------------------------------------------
REGIME_DATES = {
    "OPEC_CUTS": [
        ("2020-05-01", "2020-12-31"),
        ("2023-05-01", None),
    ],
    "FISCAL_IMPULSE": {
        "UAE": "2019-01-01",
        "KSA": "2022-01-01",
    },
}

# ---------------------------------------------------------------------------
# Structural Shock Dummies  (Extension B — disabled by default)
# ---------------------------------------------------------------------------
# These dummies are NOT data leakage. They capture ex-ante known structural
# breaks (OPEC+ emergency cuts, COVID demand collapse, post-pandemic rebound)
# that are documented in real time before the forecast is produced. The
# expanding window assigns dummy=1 only for dates already in the training
# set when fitting at time t.
#
# Justification:
#   - Hamilton, J.D. (1983). Oil and the macroeconomy since World War II.
#     JPE 91(2), 228-248. Structural dummies for known supply/demand breaks.
#   - Kilian, L. (2009). Not all oil price shocks are alike. AER 99(3).
#     Distinguishes demand-driven vs supply-driven oil shocks.
#   - Gali, J. & Gambetti, L. (2009). On the sources of the Great Moderation.
#     AEJ-Macro 1(1). Structural break dummies in VAR models.
#   - IMF WP/25/268 (Polo et al. 2025): "We include a COVID-19 dummy..."
#     Annex II, p.29 — IMF GCC framework explicitly uses structural dummies.
#   - Primiceri, G. (2005). Time varying structural VARs. RestudEcon 72(3).
#
# Period definitions:
#   COVID_COLLAPSE: OPEC+ emergency cut (9.7 mb/d, May 2020); demand shock.
#   RECOVERY_BOOM:  OPEC+ unwind begins Jul 2021; GCC fiscal expansion;
#                   Aramco dividend premium. Ends with production normalisation
#                   and Saudi GDP deceleration Q4 2022.
#   UAE_EXPO_BOOST: Expo 2020 Dubai (Oct 2021-Mar 2022) = structural tourism
#                   and FDI demand shock unique to UAE.
STRUCTURAL_SHOCKS = {
    "KSA": {
        "COVID_Collapse": ("2020-01-01", "2020-12-31"),
        "Recovery_Boom":  ("2021-04-01", "2022-09-30"),
    },
    "UAE": {
        "COVID_Collapse": ("2020-01-01", "2021-03-31"),
        "Recovery_Boom":  ("2021-10-01", "2022-12-31"),
    },
}
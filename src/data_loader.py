"""
data_loader.py
--------------
Financial data ingestion pipeline for the MENA Nowcasting Engine.

Fetches monthly price data from Yahoo Finance, applies stationarity
transformations (log-differences), and injects regime dummy variables.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from src.config import MENA_TICKERS, TRANSFORMATION_LAG, REGIME_DATES, GOOGLE_TRENDS_CONFIG
from src.google_trends import GoogleTrendsLoader

logger = logging.getLogger(__name__)


class MENADataLoader:
    """
    Handles data acquisition, transformation, and regime-tagging for
    GCC financial time series.
    """

    def __init__(self, start_date: str = "2010-01-01"):
        self.start_date = start_date
        self.end_date = datetime.today().strftime("%Y-%m-%d")

    def _fetch_single_ticker(self, ticker: str, name: str) -> pd.Series:
        """Downloads adjusted close prices. Returns None on failure."""
        try:
            raw = yf.download(
                ticker, start=self.start_date, end=self.end_date,
                progress=False, auto_adjust=False, timeout=15, threads=False,
            )
            if raw.empty:
                logger.warning("Empty response for %s (%s)", name, ticker)
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
            series = raw[price_col]
            series.name = name
            return series
        except (ConnectionError, TimeoutError) as e:
            logger.warning("Network error fetching %s: %s", ticker, e)
            return None
        except Exception as e:
            logger.warning("Unexpected error fetching %s (%s): %s", name, ticker, e)
            return None

    @staticmethod
    def _deduplicate_index(series: pd.Series, name: str) -> pd.Series:
        """
        Remove duplicate timestamps from a monthly series.

        Root cause of InvalidIndexError: yfinance occasionally returns daily
        data with two entries on the same calendar date (dividend adjustment
        rows, timezone boundary edge cases, or exchange-specific dual-listing
        artefacts). After resample("ME").last() these collapse to duplicate
        month-end timestamps when two daily rows share the same month-end date.

        Strategy: keep the last occurrence (most recent intraday value) for
        each duplicate timestamp. This is equivalent to what resample("ME").last()
        would have done if the source were clean.

        Ref: pandas docs — Index.is_unique required for axis=1 concat with
        non-overlapping column names but overlapping row indices.
        """
        if not series.index.is_unique:
            n_dupes = series.index.duplicated().sum()
            logger.warning(
                "%s: %d duplicate month-end timestamp(s) detected — "
                "keeping last occurrence. Likely cause: yfinance returned "
                "dual-listed or dividend-adjusted duplicate rows.",
                name, n_dupes
            )
            series = series[~series.index.duplicated(keep="last")]
        return series

    def _load_ticker_group(self, tickers_dict: dict) -> pd.DataFrame:
        """
        Downloads and resamples a dictionary of tickers to monthly.

        FIX — InvalidIndexError (pandas axis=1 concat):
            pd.concat([...], axis=1) raises InvalidIndexError when any
            DataFrame's row index contains duplicate timestamps. This happens
            because yfinance occasionally returns daily data with duplicate
            dates (dividend rows, timezone boundary artefacts, or dual-listing
            quirks on DFM/Tadawul). After resample("ME").last() these produce
            duplicate month-end index values.

            Fix: call _deduplicate_index() on every monthly Series before
            appending to series_list, and validate the merged DataFrame's
            index before returning.

            Secondary fix: generate_data_dictionary() in main.py also calls
            pd.concat([df_oil, df_nonoil], axis=1). Since we clean at the
            source here, that concat and all downstream concats are safe.
        """
        series_list = []
        for name, ticker in tickers_dict.items():
            logger.info("Fetching %s (%s)...", name, ticker)
            s = self._fetch_single_ticker(ticker, name)
            if s is None and ".AE" in ticker:
                alt = ticker.replace(".AE", ".DU")
                logger.info("Retrying with alternate exchange: %s", alt)
                s = self._fetch_single_ticker(alt, name)
            if s is not None:
                monthly = s.resample("ME").last()
                monthly.name = name
                # ── FIX: deduplicate before appending ──────────────────────
                monthly = self._deduplicate_index(monthly, name)
                series_list.append(monthly)
                logger.info("  %s: %d monthly observations", name, len(monthly))
            else:
                logger.warning("  %s: FAILED - excluded from pipeline", name)
        if not series_list:
            return pd.DataFrame()
        combined = pd.concat(series_list, axis=1)

        # Defensive post-concat validation: if somehow duplicates slipped
        # through (e.g. two tickers map to identical column names), deduplicate
        # the row index of the combined frame.
        if not combined.index.is_unique:
            n = combined.index.duplicated().sum()
            logger.warning(
                "_load_ticker_group: %d duplicate row index entries after concat — "
                "deduplicating. Check ticker universe for overlapping series.", n
            )
            combined = combined[~combined.index.duplicated(keep="last")]

        # Forward-fill gaps up to 3 months, but do NOT dropna here.
        # Short-history tickers (e.g. Aramco IPO 2019) would kill the
        # entire pre-2020 dataset. Let main.py handle missing data.
        # Ref: data_loader.py documentation — ffill(3) consequences.
        combined = combined.ffill(limit=3)
        return combined

    def _inject_regime_dummies(self, df, country, sector):
        """Adds binary indicators for known structural breaks."""
        df = df.copy()
        if sector == "oil":
            df["OPEC_Cuts"] = 0
            for start, end in REGIME_DATES["OPEC_CUTS"]:
                if end is not None:
                    df.loc[start:end, "OPEC_Cuts"] = 1
                else:
                    df.loc[start:, "OPEC_Cuts"] = 1
        if sector == "non_oil":
            df["Fiscal_Impulse"] = 0
            fiscal_start = REGIME_DATES["FISCAL_IMPULSE"].get(country)
            if fiscal_start:
                df.loc[fiscal_start:, "Fiscal_Impulse"] = 1
        return df

    def get_split_data(self, country: str):
        """Returns (oil_drivers_df, non_oil_drivers_df) with regime dummies."""
        if country not in MENA_TICKERS:
            raise ValueError(f"Country '{country}' not in ticker universe.")
        logger.info("=" * 60)
        logger.info("Ingesting split data for %s", country)

        df_oil = self._load_ticker_group(MENA_TICKERS[country]["OIL_DRIVERS"])
        df_nonoil = self._load_ticker_group(MENA_TICKERS[country]["NON_OIL_DRIVERS"])

        if not df_oil.empty:
            df_oil = self._inject_regime_dummies(df_oil, country, "oil")
        if not df_nonoil.empty:
            df_nonoil = self._inject_regime_dummies(df_nonoil, country, "non_oil")
            df_nonoil = self._merge_google_trends(df_nonoil, country)
        return df_oil, df_nonoil

    def _merge_google_trends(self, df_nonoil, country):
        """Merges Google Trends sentiment. Failure is non-fatal."""
        try:
            gt_loader = GoogleTrendsLoader(country)
            trends = gt_loader.fetch_trends(self.start_date)
            if trends is not None and not df_nonoil.empty:
                df_nonoil = df_nonoil.join(trends, how="left")
                if "GT_Sentiment" in df_nonoil.columns:
                    df_nonoil["GT_Sentiment"] = df_nonoil["GT_Sentiment"].ffill()
                    logger.info("Google Trends sentiment merged.")
        except Exception as e:
            logger.warning("Google Trends unavailable: %s. Continuing.", e)
        return df_nonoil

    @staticmethod
    def transform_to_stationary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies stationarity transformations:
          - Dummy variables: passed through unchanged.
          - Yield/VIX/sentiment: first differences.
          - All other series: year-on-year log differences.
        """
        lag = TRANSFORMATION_LAG
        transformed = pd.DataFrame(index=df.index)
        level_diff_vars = {"GT_Sentiment", "Google_Sentiment_Index"}

        for col in df.columns:
            if "Cuts" in col or "Impulse" in col:
                transformed[col] = df[col]
            elif "Yield" in col or "VIX" in col or col in level_diff_vars:
                transformed[col] = df[col].diff(lag)
            else:
                if (df[col] > 0).all():
                    transformed[col] = np.log(df[col]).diff(lag)
                else:
                    logger.warning("'%s' has non-positive values. Using pct_change.", col)
                    transformed[col] = df[col].pct_change(lag)

        result = transformed.dropna()
        logger.info("Stationarity transform: %d obs, %d features", len(result), result.shape[1])
        return result
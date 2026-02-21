"""
integration_diagnostics.py
--------------------------
Diagnostic tools for analyzing integration order of time series
and exporting raw/transformed data.

Features:
    1. ADF (Augmented Dickey-Fuller) test for unit roots
    2. KPSS test for stationarity confirmation
    3. Automatic integration order detection (I(0), I(1), I(2))
    4. Export raw and transformed data to CSV
    5. Summary report generation

Usage:
    python integration_diagnostics.py --country KSA
    python integration_diagnostics.py --country UAE --export-data
    python integration_diagnostics.py --country ALL

References:
    - Dickey, D.A. & Fuller, W.A. (1979). Distribution of the Estimators
      for Autoregressive Time Series with a Unit Root. JASA 74(366).
    - Kwiatkowski et al. (1992). Testing the null hypothesis of 
      stationarity against the alternative of a unit root. JoE 54(1-3).
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# Statistical tests
from statsmodels.tsa.stattools import adfuller, kpss

# Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger("integration_test")


# =========================================================================
# INTEGRATION ORDER TESTS
# =========================================================================

def adf_test(series, significance=0.05):
    """
    Augmented Dickey-Fuller test for unit root.
    
    H0: Series has a unit root (non-stationary)
    H1: Series is stationary
    
    Returns:
        dict with test statistic, p-value, critical values, and conclusion
    """
    series_clean = series.dropna()
    if len(series_clean) < 20:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "critical_values": {},
            "is_stationary": None,
            "conclusion": "Insufficient data (n < 20)"
        }
    
    try:
        result = adfuller(series_clean, autolag='AIC')
        statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        is_stationary = p_value < significance
        
        return {
            "statistic": round(statistic, 4),
            "p_value": round(p_value, 4),
            "critical_values": {k: round(v, 4) for k, v in critical_values.items()},
            "is_stationary": is_stationary,
            "conclusion": "Stationary (reject H0)" if is_stationary else "Non-stationary (fail to reject H0)"
        }
    except Exception as e:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "critical_values": {},
            "is_stationary": None,
            "conclusion": f"Test failed: {str(e)}"
        }


def kpss_test(series, significance=0.05):
    """
    KPSS test for stationarity.
    
    H0: Series is stationary
    H1: Series has a unit root (non-stationary)
    
    Note: KPSS is a confirmatory test - use alongside ADF.
    
    Returns:
        dict with test statistic, p-value, critical values, and conclusion
    """
    series_clean = series.dropna()
    if len(series_clean) < 20:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "critical_values": {},
            "is_stationary": None,
            "conclusion": "Insufficient data (n < 20)"
        }
    
    try:
        # regression='c' for level stationarity
        result = kpss(series_clean, regression='c', nlags='auto')
        statistic = result[0]
        p_value = result[1]
        critical_values = result[3]
        
        # For KPSS, we reject H0 (stationarity) if stat > critical value
        # or equivalently if p_value < significance
        is_stationary = p_value >= significance
        
        return {
            "statistic": round(statistic, 4),
            "p_value": round(p_value, 4),
            "critical_values": {k: round(v, 4) for k, v in critical_values.items()},
            "is_stationary": is_stationary,
            "conclusion": "Stationary (fail to reject H0)" if is_stationary else "Non-stationary (reject H0)"
        }
    except Exception as e:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "critical_values": {},
            "is_stationary": None,
            "conclusion": f"Test failed: {str(e)}"
        }


def determine_integration_order(series, max_diff=2, significance=0.05):
    """
    Determine the integration order of a time series.
    
    A series is I(d) if it needs to be differenced d times to become stationary.
    - I(0): Stationary in levels
    - I(1): Stationary after first difference
    - I(2): Stationary after second difference
    
    Uses both ADF and KPSS for robustness.
    
    Returns:
        dict with integration order and test results at each level
    """
    results = {
        "variable": series.name if hasattr(series, 'name') else "unknown",
        "n_obs": len(series.dropna()),
        "tests": {}
    }
    
    current_series = series.copy()
    
    for d in range(max_diff + 1):
        if d > 0:
            current_series = current_series.diff().dropna()
        
        level_name = f"d={d}" if d > 0 else "level"
        
        adf = adf_test(current_series, significance)
        kpss_result = kpss_test(current_series, significance)
        
        results["tests"][level_name] = {
            "ADF": adf,
            "KPSS": kpss_result
        }
        
        # Both tests agree on stationarity
        adf_stationary = adf.get("is_stationary", False)
        kpss_stationary = kpss_result.get("is_stationary", False)
        
        if adf_stationary and kpss_stationary:
            results["integration_order"] = d
            results["conclusion"] = f"I({d}) - Stationary after {d} difference(s)" if d > 0 else "I(0) - Stationary in levels"
            return results
        elif adf_stationary and not kpss_stationary:
            # Conflicting results - note it but continue
            results["tests"][level_name]["note"] = "ADF and KPSS disagree"
    
    # If we get here, series might be I(2) or higher
    results["integration_order"] = max_diff + 1
    results["conclusion"] = f"I({max_diff + 1}) or higher - May need more differencing"
    
    return results


def analyze_all_variables(df, significance=0.05):
    """
    Analyze integration order for all variables in a DataFrame.
    
    Returns:
        DataFrame with summary of integration orders
    """
    results = []
    
    for col in df.columns:
        logger.info(f"Testing: {col}")
        analysis = determine_integration_order(df[col], significance=significance)
        
        # Extract key info for summary
        level_adf = analysis["tests"].get("level", {}).get("ADF", {})
        level_kpss = analysis["tests"].get("level", {}).get("KPSS", {})
        
        results.append({
            "Variable": col,
            "N_Obs": analysis["n_obs"],
            "Integration_Order": analysis.get("integration_order", np.nan),
            "Conclusion": analysis.get("conclusion", ""),
            "ADF_Level_Stat": level_adf.get("statistic", np.nan),
            "ADF_Level_PValue": level_adf.get("p_value", np.nan),
            "KPSS_Level_Stat": level_kpss.get("statistic", np.nan),
            "KPSS_Level_PValue": level_kpss.get("p_value", np.nan),
        })
    
    return pd.DataFrame(results)


# =========================================================================
# DATA EXPORT
# =========================================================================

def export_data(df_raw, df_transformed, country, output_dir):
    """
    Export raw and transformed data to CSV files.
    
    Creates:
        - {country}_Raw_Data.csv: Original price/level data
        - {country}_Transformed_Data.csv: Stationary transformed data
        - {country}_Data_Summary.csv: Descriptive statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Raw data
    raw_path = f"{output_dir}/{country}_Raw_Data.csv"
    df_raw.to_csv(raw_path)
    logger.info(f"Raw data exported: {raw_path} ({df_raw.shape[0]} rows, {df_raw.shape[1]} cols)")
    
    # Transformed data
    trans_path = f"{output_dir}/{country}_Transformed_Data.csv"
    df_transformed.to_csv(trans_path)
    logger.info(f"Transformed data exported: {trans_path} ({df_transformed.shape[0]} rows, {df_transformed.shape[1]} cols)")
    
    # Summary statistics
    summary = pd.DataFrame({
        "Variable": df_raw.columns,
        "N_Obs_Raw": [df_raw[c].notna().sum() for c in df_raw.columns],
        "Start_Date": [df_raw[c].first_valid_index() for c in df_raw.columns],
        "End_Date": [df_raw[c].last_valid_index() for c in df_raw.columns],
        "Mean_Raw": [df_raw[c].mean() for c in df_raw.columns],
        "Std_Raw": [df_raw[c].std() for c in df_raw.columns],
        "Min_Raw": [df_raw[c].min() for c in df_raw.columns],
        "Max_Raw": [df_raw[c].max() for c in df_raw.columns],
    })
    
    # Add transformed stats for matching columns
    for col in df_transformed.columns:
        if col in summary["Variable"].values:
            idx = summary[summary["Variable"] == col].index[0]
            summary.loc[idx, "Mean_Transformed"] = df_transformed[col].mean()
            summary.loc[idx, "Std_Transformed"] = df_transformed[col].std()
    
    summary_path = f"{output_dir}/{country}_Data_Summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary exported: {summary_path}")
    
    return {
        "raw": raw_path,
        "transformed": trans_path,
        "summary": summary_path
    }


# =========================================================================
# REPORT GENERATION
# =========================================================================

# =========================================================================
# VISUALIZATION
# =========================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_variable_behavior(df_raw, df_transformed, country, output_dir):
    """
    Generate comprehensive plots for each variable showing:
    - Raw series (levels)
    - Transformed series (stationary)
    - Histogram of transformed values
    - ACF-like visualization
    
    Creates individual plots and a summary dashboard.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Color palette
    colors = {
        'raw': '#2C3E50',
        'transformed': '#E74C3C',
        'hist': '#3498DB',
        'grid': '#ECF0F1'
    }
    
    variables = [col for col in df_raw.columns if col in df_transformed.columns]
    n_vars = len(variables)
    
    logger.info(f"Generating plots for {n_vars} variables...")
    
    # --- Individual variable plots ---
    for col in variables:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{country} - {col}', fontsize=14, fontweight='bold')
        
        raw = df_raw[col].dropna()
        trans = df_transformed[col].dropna()
        
        # 1. Raw series (top left)
        ax1 = axes[0, 0]
        ax1.plot(raw.index, raw.values, color=colors['raw'], linewidth=1.2)
        ax1.set_title('Raw Series (Levels)', fontsize=11)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3, color=colors['grid'])
        ax1.tick_params(axis='x', rotation=45)
        
        # Add trend line
        if len(raw) > 10:
            z = np.polyfit(range(len(raw)), raw.values, 1)
            p = np.poly1d(z)
            ax1.plot(raw.index, p(range(len(raw))), '--', color='orange', 
                    alpha=0.7, label=f'Trend (slope={z[0]:.4f})')
            ax1.legend(fontsize=9)
        
        # 2. Transformed series (top right)
        ax2 = axes[0, 1]
        ax2.plot(trans.index, trans.values, color=colors['transformed'], linewidth=1.2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.axhline(y=trans.mean(), color='green', linestyle='--', linewidth=1, 
                   alpha=0.7, label=f'Mean={trans.mean():.4f}')
        ax2.fill_between(trans.index, trans.mean() - 2*trans.std(), trans.mean() + 2*trans.std(),
                        alpha=0.1, color='green', label='±2σ band')
        ax2.set_title('Transformed Series (Stationary)', fontsize=11)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3, color=colors['grid'])
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(fontsize=9)
        
        # 3. Histogram of transformed (bottom left)
        ax3 = axes[1, 0]
        n_bins = min(50, max(10, len(trans) // 5))
        ax3.hist(trans.values, bins=n_bins, color=colors['hist'], alpha=0.7, 
                edgecolor='white', density=True)
        
        # Overlay normal distribution
        from scipy import stats
        xmin, xmax = ax3.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        pdf = stats.norm.pdf(x, trans.mean(), trans.std())
        ax3.plot(x, pdf, 'r-', linewidth=2, label='Normal fit')
        
        # Add statistics
        skew = stats.skew(trans.values)
        kurt = stats.kurtosis(trans.values)
        ax3.set_title(f'Distribution (Skew={skew:.2f}, Kurt={kurt:.2f})', fontsize=11)
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Density')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, color=colors['grid'])
        
        # 4. Rolling statistics (bottom right)
        ax4 = axes[1, 1]
        window = min(12, len(trans) // 4)
        if window >= 3:
            rolling_mean = trans.rolling(window=window).mean()
            rolling_std = trans.rolling(window=window).std()
            
            ax4.plot(trans.index, trans.values, color=colors['transformed'], 
                    alpha=0.3, linewidth=0.8, label='Original')
            ax4.plot(rolling_mean.index, rolling_mean.values, color='blue', 
                    linewidth=1.5, label=f'Rolling Mean ({window}m)')
            ax4.plot(rolling_std.index, rolling_std.values, color='orange', 
                    linewidth=1.5, label=f'Rolling Std ({window}m)')
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            ax4.set_title('Rolling Statistics (Stationarity Check)', fontsize=11)
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Value')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, color=colors['grid'])
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor rolling stats', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # Save individual plot
        safe_col = col.replace('/', '_').replace('\\', '_')
        plot_path = f"{output_dir}/{country}_{safe_col}_Analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  Saved: {plot_path}")
    
    # --- Summary Dashboard ---
    plot_summary_dashboard(df_raw, df_transformed, country, output_dir, colors)
    
    return output_dir


def plot_summary_dashboard(df_raw, df_transformed, country, output_dir, colors):
    """
    Create a summary dashboard with all variables.
    """
    variables = [col for col in df_raw.columns if col in df_transformed.columns]
    n_vars = len(variables)
    
    if n_vars == 0:
        return
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    # --- Raw Data Dashboard ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle(f'{country} - Raw Data (All Variables)', fontsize=16, fontweight='bold', y=1.02)
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    for i, col in enumerate(variables):
        ax = axes[i]
        raw = df_raw[col].dropna()
        ax.plot(raw.index, raw.values, color=colors['raw'], linewidth=1)
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add min/max annotations
        ax.annotate(f'Min: {raw.min():.1f}', xy=(0.02, 0.02), xycoords='axes fraction', fontsize=7)
        ax.annotate(f'Max: {raw.max():.1f}', xy=(0.02, 0.10), xycoords='axes fraction', fontsize=7)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    raw_dashboard_path = f"{output_dir}/{country}_Dashboard_Raw.png"
    plt.savefig(raw_dashboard_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Raw dashboard saved: {raw_dashboard_path}")
    
    # --- Transformed Data Dashboard ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle(f'{country} - Transformed Data (All Variables)', fontsize=16, fontweight='bold', y=1.02)
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    for i, col in enumerate(variables):
        ax = axes[i]
        trans = df_transformed[col].dropna()
        ax.plot(trans.index, trans.values, color=colors['transformed'], linewidth=1)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axhline(y=trans.mean(), color='green', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.fill_between(trans.index, -2*trans.std(), 2*trans.std(), alpha=0.1, color='green')
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add std annotation
        ax.annotate(f'σ: {trans.std():.4f}', xy=(0.02, 0.02), xycoords='axes fraction', fontsize=7)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    trans_dashboard_path = f"{output_dir}/{country}_Dashboard_Transformed.png"
    plt.savefig(trans_dashboard_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Transformed dashboard saved: {trans_dashboard_path}")
    
    # --- Correlation Heatmap ---
    plot_correlation_heatmap(df_transformed, country, output_dir)


def plot_correlation_heatmap(df, country, output_dir):
    """
    Create correlation heatmap of transformed variables.
    """
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    
    if numeric_df.shape[1] < 2:
        return
    
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    im = ax.imshow(corr.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', fontsize=10)
    
    # Set ticks
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    
    # Add correlation values as text
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   fontsize=7, color=color)
    
    ax.set_title(f'{country} - Correlation Matrix (Transformed Data)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    heatmap_path = f"{output_dir}/{country}_Correlation_Heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Correlation heatmap saved: {heatmap_path}")


def generate_integration_report(integration_df, country, output_dir):
    """
    Generate a formatted report of integration order analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV export
    csv_path = f"{output_dir}/{country}_Integration_Order.csv"
    integration_df.to_csv(csv_path, index=False)
    logger.info(f"Integration order analysis exported: {csv_path}")
    
    # Summary by integration order
    order_counts = integration_df["Integration_Order"].value_counts().sort_index()
    
    logger.info("=" * 60)
    logger.info(f"INTEGRATION ORDER SUMMARY: {country}")
    logger.info("=" * 60)
    
    for order, count in order_counts.items():
        vars_at_order = integration_df[integration_df["Integration_Order"] == order]["Variable"].tolist()
        logger.info(f"I({int(order)}): {count} variables")
        for v in vars_at_order:
            logger.info(f"    - {v}")
    
    logger.info("=" * 60)
    
    # Recommendations
    i2_vars = integration_df[integration_df["Integration_Order"] >= 2]["Variable"].tolist()
    if i2_vars:
        logger.warning("Variables with I(2) or higher may need special treatment:")
        for v in i2_vars:
            logger.warning(f"    - {v}")
    
    return csv_path


# =========================================================================
# MAIN PIPELINE
# =========================================================================

def run_diagnostics(country, export_data_flag=True, generate_plots=True,
                    output_dir=None, df_oil_override=None, df_nonoil_override=None):
    """
    Run full diagnostics for a country.

    Args:
        country: 'KSA' or 'UAE'
        export_data_flag: If True, export raw and transformed data to CSV.
            This gives users direct visibility into raw prices and the
            YoY log-diff transformed series before any model runs.
        generate_plots: If True, generate visualization plots (raw + transformed
            time series, histograms, rolling statistics, correlation heatmap).
        output_dir: Output directory (default: outputs/{country}/diagnostics)
        df_oil_override: Pre-loaded oil DataFrame from data_loader.
            If provided, skips downloading. Used when called from main.py
            run_country() to reuse already-downloaded data.
        df_nonoil_override: Pre-loaded non-oil DataFrame from data_loader.

    Returns:
        dict with integration_raw, integration_transformed, export_paths, summary
    """
    if output_dir is None:
        output_dir = f"outputs/{country.lower()}/diagnostics"

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"INTEGRATION DIAGNOSTICS: {country}")
    logger.info("=" * 70)

    # Load data — use overrides if provided (avoids double-downloading)
    if df_oil_override is not None and df_nonoil_override is not None:
        df_oil, df_nonoil = df_oil_override.copy(), df_nonoil_override.copy()
        logger.info("Using pre-loaded DataFrames (override mode).")
    else:
        try:
            from src.data_loader import MENADataLoader
            loader = MENADataLoader()
            df_oil, df_nonoil = loader.get_split_data(country)
        except ImportError:
            logger.error("Could not import MENADataLoader. Run from project root.")
            return None

    # Defensive deduplication (mirrors data_loader._deduplicate_index)
    for _df, _lbl in [(df_oil, "df_oil"), (df_nonoil, "df_nonoil")]:
        if not _df.index.is_unique:
            n = int(_df.index.duplicated().sum())
            logger.warning("run_diagnostics: %s has %d duplicate index entries — deduplicating.", _lbl, n)
            _df = _df[~_df.index.duplicated(keep="last")]

    # Combine raw data
    df_raw = pd.concat([df_oil, df_nonoil], axis=1)
    # Post-concat dedup (belt-and-suspenders)
    if not df_raw.index.is_unique:
        df_raw = df_raw[~df_raw.index.duplicated(keep="last")]

    logger.info(f"Raw data: {df_raw.shape[0]} months × {df_raw.shape[1]} variables")
    
    # Transform to stationary
    from src.config import TRANSFORMATION_LAG
    
    def transform_stationary(df):
        lag = TRANSFORMATION_LAG
        out = pd.DataFrame(index=df.index)
        for col in df.columns:
            if "Cuts" in col or "Impulse" in col:
                out[col] = df[col]
            elif "Yield" in col or "VIX" in col:
                out[col] = df[col].diff(lag)
            else:
                if (df[col] > 0).all():
                    out[col] = np.log(df[col]).diff(lag)
                else:
                    out[col] = df[col].pct_change(lag)
        return out
    
    df_transformed = transform_stationary(df_raw)
    df_transformed = df_transformed.dropna(how='all')
    
    # Integration order analysis on RAW data (before transformation)
    logger.info("\n--- Integration Order Analysis (Raw Data) ---")
    integration_raw = analyze_all_variables(df_raw)
    integration_raw_path = generate_integration_report(integration_raw, f"{country}_Raw", output_dir)
    
    # Integration order analysis on TRANSFORMED data (verify stationarity)
    logger.info("\n--- Integration Order Analysis (Transformed Data) ---")
    integration_trans = analyze_all_variables(df_transformed)
    integration_trans_path = generate_integration_report(integration_trans, f"{country}_Transformed", output_dir)
    
    # Export data if requested
    export_paths = None
    if export_data_flag:
        logger.info("\n--- Exporting Data ---")
        export_paths = export_data(df_raw, df_transformed, country, output_dir)
    
    # Generate plots
    if generate_plots:
        logger.info("\n--- Generating Visualizations ---")
        plot_variable_behavior(df_raw, df_transformed, country, output_dir)
    
    # Summary comparison
    logger.info("\n" + "=" * 70)
    logger.info("TRANSFORMATION EFFECTIVENESS")
    logger.info("=" * 70)
    
    raw_stationary = (integration_raw["Integration_Order"] == 0).sum()
    trans_stationary = (integration_trans["Integration_Order"] == 0).sum()
    total = len(integration_raw)
    
    logger.info(f"Raw data:         {raw_stationary}/{total} variables stationary I(0)")
    logger.info(f"Transformed data: {trans_stationary}/{total} variables stationary I(0)")
    logger.info(f"Improvement:      +{trans_stationary - raw_stationary} variables")
    
    # Flag any transformed variables still non-stationary
    still_nonstationary = integration_trans[integration_trans["Integration_Order"] > 0]["Variable"].tolist()
    if still_nonstationary:
        logger.warning("\nVariables still non-stationary after transformation:")
        for v in still_nonstationary:
            order = integration_trans[integration_trans["Variable"] == v]["Integration_Order"].values[0]
            logger.warning(f"    - {v}: I({int(order)})")
    
    return {
        "country": country,
        "integration_raw": integration_raw,
        "integration_transformed": integration_trans,
        "export_paths": export_paths,
        "summary": {
            "raw_stationary": raw_stationary,
            "transformed_stationary": trans_stationary,
            "total_variables": total,
            "still_nonstationary": still_nonstationary
        }
    }


# =========================================================================
# ENTRY POINT
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Integration Order Diagnostics")
    parser.add_argument("--country", default="ALL", choices=["KSA", "UAE", "ALL"],
                        help="Country to analyze")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip data export to CSV")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--output-dir", default=None,
                        help="Custom output directory")
    args = parser.parse_args()
    
    countries = ["KSA", "UAE"] if args.country == "ALL" else [args.country]
    
    all_results = {}
    for country in countries:
        result = run_diagnostics(
            country,
            export_data_flag=not args.no_export,
            generate_plots=not args.no_plots,
            output_dir=args.output_dir
        )
        if result:
            all_results[country] = result
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    for country, result in all_results.items():
        s = result["summary"]
        logger.info(f"{country}: {s['transformed_stationary']}/{s['total_variables']} stationary after transformation")
        if s["still_nonstationary"]:
            logger.info(f"    Still non-stationary: {', '.join(s['still_nonstationary'])}")


if __name__ == "__main__":
    main()
"""
validation.py
-------------
Model evaluation, statistical validation, and visualization module.

Implements:
    - GDP ground truth (GASTAT quarterly for KSA, interpolated for UAE)
    - Expanding-window out-of-sample evaluation
    - Statistical comparison metrics (RMSE, MAE, directional accuracy)
    - Six formal statistical tests to demonstrate forecast quality
    - Diebold-Mariano test for model comparison
    - Institutional-quality visualization suite
    - Model coefficient / feature importance charts

Statistical Tests:
    1. Mincer-Zarnowitz Regression (Forecast Efficiency)
       Ref: Mincer, J. & Zarnowitz, V. (1969). Evaluation of Economic
            Forecasts. In Economic Forecasts and Expectations, NBER.
    2. Pesaran-Timmermann Directional Accuracy Test
       Ref: Pesaran, M.H. & Timmermann, A. (1992). A Simple Nonparametric
            Test of Predictive Performance. JBES, 10(4), 461-465.
    3. Naive Benchmark Comparison (Theil U-Statistic)
       Ref: Theil, H. (1966). Applied Economic Forecasting. North-Holland.
    4. In-Sample vs Out-of-Sample Gap Analysis
    5. Permutation Test (Randomization Inference)
       Ref: Good, P. (2005). Permutation, Parametric, and Bootstrap Tests
            of Hypotheses. Springer.
    6. Recursive RMSE Stability (Rolling Window Diagnostic)
       Ref: Giacomini, R. & Rossi, B. (2010). Forecast Comparisons in
            Unstable Environments. J. Applied Econometrics, 25(4), 595-620.

    Diebold-Mariano Test:
       Ref: Diebold, F. & Mariano, R. (1995). Comparing Predictive
            Accuracy. JBES, 13(3), 253-263.
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as sp_stats
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color Palette (Institutional)
# ---------------------------------------------------------------------------
COLORS = {
    "gdp_actual": "#1B2838", "dfm": "#2980B9", "bridge": "#27AE60",
    "gbm": "#6C3483", "umidas": "#D35400",
    "error_pos": "#3498DB", "error_neg": "#E74C3C",
    "confidence": "#BDC3C7", "scatter_fit": "#C0392B",
}
MODEL_COLORS = {
    "DFM": "#2980B9",
    "Bridge": "#27AE60",
    "ElasticNet": "#F39C12",
    "GBM": "#6C3483",
    "U-MIDAS": "#D35400",
    "Ensemble": "#1ABC9C",
}


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC", "axes.grid": True,
        "grid.color": "#E8E8E8", "grid.linewidth": 0.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC", "figure.dpi": 150,
    })


# =========================================================================
# GDP GROUND TRUTH
# =========================================================================
class GDPGroundTruth:
    """
    Official GDP growth data for validation.
    KSA: GASTAT quarterly. UAE: FCSC/IMF annual, linearly interpolated.
    """

    _KSA_QUARTERLY = {
        # Source: GASTAT quarterly GDP bulletins (YoY real growth, %)
        "2010-03-31": 4.1, "2010-06-30": 3.8, "2010-09-30": 4.5, "2010-12-31": 5.0,
        "2011-03-31": 7.5, "2011-06-30": 8.2, "2011-09-30": 9.3, "2011-12-31": 8.6,
        "2012-03-31": 5.4, "2012-06-30": 5.1, "2012-09-30": 5.5, "2012-12-31": 5.8,
        "2013-03-31": 3.3, "2013-06-30": 1.7, "2013-09-30": 2.8, "2013-12-31": 2.7,
        "2014-03-31": 4.0, "2014-06-30": 4.5, "2014-09-30": 3.2, "2014-12-31": 2.8,
        "2015-03-31": 2.5, "2015-06-30": 1.3, "2015-09-30": 2.2, "2015-12-31": 1.0,
        "2016-03-31": 1.8, "2016-06-30": 3.7, "2016-09-30": 1.2, "2016-12-31": 0.2,
        "2017-03-31": -2.0, "2017-06-30": -1.0, "2017-09-30": -0.5, "2017-12-31": 0.6,
        "2018-03-31": 1.2, "2018-06-30": 1.7, "2018-09-30": 3.1, "2018-12-31": 3.6,
        "2019-03-31": 1.2, "2019-06-30": 0.5, "2019-09-30": -0.5, "2019-12-31": 0.0,
        "2020-03-31": -1.1, "2020-06-30": -7.0, "2020-09-30": -4.6, "2020-12-31": -3.9,
        "2021-03-31": -1.1, "2021-06-30": 6.8, "2021-09-30": 6.8, "2021-12-31": 1.6,
        "2022-03-31": 9.9, "2022-06-30": 12.2, "2022-09-30": 8.8, "2022-12-31": 3.9,
        "2023-03-31": 3.9, "2023-06-30": -0.4, "2023-09-30": -3.0, "2023-12-31": -2.6,
        "2024-03-31": 2.8, "2024-06-30": 1.4, "2024-09-30": 2.3, "2024-12-31": 4.4,
        "2025-03-31": 4.9,
    }

    _UAE_ANNUAL = {
        # Source: FCSC, World Bank, IMF WEO (real GDP growth, %)
        "2009-12-31": -5.2, "2010-12-31": 1.6, "2011-12-31": 4.9,
        "2012-12-31": 4.5, "2013-12-31": 4.3, "2014-12-31": 4.4,
        "2015-12-31": 5.1, "2016-12-31": 3.1, "2017-12-31": 2.4,
        "2018-12-31": 1.2, "2019-12-31": 1.7, "2020-12-31": -4.8,
        "2021-12-31": 3.8, "2022-12-31": 7.6, "2023-12-31": 3.4,
        "2024-12-31": 4.0,
    }

    @classmethod
    def get_quarterly(cls, country):
        if country == "KSA":
            dates = pd.to_datetime(list(cls._KSA_QUARTERLY.keys()))
            return pd.Series(list(cls._KSA_QUARTERLY.values()), index=dates, name="GDP_YoY")
        elif country == "UAE":
            dates = pd.to_datetime(list(cls._UAE_ANNUAL.keys()))
            annual = pd.Series(list(cls._UAE_ANNUAL.values()), index=dates)
            q = annual.resample("QE").interpolate("linear").dropna()
            logger.warning("UAE quarterly GDP is interpolated from annual data.")
            return q
        raise ValueError(f"No GDP data for: {country}")


# =========================================================================
# EVALUATION METRICS
# =========================================================================
def compute_metrics(actual, forecast):
    common = actual.index.intersection(forecast.index)
    y, yhat = actual.loc[common], forecast.loc[common]
    if len(common) < 3:
        return {"RMSE": np.nan, "MAE": np.nan, "DirAcc": np.nan, "Bias": np.nan, "Corr": np.nan, "N": 0}
    rmse = np.sqrt(mean_squared_error(y, yhat))
    mae = np.mean(np.abs(y - yhat))
    bias = np.mean(yhat - y)
    corr = np.corrcoef(y, yhat)[0, 1]
    dy = np.sign(y.diff().dropna())
    dyhat = np.sign(yhat.diff().dropna())
    ci = dy.index.intersection(dyhat.index)
    dir_acc = (dy.loc[ci] == dyhat.loc[ci]).mean() * 100 if len(ci) > 0 else np.nan
    return {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "Bias": round(bias, 3),
            "DirAcc_%": round(dir_acc, 1), "Corr": round(corr, 3), "N": len(common)}


def diebold_mariano_test(actual, fc1, fc2):
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.
    H0: E[d_t] = 0, where d_t = e1_t^2 - e2_t^2.
    """
    common = actual.index.intersection(fc1.index).intersection(fc2.index)
    e1 = (actual.loc[common] - fc1.loc[common]).values
    e2 = (actual.loc[common] - fc2.loc[common]).values
    d = e1**2 - e2**2
    n = len(d)
    if n < 5:
        return {"DM_stat": np.nan, "p_value": np.nan}
    dm = np.mean(d) / np.sqrt(np.var(d, ddof=1) / n)
    p = 2 * (1 - sp_stats.t.cdf(abs(dm), df=n-1))
    return {"DM_stat": round(dm, 3), "p_value": round(p, 4)}


# =========================================================================
# 6 STATISTICAL VALIDATION TESTS
# =========================================================================

def mincer_zarnowitz_test(actual, forecast):
    """
    Tests H0: alpha=0, beta=1 jointly (forecast efficiency).
    Ref: Mincer & Zarnowitz (1969).
    """
    common = actual.index.intersection(forecast.index)
    y, yhat = actual.loc[common].values, forecast.loc[common].values
    n = len(y)
    X = np.column_stack([np.ones(n), yhat])
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha, beta = beta_hat[0], beta_hat[1]
    resid = y - X @ beta_hat
    sigma2 = np.sum(resid**2) / (n - 2)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_a, se_b = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    R = np.array([[1, 0], [0, 1]])
    r = np.array([0, 1])
    diff = R @ beta_hat - r
    F_stat = (diff.T @ np.linalg.inv(R @ cov @ R.T) @ diff) / 2
    p_joint = 1 - sp_stats.f.cdf(F_stat, 2, n-2)
    verdict = "PASS" if p_joint > 0.05 else "FAIL"
    return {"alpha": round(alpha, 4), "beta": round(beta, 4),
            "se_alpha": round(se_a, 4), "se_beta": round(se_b, 4),
            "F_stat": round(F_stat, 3), "p_joint": round(p_joint, 4), "verdict": verdict}


def pesaran_timmermann_test(actual, forecast):
    """
    Tests directional accuracy vs random guessing.
    Ref: Pesaran & Timmermann (1992).
    """
    common = actual.index.intersection(forecast.index)
    dy = actual.loc[common].diff().dropna()
    dyhat = forecast.loc[common].diff().dropna()
    ci = dy.index.intersection(dyhat.index)
    dy, dyhat = dy.loc[ci], dyhat.loc[ci]
    n = len(dy)
    hits = ((dy > 0) & (dyhat > 0)) | ((dy <= 0) & (dyhat <= 0))
    P_hat = hits.mean()
    py, pyhat = (dy > 0).mean(), (dyhat > 0).mean()
    P_star = py * pyhat + (1 - py) * (1 - pyhat)
    var_Phat = P_star * (1 - P_star) / n
    var_extra = ((2*py-1)**2 * pyhat*(1-pyhat)/n + (2*pyhat-1)**2 * py*(1-py)/n
                 + 4*py*pyhat*(1-py)*(1-pyhat)/(n**2))
    denom = np.sqrt(var_Phat + var_extra)
    if denom < 1e-10:
        return {"hit_rate_%": round(P_hat*100, 1), "verdict": "INCONCLUSIVE"}
    PT = (P_hat - P_star) / denom
    p = 1 - sp_stats.norm.cdf(PT)
    return {"hit_rate_%": round(P_hat*100, 1), "expected_%": round(P_star*100, 1),
            "PT_stat": round(PT, 3), "p_value": round(p, 4),
            "verdict": "PASS" if p < 0.10 else "FAIL"}


def naive_benchmark_test(actual, forecast):
    """
    Compares RMSE against historical mean and random walk.
    Theil U < 1 indicates the model outperforms the benchmark.
    Ref: Theil (1966).
    """
    common = actual.index.intersection(forecast.index)
    y, yhat = actual.loc[common], forecast.loc[common]
    rmse_m = np.sqrt(np.mean((y - yhat)**2))
    naive_mean = y.expanding().mean().shift(1).dropna()
    ci_m = y.index.intersection(naive_mean.index)
    rmse_mean = np.sqrt(np.mean((y.loc[ci_m] - naive_mean.loc[ci_m])**2))
    naive_rw = y.shift(1).dropna()
    ci_r = y.index.intersection(naive_rw.index)
    rmse_rw = np.sqrt(np.mean((y.loc[ci_r] - naive_rw.loc[ci_r])**2))
    u_mean = rmse_m / rmse_mean if rmse_mean > 0 else np.nan
    u_rw = rmse_m / rmse_rw if rmse_rw > 0 else np.nan
    verdict = "PASS" if (u_mean < 1 and u_rw < 1) else "MARGINAL" if (u_mean < 1 or u_rw < 1) else "FAIL"
    return {"Model_RMSE": round(rmse_m, 3), "HistMean_RMSE": round(rmse_mean, 3),
            "RW_RMSE": round(rmse_rw, 3), "Theil_U_Mean": round(u_mean, 3),
            "Theil_U_RW": round(u_rw, 3), "verdict": verdict}


def overfitting_gap_test(actual, forecast, min_train=16):
    """
    Compares first-half vs second-half OOS RMSE to detect temporal
    degradation in forecast accuracy. A ratio substantially above 1
    suggests the model overfits early training samples.

    Note: Both halves are genuinely out-of-sample (from the expanding
    window). This test checks whether OOS performance is stable over
    time, not IS vs OOS.
    """
    common = actual.index.intersection(forecast.index)
    y, yhat = actual.loc[common], forecast.loc[common]
    midpoint = len(y) // 2
    if midpoint < 4:
        return {"RMSE_IS": np.nan, "RMSE_OOS": np.nan, "Ratio": np.nan, "verdict": "INSUFFICIENT DATA"}
    rmse_first = np.sqrt(np.mean((y.iloc[:midpoint] - yhat.iloc[:midpoint])**2))
    rmse_second = np.sqrt(np.mean((y.iloc[midpoint:] - yhat.iloc[midpoint:])**2))
    ratio = rmse_second / rmse_first if rmse_first > 0 else np.nan
    verdict = "PASS" if ratio < 1.5 else "CAUTION" if ratio < 2.0 else "FAIL"
    return {"RMSE_IS": round(rmse_first, 3), "RMSE_OOS": round(rmse_second, 3),
            "Ratio": round(ratio, 3), "verdict": verdict}


def permutation_test(actual, forecast, n_perm=1000):
    """
    Shuffles GDP 1000x. If actual corr > 95th percentile, signal is real.
    Ref: Good (2005).
    """
    common = actual.index.intersection(forecast.index)
    y, yhat = actual.loc[common].values, forecast.loc[common].values
    actual_corr = np.corrcoef(y, yhat)[0, 1]
    np.random.seed(2026)
    shuffled = [np.corrcoef(np.random.permutation(y), yhat)[0, 1] for _ in range(n_perm)]
    shuffled = np.array(shuffled)
    p = np.mean(np.abs(shuffled) >= abs(actual_corr))
    pctile = (np.abs(shuffled) < abs(actual_corr)).mean() * 100
    return {"actual_corr": round(actual_corr, 4), "mean_shuffled": round(np.mean(np.abs(shuffled)), 4),
            "percentile": round(pctile, 1), "p_value": round(p, 4),
            "verdict": "PASS" if p < 0.05 else "FAIL", "_shuffled": shuffled}


def recursive_rmse_stability(actual, forecast, window=8):
    """
    Rolling RMSE stability. CV < 0.5 = stable performance.
    Ref: Giacomini & Rossi (2010).
    """
    common = actual.index.intersection(forecast.index)
    errors = (actual.loc[common] - forecast.loc[common]).values
    if len(errors) < window + 2:
        return {"verdict": "INSUFFICIENT DATA"}
    rolling = []
    dates = []
    for i in range(len(errors) - window + 1):
        rolling.append(np.sqrt(np.mean(errors[i:i+window]**2)))
        dates.append(common[i + window - 1])
    rolling = np.array(rolling)
    cv = np.std(rolling) / np.mean(rolling)
    verdict = "PASS" if cv < 0.50 else "CAUTION" if cv < 0.75 else "FAIL"
    return {"mean_RMSE": round(np.mean(rolling), 3), "CV": round(cv, 3),
            "verdict": verdict, "_rolling": rolling, "_dates": dates}


def run_all_statistical_tests(actual, forecast, min_train=16):
    """Runs all 6 tests and returns consolidated results dict."""
    results = {
        "mincer_zarnowitz": mincer_zarnowitz_test(actual, forecast),
        "pesaran_timmermann": pesaran_timmermann_test(actual, forecast),
        "naive_benchmark": naive_benchmark_test(actual, forecast),
        "overfitting_gap": overfitting_gap_test(actual, forecast, min_train),
        "permutation": permutation_test(actual, forecast),
        "recursive_stability": recursive_rmse_stability(actual, forecast),
    }
    verdicts = [r["verdict"] for r in results.values() if "verdict" in r]
    results["_passes"] = verdicts.count("PASS")
    results["_total"] = len(verdicts)
    return results


# =========================================================================
# VISUALIZATION: HORSE RACE DASHBOARD
# =========================================================================
def plot_dashboard(country, forecasts, gdp, metrics, best_model, out_dir):
    _apply_style()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

    best_fc = forecasts[best_model]
    common = gdp.index.intersection(best_fc.index)

    # Panel 1: Horse Race
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(gdp.index, gdp.values, color="#D5DBDB", alpha=0.4, zorder=0)
    ax1.plot(gdp.index, gdp.values, color=COLORS["gdp_actual"], linewidth=2.5, label="Actual GDP", zorder=5)
    for model, fc in forecasts.items():
        is_best = model == best_model
        ax1.plot(fc.index, fc.values, color=MODEL_COLORS.get(model, "#999"),
                 linewidth=2.5 if is_best else 1.0, alpha=1.0 if is_best else 0.35,
                 linestyle="--" if is_best else ":",
                 label=f"{model} {'[BEST]' if is_best else ''} (RMSE: {metrics[model]['RMSE']})")
    ax1.set_title(f"{country}: Model Horse Race -- Quarterly GDP Growth", fontsize=14, fontweight="bold")
    ax1.set_ylabel("YoY Growth (%)"); ax1.legend(loc="upper left", ncol=3)
    ax1.axhline(0, color="#AAAAAA", linewidth=0.8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 2: Residuals
    ax2 = fig.add_subplot(gs[1, 0])
    errors = best_fc.loc[common] - gdp.loc[common]
    colors = ["#3498DB" if e >= 0 else "#E74C3C" for e in errors]
    ax2.bar(common, errors, color=colors, alpha=0.7, width=50)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title(f"Residual Analysis ({best_model})", fontweight="bold")
    ax2.set_ylabel("Error (pp)"); ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 3: Scatter
    ax3 = fig.add_subplot(gs[1, 1])
    y, yhat = gdp.loc[common].values, best_fc.loc[common].values
    ax3.scatter(y, yhat, color=MODEL_COLORS.get(best_model, "#2980B9"), alpha=0.6, s=55, edgecolors="white")
    slope, intercept, r, _, _ = sp_stats.linregress(y, yhat)
    xline = np.linspace(y.min(), y.max(), 100)
    ax3.plot(xline, slope*xline + intercept, color="#C0392B", linewidth=2)
    lims = [min(y.min(), yhat.min())-1, max(y.max(), yhat.max())+1]
    ax3.plot(lims, lims, color="#CCCCCC", linestyle="--", linewidth=1, zorder=0)
    ax3.set_xlim(lims); ax3.set_ylim(lims)
    ax3.text(0.05, 0.92, f"Corr = {metrics[best_model]['Corr']:.3f}\nSlope = {slope:.2f}",
             transform=ax3.transAxes, fontsize=11,
             bbox=dict(facecolor="white", alpha=0.9, edgecolor="#CCCCCC"))
    ax3.set_title("Predicted vs Observed", fontweight="bold")
    ax3.set_xlabel("Actual GDP (%)"); ax3.set_ylabel("Forecast GDP (%)")

    plt.tight_layout()
    fp = f"{out_dir}/{country}_Dashboard.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    logger.info("Dashboard saved: %s", fp)


# =========================================================================
# VISUALIZATION: ACTUALS VS PREDICTED (DUAL PANEL)
# =========================================================================
def plot_actuals_vs_predicted(country, gdp, forecasts, metrics, out_dir):
    _apply_style()
    best = min(metrics, key=lambda m: metrics[m]["RMSE"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [2.5, 1]}, sharex=True)

    ax1.plot(gdp.index, gdp.values, color=COLORS["gdp_actual"], linewidth=2.5,
             label="Actual GDP (YoY %)", marker="o", markersize=3.5, zorder=10)
    for model, fc in forecasts.items():
        is_best = model == best
        ax1.plot(fc.index, fc.values, color=MODEL_COLORS.get(model, "#999"),
                 linewidth=2.2 if is_best else 1.0, alpha=1.0 if is_best else 0.4,
                 linestyle="--" if is_best else ":",
                 label=f"{model} (RMSE: {metrics[model]['RMSE']}pp)" if is_best else f"{model} ({metrics[model]['RMSE']}pp)")
    ax1.axhline(0, color="#AAAAAA", linewidth=0.8)
    ax1.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-30"), color="#FFE0E0", alpha=0.3, zorder=0)
    ax1.set_title(f"{country}: Quarterly GDP Nowcast vs Actual (Out-of-Sample)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Real GDP Growth, YoY (%)"); ax1.legend(loc="upper left", ncol=2)

    best_fc = forecasts[best]
    common = gdp.index.intersection(best_fc.index)
    errors = best_fc.loc[common] - gdp.loc[common]
    bar_c = ["#3498DB" if e >= 0 else "#E74C3C" for e in errors]
    ax2.bar(common, errors, color=bar_c, alpha=0.7, width=50)
    ax2.axhline(0, color="black", linewidth=0.8)
    rmse = metrics[best]["RMSE"]
    ax2.axhline(rmse, color="#AAA", linewidth=0.7, linestyle="--")
    ax2.axhline(-rmse, color="#AAA", linewidth=0.7, linestyle="--")
    ax2.set_ylabel(f"Error (pp) [{best}]"); ax2.set_xlabel("Quarter")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fp = f"{out_dir}/{country}_Actuals_vs_Predicted.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    logger.info("Actuals vs Predicted saved: %s", fp)


# =========================================================================
# VISUALIZATION: INVESTMENT STORY CHART (WITH FORWARD NOWCAST)
# =========================================================================
def plot_investment_story(country, gdp, best_fc, nowcast, conf_bands, best_model, metrics, out_dir):
    _apply_style()
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(gdp.index, gdp.values, color=COLORS["gdp_actual"], linewidth=2.8,
            label="Official GDP", marker="o", markersize=3, zorder=10)
    ax.plot(best_fc.index, best_fc.values, color=MODEL_COLORS.get(best_model, "#27AE60"),
            linewidth=2, linestyle="--", alpha=0.6, label=f"Model ({best_model})")

    combined_idx = pd.Index([gdp.index[-1]]).append(nowcast.index)
    combined_val = np.concatenate([[gdp.values[-1]], nowcast.values])
    ax.plot(combined_idx, combined_val, color="#C0392B", linewidth=2.8, marker="D", markersize=6,
            zorder=12, label="Forward Nowcast")

    # Confidence bands: use per-horizon bands if available
    if isinstance(conf_bands, pd.Series):
        bands = conf_bands.values
    else:
        bands = np.full(len(nowcast), conf_bands)

    for mult, alpha in [(0.5, 0.19), (1.0, 0.13), (1.5, 0.07)]:
        ax.fill_between(nowcast.index, nowcast.values - bands * mult,
                        nowcast.values + bands * mult, color="#E74C3C", alpha=alpha)

    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-03-31"), color="#FFE0E0", alpha=0.2, zorder=0)
    ax.axvline(gdp.index[-1], color="#7F8C8D", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.axhline(0, color="#AAAAAA", linewidth=0.8)

    rmse = metrics[best_model]["RMSE"]
    dir_acc = metrics[best_model]["DirAcc_%"]
    corr_val = metrics[best_model]["Corr"]
    stats_text = (f"Model: {best_model} | OOS RMSE: {rmse}pp | Dir. Accuracy: {dir_acc}%\n"
                  f"Correlation: {corr_val:.3f} | 2025 Nowcast Avg: {nowcast.mean():.1f}%")
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CCCCCC", alpha=0.95))

    ax.set_title(f"{country}: Economic Cycle and Forward Nowcast", fontsize=15, fontweight="bold")
    ax.set_ylabel("Real GDP Growth, YoY (%)"); ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fp = f"{out_dir}/{country}_Investment_Story.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    logger.info("Investment Story saved: %s", fp)


# =========================================================================
# VISUALIZATION: MODEL COEFFICIENTS / FEATURE IMPORTANCE
# =========================================================================
def plot_coefficients(country, all_coefficients, out_dir):
    """Plots model coefficients and feature importance for all models."""
    _apply_style()

    plottable = {}
    for model_name, coeffs in all_coefficients.items():
        if not coeffs or isinstance(coeffs, str):
            continue
        if "loadings" in coeffs:
            raw = coeffs["loadings"]
            label = "Factor Loading"
        elif "weights" in coeffs:
            raw = coeffs["weights"]
            label = "Ensemble Weight"
        elif "intercept" in coeffs:
            raw = {k: v for k, v in coeffs.items()
                   if k not in ("intercept", "_alpha", "_l1_ratio")}
            label = "Ridge Coefficient"
        else:
            raw = coeffs
            label = "Feature Importance"
        clean = {}
        for k, v in raw.items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                continue
        if clean:
            plottable[model_name] = (clean, label)

    if not plottable:
        logger.warning("No plottable coefficients for %s.", country)
        return

    n = len(plottable)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for idx, (model_name, (data, bar_label)) in enumerate(plottable.items()):
        ax = axes[idx]
        sorted_items = sorted(data.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        names = [item[0][:20] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        colors = [MODEL_COLORS.get(model_name, "#2980B9") if v >= 0 else "#E74C3C" for v in values]
        ax.barh(range(len(names)), values, color=colors, alpha=0.8, edgecolor="white")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel(bar_label, fontsize=9)
        ax.set_title(f"{model_name}", fontweight="bold", fontsize=12)
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.invert_yaxis()

    fig.suptitle(f"{country}: Model Coefficients and Feature Importance",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fp = f"{out_dir}/{country}_Coefficients.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    logger.info("Coefficients chart saved: %s", fp)


# =========================================================================
# VISUALIZATION: STATISTICAL VALIDATION DASHBOARD
# =========================================================================
def plot_statistical_validation(country, actual, forecast, model_name, test_results, out_dir):
    _apply_style()
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.30)

    common = actual.index.intersection(forecast.index)
    y, yhat = actual.loc[common].values, forecast.loc[common].values

    mz = test_results["mincer_zarnowitz"]
    perm = test_results["permutation"]
    naive = test_results["naive_benchmark"]
    gap = test_results["overfitting_gap"]
    stab = test_results["recursive_stability"]
    pt = test_results["pesaran_timmermann"]

    # Panel 1: Mincer-Zarnowitz
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(yhat, y, color="#2980B9", alpha=0.6, s=50, edgecolors="white")
    xline = np.linspace(yhat.min(), yhat.max(), 100)
    ax1.plot(xline, mz["alpha"] + mz["beta"]*xline, color="#C0392B", linewidth=2,
             label=f"MZ: a={mz['alpha']:.2f}, b={mz['beta']:.2f}")
    ax1.plot(xline, xline, color="#CCC", linestyle="--", linewidth=1, label="Perfect (a=0, b=1)")
    ax1.set_xlabel("Forecast"); ax1.set_ylabel("Actual")
    ax1.set_title(f"Mincer-Zarnowitz [{mz['verdict']}]"); ax1.legend(fontsize=8)
    c = "#27AE60" if mz["verdict"] == "PASS" else "#E74C3C"
    ax1.text(0.95, 0.05, f"p(joint)={mz['p_joint']:.3f}", transform=ax1.transAxes, ha="right", fontsize=10, color=c, fontweight="bold")

    # Panel 2: Permutation Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    shuffled = perm["_shuffled"]
    ax2.hist(shuffled, bins=40, color="#BDC3C7", edgecolor="white", alpha=0.8, label="Shuffled")
    ax2.axvline(perm["actual_corr"], color="#C0392B", linewidth=2.5, label=f"Actual: {perm['actual_corr']:.3f}")
    ax2.set_xlabel("Correlation"); ax2.set_ylabel("Frequency")
    ax2.set_title(f"Permutation Test [{perm['verdict']}]"); ax2.legend(fontsize=8)
    c = "#27AE60" if perm["verdict"] == "PASS" else "#E74C3C"
    ax2.text(0.95, 0.95, f"p={perm['p_value']:.4f}", transform=ax2.transAxes, ha="right", va="top", fontsize=10, color=c, fontweight="bold")

    # Panel 3: Naive Benchmarks
    ax3 = fig.add_subplot(gs[1, 0])
    labels = ["Our Model", "Hist. Mean", "Random Walk"]
    vals = [naive["Model_RMSE"], naive["HistMean_RMSE"], naive["RW_RMSE"]]
    colors = ["#2980B9", "#27AE60" if naive["Model_RMSE"] < naive["HistMean_RMSE"] else "#E74C3C",
              "#27AE60" if naive["Model_RMSE"] < naive["RW_RMSE"] else "#E74C3C"]
    bars = ax3.bar(labels, vals, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)
    ax3.set_ylabel("RMSE (pp)"); ax3.set_title(f"Naive Benchmark [{naive['verdict']}]")
    for bar, val in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")

    # Panel 4: Temporal Stability (OOS first-half vs second-half)
    ax4 = fig.add_subplot(gs[1, 1])
    bars = ax4.bar(["OOS First Half", "OOS Second Half"], [gap["RMSE_IS"], gap["RMSE_OOS"]],
                   color=["#3498DB", "#E67E22"], alpha=0.8, edgecolor="white", linewidth=1.5)
    ax4.set_ylabel("RMSE (pp)"); ax4.set_title(f"Temporal Stability [{gap['verdict']}]")
    c = "#27AE60" if gap["verdict"] == "PASS" else "#E74C3C"
    ax4.text(0.5, 0.9, f"Ratio: {gap['Ratio']:.2f}x", transform=ax4.transAxes, ha="center",
             fontsize=12, fontweight="bold", color=c)

    # Panel 5: Recursive RMSE
    ax5 = fig.add_subplot(gs[2, 0])
    rolling = stab["_rolling"]
    dates = stab["_dates"]
    ax5.plot(dates, rolling, color="#2980B9", linewidth=2, marker="o", markersize=3)
    ax5.axhline(np.mean(rolling), color="#27AE60", linewidth=1.5, linestyle="--", label=f"Mean: {np.mean(rolling):.2f}")
    ax5.fill_between(dates, np.mean(rolling)-np.std(rolling), np.mean(rolling)+np.std(rolling),
                     color="#2980B9", alpha=0.15, label=f"CV: {stab['CV']:.2f}")
    ax5.set_ylabel("Rolling RMSE (pp)"); ax5.set_title(f"RMSE Stability [{stab['verdict']}]")
    ax5.legend(fontsize=8); ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 6: Scorecard
    ax6 = fig.add_subplot(gs[2, 1]); ax6.axis("off")
    tests = [("Mincer-Zarnowitz (Efficiency)", mz["verdict"]),
             ("Pesaran-Timmermann (Direction)", pt["verdict"]),
             ("Naive Benchmark (Theil U)", naive["verdict"]),
             ("Temporal Stability (OOS)", gap["verdict"]),
             ("Permutation Test (Signal)", perm["verdict"]),
             ("RMSE Stability (Temporal)", stab["verdict"])]

    ax6.text(0.5, 0.97, f"{country} -- Statistical Validation Scorecard",
             ha="center", va="top", fontsize=14, fontweight="bold", transform=ax6.transAxes)
    ax6.text(0.5, 0.90, f"Model: {model_name}", ha="center", va="top", fontsize=11, color="#7F8C8D", transform=ax6.transAxes)

    passes = sum(1 for _, v in tests if v == "PASS")
    for i, (name, verdict) in enumerate(tests):
        y_pos = 0.78 - i * 0.12
        c = "#27AE60" if verdict == "PASS" else "#E67E22" if verdict in ("CAUTION", "MARGINAL") else "#E74C3C"
        ax6.text(0.08, y_pos, verdict, transform=ax6.transAxes, fontsize=12, fontweight="bold", color=c, fontfamily="monospace")
        ax6.text(0.30, y_pos, name, transform=ax6.transAxes, fontsize=11, color="#2C3E50")

    overall = "VALIDATED" if passes >= 5 else "PARTIALLY VALIDATED" if passes >= 3 else "NOT VALIDATED"
    oc = "#27AE60" if passes >= 5 else "#E67E22" if passes >= 3 else "#E74C3C"
    ax6.text(0.5, 0.05, f"Overall: {overall} ({passes}/{len(tests)} tests passed)",
             ha="center", transform=ax6.transAxes, fontsize=13, fontweight="bold", color=oc,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=oc, linewidth=2))

    fig.suptitle(f"{country}: Statistical Validation Dashboard", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fp = f"{out_dir}/{country}_Statistical_Validation.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    logger.info("Statistical Validation saved: %s", fp)

"""Rebuild cleaned data, merged outputs, SQLite tables, and figures.

This script is the source of truth for the corrected notebook workflow. It fixes
issues found during review:

1. Fetch World Bank data directly for USA instead of country/all.
2. Validate every annual merge with validate='one_to_one'.
3. Treat all known Shiller 0.0 unavailable-data placeholders as missing.
4. Rename ambiguous engineered columns so they describe the actual calculation.
5. Label S&P annual metrics with the number of aligned months used.
6. Store current raw, cleaned, and merged tables in SQLite.
7. Keep regenerated figures aligned with the executed notebook.
"""


import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "images"
START_YEAR = 2000
END_YEAR = 2023
EXPECTED_YEARS = set(range(START_YEAR, END_YEAR + 1))
SP500_MIN_MONTHS_BY_YEAR = {END_YEAR: 9}
GOLD_MIN_DAYS_BY_YEAR = {START_YEAR: 80}
DEFAULT_SP500_MIN_MONTHS = 12
DEFAULT_GOLD_MIN_DAYS = 200
REGPLOT_SEED = 42

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)


def assert_expected_years(years: pd.Series, label: str) -> None:
    actual_years = set(years.dropna().astype(int))
    missing_years = EXPECTED_YEARS - actual_years
    extra_years = actual_years - EXPECTED_YEARS
    assert not missing_years, f"{label} is missing years: {sorted(missing_years)}"
    assert not extra_years, f"{label} has unexpected years: {sorted(extra_years)}"


def assert_yearly_coverage(
    dates: pd.Series,
    label: str,
    *,
    default_min_count: int,
    min_count_by_year: dict[int, int],
) -> None:
    counts = dates.dt.year.value_counts().sort_index()
    for year in range(START_YEAR, END_YEAR + 1):
        min_count = min_count_by_year.get(year, default_min_count)
        actual_count = int(counts.get(year, 0))
        assert actual_count >= min_count, (
            f"{label} has thin coverage for {year}: "
            f"{actual_count} observations, expected at least {min_count}"
        )


def fetch_world_bank_indicator(indicator: str, label: str) -> list:
    """Fetch one World Bank indicator for the United States."""
    url = (
        f"https://api.worldbank.org/v2/country/USA/indicator/{indicator}"
        f"?date={START_YEAR}:{END_YEAR}&format=json&per_page=2000"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if len(payload) < 2 or payload[1] is None:
        raise ValueError(f"No data returned for {label} ({indicator})")
    return payload


def load_cached_world_bank_indicator(path: Path, label: str) -> list:
    """Load a previously saved World Bank API response when the API is unavailable."""
    payload = json.loads(path.read_text())
    if len(payload) < 2 or payload[1] is None:
        raise ValueError(f"Cached World Bank data is invalid for {label}: {path}")
    return payload


def normalize_world_bank_indicator(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Normalize nested World Bank country/indicator records."""
    out = pd.DataFrame(
        {
            "country_name": df["country"].apply(
                lambda value: value.get("value") if isinstance(value, dict) else np.nan
            ),
            "countryiso3code": df["countryiso3code"].replace("", np.nan),
            "date": pd.to_numeric(df["date"], errors="coerce").astype("Int64"),
            value_col: pd.to_numeric(df["value"], errors="coerce"),
        }
    )
    out["countryiso3code"] = out["countryiso3code"].astype("string").str.strip()
    return out


def prepare_sqlite_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert nested raw API objects to JSON strings before SQLite storage."""
    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda value: isinstance(value, (dict, list))).any():
            out[col] = out[col].map(
                lambda value: json.dumps(value) if isinstance(value, (dict, list)) else value
            )
    return out


def first_matching_column(
    columns: pd.Index,
    *,
    exact_names: set[str],
    prefixes: tuple[str, ...] = (),
) -> object | None:
    """Return the first column whose normalized name matches common variants."""
    for col in columns:
        col_name = str(col).strip()
        lowered = col_name.lower()
        if lowered in exact_names or any(lowered.startswith(prefix) for prefix in prefixes):
            return col
    return None


def load_sp500() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv"
    raw_path = DATA_DIR / "sp500_shiller_raw.csv"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        raw_path.write_bytes(response.content)
    except requests.RequestException as exc:
        if raw_path.exists():
            print(f"S&P 500 download failed ({exc}); using cached raw file at {raw_path}.")
        else:
            raise

    df = pd.read_csv(raw_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = [
        "SP500",
        "Dividend",
        "Earnings",
        "Consumer Price Index",
        "Long Interest Rate",
        "Real Price",
        "Real Dividend",
        "Real Earnings",
        "PE10",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    placeholder_zero_cols = [
        "Dividend",
        "Earnings",
        "Consumer Price Index",
        "Long Interest Rate",
        "Real Price",
        "Real Dividend",
        "Real Earnings",
        "PE10",
    ]
    df[placeholder_zero_cols] = df[placeholder_zero_cols].replace(0, np.nan)
    df = df[df["Date"].dt.year.between(START_YEAR, END_YEAR)].copy()
    required = [
        "Date",
        "SP500",
        "Consumer Price Index",
        "Long Interest Rate",
        "PE10",
    ]
    df = df.dropna(subset=required)

    assert df["Date"].is_unique, "S&P 500 monthly dates should be unique"
    assert_expected_years(df["Date"].dt.year, "S&P 500 clean data")
    assert_yearly_coverage(
        df["Date"],
        "S&P 500 clean data",
        default_min_count=DEFAULT_SP500_MIN_MONTHS,
        min_count_by_year=SP500_MIN_MONTHS_BY_YEAR,
    )

    df.to_csv(DATA_DIR / "sp500_shiller_clean.csv", index=False)
    return df


def load_world_bank() -> pd.DataFrame:
    gdp_path = DATA_DIR / "world_bank_gdp_raw.json"
    inflation_path = DATA_DIR / "world_bank_inflation_raw.json"
    try:
        gdp_data = fetch_world_bank_indicator("NY.GDP.MKTP.KD.ZG", "GDP growth")
        gdp_path.write_text(json.dumps(gdp_data, indent=2))
    except (requests.RequestException, ValueError) as exc:
        if gdp_path.exists():
            print(f"World Bank GDP download failed ({exc}); using cached raw file at {gdp_path}.")
            gdp_data = load_cached_world_bank_indicator(gdp_path, "GDP growth")
        else:
            raise

    try:
        inflation_data = fetch_world_bank_indicator("FP.CPI.TOTL.ZG", "CPI inflation")
        inflation_path.write_text(json.dumps(inflation_data, indent=2))
    except (requests.RequestException, ValueError) as exc:
        if inflation_path.exists():
            print(
                "World Bank inflation download failed "
                f"({exc}); using cached raw file at {inflation_path}."
            )
            inflation_data = load_cached_world_bank_indicator(inflation_path, "CPI inflation")
        else:
            raise

    df_gdp = normalize_world_bank_indicator(pd.DataFrame(gdp_data[1]), "gdp_growth")
    df_inflation = normalize_world_bank_indicator(pd.DataFrame(inflation_data[1]), "inflation_rate")

    df = pd.merge(
        df_gdp,
        df_inflation,
        on=["country_name", "countryiso3code", "date"],
        how="inner",
        validate="one_to_one",
    )
    df = df[df["countryiso3code"] == "USA"].sort_values("date").reset_index(drop=True)

    assert df["date"].is_unique, "World Bank clean data should have one row per year"
    assert_expected_years(df["date"], "World Bank clean data")
    assert df[["gdp_growth", "inflation_rate"]].notna().all().all()

    df.to_csv(DATA_DIR / "world_bank_clean.csv", index=False)
    return df


def load_gold_proxy() -> pd.DataFrame:
    raw_path = DATA_DIR / "gold_prices_raw.csv"
    raw_is_cached_csv = False
    try:
        df = yf.download(
            "GC=F",
            start=f"{START_YEAR}-01-01",
            end=f"{END_YEAR + 1}-01-01",
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            raise ValueError("No gold data returned from yfinance for GC=F")
    except Exception as exc:
        if raw_path.exists():
            print(f"Gold proxy download failed ({exc}); using cached raw file at {raw_path}.")
            df = pd.read_csv(raw_path)
            raw_is_cached_csv = True
        else:
            raise

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(part) for part in col if part).strip("_")
            for col in df.columns.to_flat_index()
        ]

    date_col = first_matching_column(
        df.columns,
        exact_names={"date", "datetime", "index"},
        prefixes=("date_", "datetime_"),
    )
    if date_col is None and not raw_is_cached_csv:
        df = df.reset_index()
        date_col = first_matching_column(
            df.columns,
            exact_names={"date", "datetime", "index"},
            prefixes=("date_", "datetime_"),
        )
    if date_col is not None and str(date_col) != "Date":
        df = df.rename(columns={date_col: "Date"})
        date_col = "Date"

    close_col = first_matching_column(df.columns, exact_names={"close"}, prefixes=("close_",))
    volume_col = first_matching_column(df.columns, exact_names={"volume"}, prefixes=("volume_",))

    missing = [
        name
        for name, col in {"date": date_col, "close": close_col, "volume": volume_col}.items()
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Could not find required gold data column(s): {', '.join(missing)}. "
            f"Columns returned by yfinance: {[str(col) for col in df.columns]}"
        )

    df.to_csv(raw_path, index=False)

    df = df.rename(columns={date_col: "Date", close_col: "Close", volume_col: "Volume"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df[df["Date"].dt.year.between(START_YEAR, END_YEAR)].copy()
    assert df["Date"].is_unique, "Gold clean data should have unique trading dates"
    assert_expected_years(df["Date"].dt.year, "Gold clean data")
    assert_yearly_coverage(
        df["Date"],
        "Gold clean data",
        default_min_count=DEFAULT_GOLD_MIN_DAYS,
        min_count_by_year=GOLD_MIN_DAYS_BY_YEAR,
    )

    df.to_csv(DATA_DIR / "gold_prices_clean.csv", index=False)
    return df


def build_merged_dataset(
    sp500_clean: pd.DataFrame,
    world_bank_clean: pd.DataFrame,
    gold_clean: pd.DataFrame,
) -> pd.DataFrame:
    sp500_annual = (
        sp500_clean.assign(Year=sp500_clean["Date"].dt.year)
        .groupby("Year", as_index=False)[
            [
                "SP500",
                "Dividend",
                "Earnings",
                "Consumer Price Index",
                "Long Interest Rate",
                "PE10",
            ]
        ]
        .mean()
    )
    sp500_months_used = (
        sp500_clean.assign(Year=sp500_clean["Date"].dt.year)
        .groupby("Year", as_index=False)
        .agg(sp500_months_used=("SP500", "count"))
    )
    sp500_annual = pd.merge(
        sp500_annual,
        sp500_months_used,
        on="Year",
        how="inner",
        validate="one_to_one",
    )
    sp500_annual["sp500_coverage_label"] = np.where(
        sp500_annual["sp500_months_used"] == DEFAULT_SP500_MIN_MONTHS,
        "full_year",
        "aligned_9_month_partial_year",
    )

    gold_annual = (
        gold_clean.assign(Year=gold_clean["Date"].dt.year)
        .groupby("Year", as_index=False)
        .agg(gold_close=("Close", "mean"), gold_volume=("Volume", "mean"))
    )

    wb_annual = world_bank_clean.rename(columns={"date": "Year"})[
        ["Year", "gdp_growth", "inflation_rate"]
    ].copy()
    wb_annual["Year"] = wb_annual["Year"].astype(int)

    for name, df in {
        "S&P annual": sp500_annual,
        "World Bank annual": wb_annual,
        "Gold annual": gold_annual,
    }.items():
        assert df["Year"].is_unique, f"{name} has duplicate years"
        assert_expected_years(df["Year"], name)

    merged = pd.merge(
        sp500_annual,
        wb_annual,
        on="Year",
        how="inner",
        validate="one_to_one",
    )
    merged = pd.merge(
        merged,
        gold_annual,
        on="Year",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.sort_values("Year").reset_index(drop=True)

    assert_expected_years(merged["Year"], "Merged data")

    merged["cape_yield"] = 100 / merged["PE10"]
    merged["cape_yield_minus_10y_yield"] = merged["cape_yield"] - merged["Long Interest Rate"]
    merged["dividend_yield"] = (merged["Dividend"] / merged["SP500"]) * 100
    merged["sp500_annual_avg_yoy_change"] = merged["SP500"].pct_change() * 100
    merged["gold_annual_avg_yoy_change"] = merged["gold_close"].pct_change() * 100
    # Set gold YoY change to NaN for 2000 and 2001 due to partial data in 2000
    merged.loc[merged["Year"] <= 2001, "gold_annual_avg_yoy_change"] = np.nan
    merged["gdp_growth_lag1"] = merged["gdp_growth"].shift(1)
    merged["gdp_growth_lead1"] = merged["gdp_growth"].shift(-1)

    assert merged["Year"].is_unique
    assert (
        merged.loc[merged["Year"] == END_YEAR, "sp500_coverage_label"].item()
        == "aligned_9_month_partial_year"
    )
    assert merged.loc[merged["Year"] == END_YEAR, "sp500_months_used"].item() == 9
    assert merged[
        ["SP500", "Dividend", "Earnings", "PE10", "gdp_growth", "inflation_rate", "gold_close"]
    ].notna().all().all()

    merged.to_csv(DATA_DIR / "macro_stock_merged.csv", index=False)
    return merged


def store_sqlite(
    sp500_clean: pd.DataFrame,
    world_bank_clean: pd.DataFrame,
    gold_clean: pd.DataFrame,
    merged: pd.DataFrame,
) -> None:
    raw_sp500 = pd.read_csv(DATA_DIR / "sp500_shiller_raw.csv")
    raw_gold = pd.read_csv(DATA_DIR / "gold_prices_raw.csv")
    raw_gdp = pd.DataFrame(
        load_cached_world_bank_indicator(DATA_DIR / "world_bank_gdp_raw.json", "GDP growth")[1]
    )
    raw_inflation = pd.DataFrame(
        load_cached_world_bank_indicator(
            DATA_DIR / "world_bank_inflation_raw.json", "CPI inflation"
        )[1]
    )

    with sqlite3.connect(DATA_DIR / "macro_stock_data.db") as conn:
        raw_sp500.to_sql("sp500_shiller_raw", conn, if_exists="replace", index=False)
        prepare_sqlite_frame(raw_gdp).to_sql(
            "world_bank_gdp_raw", conn, if_exists="replace", index=False
        )
        prepare_sqlite_frame(raw_inflation).to_sql(
            "world_bank_inflation_raw", conn, if_exists="replace", index=False
        )
        raw_gold.to_sql("gold_prices_raw", conn, if_exists="replace", index=False)
        sp500_clean.to_sql("sp500_shiller_clean", conn, if_exists="replace", index=False)
        world_bank_clean.to_sql("world_bank_clean", conn, if_exists="replace", index=False)
        gold_clean.to_sql("gold_prices_clean", conn, if_exists="replace", index=False)
        merged.to_sql("macro_stock_merged", conn, if_exists="replace", index=False)


def save_figures(merged: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.regplot(data=merged, x="gdp_growth", y="PE10", seed=REGPLOT_SEED)
    plt.title("Shiller PE10 vs. Annual U.S. GDP Growth (2000-2023)")
    plt.xlabel("GDP Growth Rate (%)")
    plt.ylabel("Shiller PE10")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "visual1_pe10_vs_gdp.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=merged,
        x="inflation_rate",
        y="Long Interest Rate",
        label="10-year Treasury yield",
        seed=REGPLOT_SEED,
    )
    sns.regplot(
        data=merged,
        x="inflation_rate",
        y="cape_yield",
        label="CAPE yield proxy (100 / PE10)",
        seed=REGPLOT_SEED,
    )
    plt.title("U.S. CPI Inflation vs. Long-Term Interest Rates & CAPE Yield (2000-2023)")
    plt.xlabel("Annual CPI Inflation Rate (%)")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "visual2_inflation_vs_yields.png", dpi=150)
    plt.close()

    corr_cols = [
        "gdp_growth",
        "inflation_rate",
        "SP500",
        "PE10",
        "Long Interest Rate",
        "cape_yield",
        "cape_yield_minus_10y_yield",
        "dividend_yield",
        "gold_close",
        "sp500_annual_avg_yoy_change",
        "gold_annual_avg_yoy_change",
    ]
    plt.figure(figsize=(10, 7))
    sns.heatmap(merged[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Macro and Market Indicators (2000-2023)")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "visual3_correlation_heatmap.png", dpi=150)
    plt.close()

    regimes = merged.copy()
    regimes["inflation_regime"] = pd.cut(
        regimes["inflation_rate"],
        bins=[-np.inf, 2, 4, np.inf],
        labels=["Low (<2%)", "Moderate (2-4%)", "High (>4%)"],
    )
    regime_summary = regimes.groupby("inflation_regime", observed=True).agg(
        years=("Year", "count"),
        sp500_n=("sp500_annual_avg_yoy_change", "count"),
        gold_n=("gold_annual_avg_yoy_change", "count"),
        sp500_avg_yoy=("sp500_annual_avg_yoy_change", "mean"),
        gold_avg_yoy=("gold_annual_avg_yoy_change", "mean"),
    )
    plot_data = (
        regime_summary.reset_index()
        .melt(
            id_vars="inflation_regime",
            value_vars=["sp500_avg_yoy", "gold_avg_yoy"],
            var_name="Series",
            value_name="Average YoY change",
        )
        .replace(
            {
                "sp500_avg_yoy": "S&P 500 annual-mean YoY",
                "gold_avg_yoy": "Gold proxy annual-mean YoY",
            }
        )
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_data, x="inflation_regime", y="Average YoY change", hue="Series")
    plt.title("Average Annual-Mean Price Changes by Inflation Regime")
    plt.ylabel("Average Change (%)")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "visual4_inflation_regimes.png", dpi=150)
    plt.close()

    # Visual 5: S&P 500 YoY Change (Year t) vs. Future GDP Growth (Year t+1)
    lag_data = merged.dropna(subset=["sp500_annual_avg_yoy_change", "gdp_growth_lead1"]).copy()
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=lag_data,
        x="sp500_annual_avg_yoy_change",
        y="gdp_growth_lead1",
        seed=REGPLOT_SEED,
    )
    plt.title("S&P 500 Annual-Average YoY Change (Year t) vs. Future GDP Growth (Year t+1)")
    plt.xlabel("S&P 500 YoY Change in Year t (%)")
    plt.ylabel("GDP Growth Rate in Year t+1 (%)")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "visual5_lagged_lead_gdp.png", dpi=150)
    plt.close()

    # Visual 6: US Inflation Rate vs. S&P 500 Dividend Yield and Long-Term Interest Rates
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Inflation on primary y-axis
    color = "crimson"
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("US Annual Inflation Rate (%)", color=color, fontsize=12)
    line1 = ax1.plot(merged["Year"], merged["inflation_rate"], color=color, linewidth=2.5, label="US Inflation Rate", marker="s")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    color_yield = "navy"
    color_rate = "darkorange"
    ax2.set_ylabel("Yield / Interest Rate (%)", color="black", fontsize=12)
    line2 = ax2.plot(merged["Year"], merged["dividend_yield"], color=color_yield, linewidth=2, linestyle="--", label="S&P 500 Dividend Yield", marker="o")
    line3 = ax2.plot(merged["Year"], merged["Long Interest Rate"], color=color_rate, linewidth=2, linestyle="-.", label="US Long Interest Rate (10Y)", marker="^")
    ax2.tick_params(axis="y", labelcolor="black")

    # Combine legends from both axes
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=10)

    plt.title("US Inflation Rate vs. S&P 500 Dividend Yield & 10Y Interest Rate (2000-2023)", fontsize=14)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "visual6_inflation_vs_yields_trends.png", dpi=150)
    plt.close()


def run_regression_model(df_merged: pd.DataFrame) -> None:
    """Run an OLS multiple linear regression model with advanced diagnostics from scratch."""
    import math
    from scipy.stats import t, chi2

    df_reg = df_merged.dropna(subset=['PE10', 'gdp_growth', 'inflation_rate', 'Long Interest Rate']).copy()
    if df_reg.empty:
        print("No rows available for regression model.")
        return

    X = df_reg[['gdp_growth', 'inflation_rate', 'Long Interest Rate']].values
    y = df_reg['PE10'].values

    # Add a column of ones to serve as the intercept term
    X_with_const = np.hstack([np.ones((X.shape[0], 1)), X])

    # Solve normal equation: beta = (X^T X)^-1 X^T y
    beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)

    # Calculate statistical metrics (R2 and Adjusted R2)
    n = X.shape[0]
    k = X.shape[1]
    y_mean = np.mean(y)
    tss = np.sum((y - y_mean) ** 2)
    e = y - X_with_const @ beta
    rss = np.sum(e ** 2)
    r2 = 1.0 - (rss / tss)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
    rse = np.sqrt(rss / (n - k - 1))

    # OLS Standard Errors
    XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
    se_ols = np.sqrt((rse ** 2) * np.diagonal(XtX_inv))

    # HAC (Newey-West) Standard Errors (lag = 2)
    S = np.zeros((k + 1, k + 1))
    for t_idx in range(n):
        xt = X_with_const[t_idx, :].reshape(-1, 1)
        S += (e[t_idx] ** 2) * (xt @ xt.T)
    for j in range(1, 3):
        weight = 1.0 - (j / 3.0)
        for t_idx in range(j, n):
            xt = X_with_const[t_idx, :].reshape(-1, 1)
            xt_lag = X_with_const[t_idx - j, :].reshape(-1, 1)
            Gamma = e[t_idx] * e[t_idx - j] * (xt @ xt_lag.T)
            S += weight * (Gamma + Gamma.T)
    cov_hac = XtX_inv @ S @ XtX_inv
    se_hac = np.sqrt(np.diagonal(cov_hac))

    # t-statistics and two-tailed p-values (using HAC standard errors and t-distribution)
    t_stats_hac = beta / se_hac
    df_resid = n - k - 1
    p_vals_hac = [2.0 * t.sf(abs(t_stat), df=df_resid) for t_stat in t_stats_hac]
    
    # 95% Confidence Intervals using t-distribution critical value
    t_critical = t.ppf(0.975, df=df_resid)
    ci_lower = beta - t_critical * se_hac
    ci_upper = beta + t_critical * se_hac

    # Variance Inflation Factors (VIF)
    vifs = []
    for i in range(k):
        y_v = X[:, i]
        X_v = np.delete(X, i, axis=1)
        X_v_c = np.hstack([np.ones((n, 1)), X_v])
        b_v, _, _, _ = np.linalg.lstsq(X_v_c, y_v, rcond=None)
        rss_v = np.sum((y_v - X_v_c @ b_v) ** 2)
        tss_v = np.sum((y_v - np.mean(y_v)) ** 2)
        r2_v = 1.0 - (rss_v / tss_v)
        vifs.append(1.0 / (1.0 - r2_v))

    # Durbin-Watson statistic for residual autocorrelation
    dw = np.sum((e[1:] - e[:-1]) ** 2) / rss

    # Jarque-Bera statistic for normality of residuals
    e_std = np.std(e)
    skew = np.sum(((e - np.mean(e)) / e_std) ** 3) / n
    kurt = np.sum(((e - np.mean(e)) / e_std) ** 4) / n
    jb = (n / 6.0) * (skew ** 2 + 0.25 * ((kurt - 3.0) ** 2))
    jb_p_value = chi2.sf(jb, df=2)

    print("====================================================================================================")
    print("                              MULTIPLE LINEAR REGRESSION ANALYSIS (OLS)")
    print("====================================================================================================")
    print(f"Dependent Variable:             S&P 500 Shiller PE10")
    print(f"Observations (n):               {n:<8} | Independent Variables (k): {k}")
    print(f"Residual Sum of Squares (RSS):  {rss:<8.4f} | Total Sum of Squares (TSS):    {tss:.4f}")
    print(f"Residual Standard Error (RSE):  {rse:<8.4f} | R-squared (R2):                {r2:.4f}")
    print(f"Durbin-Watson (DW) Statistic:   {dw:<8.4f} | Adjusted R-squared:            {adj_r2:.4f}")
    print(f"Jarque-Bera (JB) Statistic:     {jb:<8.4f} | JB p-value:                    {jb_p_value:.4f}")
    print("----------------------------------------------------------------------------------------------------")
    print(f"{'Variable':<25}{'Coef':<10}{'Std Err (OLS)':<15}{'Std Err (HAC)':<15}{'t-stat (HAC)':<15}{'P>|t| (HAC)':<15}{'[95% Conf. Interval]':<20}")
    print("----------------------------------------------------------------------------------------------------")
    
    var_names = [
        "Intercept (Constant)",
        "GDP Growth (annual %)",
        "Inflation Rate (annual %)",
        "Long Interest Rate (%)"
    ]
    
    for i in range(len(var_names)):
        print(f"{var_names[i]:<25}{beta[i]:<10.4f}{se_ols[i]:<15.4f}{se_hac[i]:<15.4f}{t_stats_hac[i]:<15.4f}{p_vals_hac[i]:<15.4f}[{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]")
        
    print("----------------------------------------------------------------------------------------------------")
    print("Variance Inflation Factors (VIF) for Collinearity Check:")
    print(f"  - GDP Growth VIF:             {vifs[0]:.4f} (VIF < 5 indicates low multicollinearity)")
    print(f"  - Inflation Rate VIF:         {vifs[1]:.4f}")
    print(f"  - Long Interest Rate VIF:     {vifs[2]:.4f}")
    print("====================================================================================================")


def main() -> None:
    sp500_clean = load_sp500()
    world_bank_clean = load_world_bank()
    gold_clean = load_gold_proxy()
    merged = build_merged_dataset(sp500_clean, world_bank_clean, gold_clean)
    store_sqlite(sp500_clean, world_bank_clean, gold_clean, merged)
    save_figures(merged)
    run_regression_model(merged)
    print("Rebuilt cleaned CSVs, merged output, SQLite database, and figures.")


if __name__ == "__main__":
    main()

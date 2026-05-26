"""Rebuild cleaned data, merged outputs, SQLite tables, and figures.

This script is the source of truth for the corrected notebook workflow. It fixes
three issues found during review:

1. Fetch World Bank data directly for USA instead of country/all.
2. Validate every annual merge with validate='one_to_one'.
3. Rename ambiguous engineered columns so they describe the actual calculation.
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

DATA_DIR = Path("data")
IMAGE_DIR = Path("images")
START_YEAR = 2000
END_YEAR = 2023

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)


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
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    raw_path.write_bytes(response.content)

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

    df["PE10"] = df["PE10"].replace(0, np.nan)
    df = df[df["Date"].dt.year.between(START_YEAR, END_YEAR)].copy()
    required = [
        "Date",
        "SP500",
        "Dividend",
        "Earnings",
        "Consumer Price Index",
        "Long Interest Rate",
        "PE10",
    ]
    df = df.dropna(subset=required)

    assert df["Date"].is_unique, "S&P 500 monthly dates should be unique"
    assert df["Date"].dt.year.min() == START_YEAR
    assert df["Date"].dt.year.max() == END_YEAR

    df.to_csv(DATA_DIR / "sp500_shiller_clean.csv", index=False)
    return df


def load_world_bank() -> pd.DataFrame:
    gdp_data = fetch_world_bank_indicator("NY.GDP.MKTP.KD.ZG", "GDP growth")
    inflation_data = fetch_world_bank_indicator("FP.CPI.TOTL.ZG", "CPI inflation")

    (DATA_DIR / "world_bank_gdp_raw.json").write_text(json.dumps(gdp_data, indent=2))
    (DATA_DIR / "world_bank_inflation_raw.json").write_text(json.dumps(inflation_data, indent=2))

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
    assert df["date"].min() == START_YEAR
    assert df["date"].max() == END_YEAR
    assert df[["gdp_growth", "inflation_rate"]].notna().all().all()

    df.to_csv(DATA_DIR / "world_bank_clean.csv", index=False)
    return df


def load_gold_proxy() -> pd.DataFrame:
    df = yf.download(
        "GC=F",
        start=f"{START_YEAR}-01-01",
        end=f"{END_YEAR + 1}-01-01",
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise ValueError("No gold data returned from yfinance for GC=F")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(part) for part in col if part).strip("_")
            for col in df.columns.to_flat_index()
        ]

    df = df.reset_index()
    df.to_csv(DATA_DIR / "gold_prices_raw.csv", index=False)

    date_col = first_matching_column(
        df.columns,
        exact_names={"date", "datetime", "index"},
        prefixes=("date_", "datetime_"),
    )
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

    df = df.rename(columns={date_col: "Date", close_col: "Close", volume_col: "Volume"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df[df["Date"].dt.year.between(START_YEAR, END_YEAR)].copy()

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

    merged["cape_yield"] = 100 / merged["PE10"]
    merged["cape_yield_minus_10y_yield"] = merged["cape_yield"] - merged["Long Interest Rate"]
    merged["dividend_yield"] = (merged["Dividend"] / merged["SP500"]) * 100
    merged["sp500_annual_avg_yoy_change"] = merged["SP500"].pct_change() * 100
    merged["gold_annual_avg_yoy_change"] = merged["gold_close"].pct_change() * 100
    # Set gold YoY change to NaN for 2000 and 2001 due to partial data in 2000
    merged.loc[merged["Year"] <= 2001, "gold_annual_avg_yoy_change"] = np.nan
    merged["gdp_growth_lag1"] = merged["gdp_growth"].shift(1)

    assert merged["Year"].is_unique
    assert merged["Year"].min() == START_YEAR
    assert merged["Year"].max() == END_YEAR
    assert merged[["SP500", "PE10", "gdp_growth", "inflation_rate", "gold_close"]].notna().all().all()

    merged.to_csv(DATA_DIR / "macro_stock_merged.csv", index=False)
    return merged


def store_sqlite(
    sp500_clean: pd.DataFrame,
    world_bank_clean: pd.DataFrame,
    gold_clean: pd.DataFrame,
    merged: pd.DataFrame,
) -> None:
    with sqlite3.connect(DATA_DIR / "macro_stock_data.db") as conn:
        sp500_clean.to_sql("sp500_shiller_clean", conn, if_exists="replace", index=False)
        world_bank_clean.to_sql("world_bank_clean", conn, if_exists="replace", index=False)
        gold_clean.to_sql("gold_prices_clean", conn, if_exists="replace", index=False)
        merged.to_sql("macro_stock_merged", conn, if_exists="replace", index=False)


def save_figures(merged: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.regplot(data=merged, x="gdp_growth", y="PE10")
    plt.title("Shiller PE10 vs. Annual U.S. GDP Growth")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "visual1_pe10_vs_gdp.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.regplot(data=merged, x="inflation_rate", y="Long Interest Rate", label="10-year Treasury yield")
    sns.regplot(data=merged, x="inflation_rate", y="cape_yield", label="CAPE yield proxy")
    plt.title("Inflation vs. Yield Measures")
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
        "gold_close",
        "sp500_annual_avg_yoy_change",
        "gold_annual_avg_yoy_change",
    ]
    plt.figure(figsize=(10, 7))
    sns.heatmap(merged[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Exploratory)")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "visual3_correlation_heatmap.png", dpi=150)
    plt.close()

    regimes = merged.copy()
    regimes["inflation_regime"] = pd.cut(
        regimes["inflation_rate"],
        bins=[-np.inf, 2, 4, np.inf],
        labels=["Low (<2%)", "Moderate (2-4%)", "High (>4%)"],
    )
    plot_data = (
        regimes.groupby("inflation_regime", observed=True)[
            ["sp500_annual_avg_yoy_change", "gold_annual_avg_yoy_change"]
        ]
        .mean()
        .reset_index()
        .melt(id_vars="inflation_regime", var_name="Series", value_name="Average YoY change")
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_data, x="inflation_regime", y="Average YoY change", hue="Series")
    plt.title("Average Annual-Mean Price Changes by Inflation Regime")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "visual4_inflation_regimes.png", dpi=150)
    plt.close()

    # Visual 5: S&P 500 YoY Change (Year t) vs. Future GDP Growth (Year t+1)
    lead_data = merged.copy()
    lead_data["gdp_growth_lead1"] = lead_data["gdp_growth"].shift(-1)
    lag_data = lead_data.dropna(subset=["sp500_annual_avg_yoy_change", "gdp_growth_lead1"]).copy()
    plt.figure(figsize=(8, 5))
    sns.regplot(data=lag_data, x="sp500_annual_avg_yoy_change", y="gdp_growth_lead1")
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
    for t in range(n):
        xt = X_with_const[t, :].reshape(-1, 1)
        S += (e[t] ** 2) * (xt @ xt.T)
    for j in range(1, 3):
        weight = 1.0 - (j / 3.0)
        for t in range(j, n):
            xt = X_with_const[t, :].reshape(-1, 1)
            xt_lag = X_with_const[t - j, :].reshape(-1, 1)
            Gamma = e[t] * e[t - j] * (xt @ xt_lag.T)
            S += weight * (Gamma + Gamma.T)
    cov_hac = XtX_inv @ S @ XtX_inv
    se_hac = np.sqrt(np.diagonal(cov_hac))

    # t-statistics and two-tailed p-values (using HAC standard errors)
    t_stats_hac = beta / se_hac
    
    def normal_cdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    
    p_vals_hac = [2.0 * (1.0 - normal_cdf(abs(t))) for t in t_stats_hac]
    
    # 95% Confidence Intervals using critical value of 1.96 (normal approximation)
    ci_lower = beta - 1.96 * se_hac
    ci_upper = beta + 1.96 * se_hac

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

    print("====================================================================================================")
    print("                              MULTIPLE LINEAR REGRESSION ANALYSIS (OLS)")
    print("====================================================================================================")
    print(f"Dependent Variable:             S&P 500 Shiller PE10")
    print(f"Observations (n):               {n:<8} | Independent Variables (k): {k}")
    print(f"Residual Sum of Squares (RSS):  {rss:<8.4f} | Total Sum of Squares (TSS):    {tss:.4f}")
    print(f"Residual Standard Error (RSE):  {rse:<8.4f} | R-squared (R2):                {r2:.4f}")
    print(f"Durbin-Watson (DW) Statistic:   {dw:<8.4f} | Adjusted R-squared:            {adj_r2:.4f}")
    print(f"Jarque-Bera (JB) Statistic:     {jb:<8.4f} | JB p-value (approx):           {2.0 * (1.0 - normal_cdf(abs(math.sqrt(jb)))):.4f}")
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

"""Rebuild cleaned data, merged outputs, SQLite tables, and figures.

This script is the source of truth for the corrected notebook workflow. It fixes
three issues found during review:

1. Fetch World Bank data directly for USA instead of country/all.
2. Validate every annual merge with validate='one_to_one'.
3. Rename ambiguous engineered columns so they describe the actual calculation.
"""

from __future__ import annotations

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
    merged["sp500_annual_avg_yoy_change"] = merged["SP500"].pct_change() * 100
    merged["gold_annual_avg_yoy_change"] = merged["gold_close"].pct_change() * 100
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

    lag_data = merged.dropna(subset=["sp500_annual_avg_yoy_change", "gdp_growth"]).copy()
    plt.figure(figsize=(8, 5))
    sns.regplot(data=lag_data, x="sp500_annual_avg_yoy_change", y="gdp_growth")
    plt.title("S&P 500 Annual-Average YoY Change vs. Same-Year GDP Growth")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "visual5_lagged_lead_gdp.png", dpi=150)
    plt.close()


def main() -> None:
    sp500_clean = load_sp500()
    world_bank_clean = load_world_bank()
    gold_clean = load_gold_proxy()
    merged = build_merged_dataset(sp500_clean, world_bank_clean, gold_clean)
    store_sqlite(sp500_clean, world_bank_clean, gold_clean, merged)
    save_figures(merged)
    print("Rebuilt cleaned CSVs, merged output, SQLite database, and figures.")


if __name__ == "__main__":
    main()

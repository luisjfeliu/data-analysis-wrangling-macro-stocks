"""Validate committed cleaned and merged data outputs.

These checks are intentionally lightweight and focus on issues that can silently
distort the notebook's analysis: duplicate merge keys, blank World Bank ISO codes,
unexpected country rows, missing required values, and ambiguous return labeling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def require(condition: bool, message: str) -> None:
    """Raise an AssertionError with a readable message when a check fails."""
    if not condition:
        raise AssertionError(message)


def validate_world_bank_clean() -> None:
    path = DATA_DIR / "world_bank_clean.csv"
    df = pd.read_csv(path, dtype={"countryiso3code": "string"})

    required_columns = {
        "country_name",
        "countryiso3code",
        "date",
        "gdp_growth",
        "inflation_rate",
    }
    require(required_columns.issubset(df.columns), f"{path} is missing required columns")
    require(df["countryiso3code"].notna().all(), "World Bank clean file has null ISO codes")
    require((df["countryiso3code"].str.strip() != "").all(), "World Bank clean file has blank ISO codes")
    require(set(df["countryiso3code"]) == {"USA"}, "World Bank clean file should contain USA rows only")
    require(df["date"].is_unique, "World Bank clean file has duplicate years")
    require(df["date"].min() == 2000 and df["date"].max() == 2023, "Unexpected World Bank year range")
    require(df[["gdp_growth", "inflation_rate"]].notna().all().all(), "World Bank clean file has missing indicator values")


def validate_macro_stock_merged() -> None:
    path = DATA_DIR / "macro_stock_merged.csv"
    df = pd.read_csv(path)

    required_columns = {
        "Year",
        "SP500",
        "Long Interest Rate",
        "PE10",
        "gdp_growth",
        "inflation_rate",
        "gold_close",
        "cape_yield",
        "cape_yield_minus_10y_yield",
        "sp500_annual_avg_yoy_change",
        "gold_annual_avg_yoy_change",
        "gdp_growth_lag1",
    }
    require(required_columns.issubset(df.columns), f"{path} is missing required columns")

    retired_columns = {
        "earnings_yield",
        "equity_risk_premium",
        "sp500_annual_return",
        "gold_annual_return",
    }
    require(not retired_columns.intersection(df.columns), "Merged output still contains ambiguous retired column names")

    require(df["Year"].is_monotonic_increasing, "Merged output is not sorted by Year")
    require(df["Year"].is_unique, "Merged output has duplicate Year values")
    require(df["Year"].min() == 2000 and df["Year"].max() == 2023, "Unexpected merged year range")
    require(
        df[["SP500", "Long Interest Rate", "PE10", "gdp_growth", "inflation_rate", "gold_close"]]
        .notna()
        .all()
        .all(),
        "Merged output has missing required analytical values",
    )
    require(pd.isna(df.loc[df["Year"] == 2000, "sp500_annual_avg_yoy_change"]).all(), "First S&P YoY change should be NA")
    require(pd.isna(df.loc[df["Year"] == 2000, "gold_annual_avg_yoy_change"]).all(), "First gold YoY change should be NA")
    require(pd.isna(df.loc[df["Year"] == 2000, "gdp_growth_lag1"]).all(), "First GDP lag should be NA")


def main() -> None:
    validate_world_bank_clean()
    validate_macro_stock_merged()
    print("All data output validation checks passed.")


if __name__ == "__main__":
    main()

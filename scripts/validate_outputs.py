"""Validate committed raw, cleaned, merged, and SQLite data outputs.

These checks are intentionally lightweight and focus on issues that can silently
distort the notebook's analysis: duplicate merge keys, blank World Bank ISO codes,
unexpected country rows, missing required values, ambiguous return labeling, and
stale database tables.
"""


import json
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
START_YEAR = 2000
END_YEAR = 2023
EXPECTED_YEARS = set(range(START_YEAR, END_YEAR + 1))
SP500_MIN_MONTHS_BY_YEAR = {END_YEAR: 9}
GOLD_MIN_DAYS_BY_YEAR = {START_YEAR: 80}
DEFAULT_SP500_MIN_MONTHS = 12
DEFAULT_GOLD_MIN_DAYS = 200


def require(condition: bool, message: str) -> None:
    """Raise an AssertionError with a readable message when a check fails."""
    if not condition:
        raise AssertionError(message)


def require_columns(df: pd.DataFrame, required_columns: set[str], path: Path) -> None:
    missing_columns = sorted(required_columns - set(df.columns))
    require(not missing_columns, f"{path} is missing required columns: {missing_columns}")


def require_expected_years(years: pd.Series, label: str) -> None:
    actual_years = set(years.dropna().astype(int))
    missing_years = EXPECTED_YEARS - actual_years
    extra_years = actual_years - EXPECTED_YEARS
    require(not missing_years, f"{label} is missing years: {sorted(missing_years)}")
    require(not extra_years, f"{label} has unexpected years: {sorted(extra_years)}")


def require_yearly_coverage(
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
        require(
            actual_count >= min_count,
            (
                f"{label} has thin coverage for {year}: "
                f"{actual_count} observations, expected at least {min_count}"
            ),
        )


def validate_sp500_clean() -> None:
    path = DATA_DIR / "sp500_shiller_clean.csv"
    df = pd.read_csv(path)

    required_columns = {
        "Date",
        "SP500",
        "Dividend",
        "Earnings",
        "Consumer Price Index",
        "Long Interest Rate",
        "Real Dividend",
        "Real Earnings",
        "PE10",
    }
    require_columns(df, required_columns, path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    require(df["Date"].notna().all(), "S&P 500 clean file has invalid dates")
    require(df["Date"].is_unique, "S&P 500 clean file has duplicate dates")
    require_expected_years(df["Date"].dt.year, "S&P 500 clean file")
    require_yearly_coverage(
        df["Date"],
        "S&P 500 clean file",
        default_min_count=DEFAULT_SP500_MIN_MONTHS,
        min_count_by_year=SP500_MIN_MONTHS_BY_YEAR,
    )

    key_numeric_cols = ["SP500", "Consumer Price Index", "Long Interest Rate", "PE10"]
    for col in key_numeric_cols + ["Dividend", "Earnings", "Real Dividend", "Real Earnings"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    require(df[key_numeric_cols].notna().all().all(), "S&P 500 clean file has missing key values")

    placeholder_cols = ["Dividend", "Earnings", "Real Dividend", "Real Earnings", "PE10"]
    require(
        not (df[placeholder_cols] == 0).any().any(),
        "S&P 500 clean file still contains zero placeholders",
    )


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
    require_columns(df, required_columns, path)
    require(df["countryiso3code"].notna().all(), "World Bank clean file has null ISO codes")
    require((df["countryiso3code"].str.strip() != "").all(), "World Bank clean file has blank ISO codes")
    require(set(df["countryiso3code"]) == {"USA"}, "World Bank clean file should contain USA rows only")
    require(df["date"].is_unique, "World Bank clean file has duplicate years")
    require_expected_years(df["date"], "World Bank clean file")
    require(df[["gdp_growth", "inflation_rate"]].notna().all().all(), "World Bank clean file has missing indicator values")


def validate_gold_clean() -> None:
    path = DATA_DIR / "gold_prices_clean.csv"
    df = pd.read_csv(path)

    required_columns = {"Date", "Close", "Volume"}
    require_columns(df, required_columns, path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    require(df["Date"].notna().all(), "Gold clean file has invalid dates")
    require(df["Date"].is_unique, "Gold clean file has duplicate dates")
    require_expected_years(df["Date"].dt.year, "Gold clean file")
    require_yearly_coverage(
        df["Date"],
        "Gold clean file",
        default_min_count=DEFAULT_GOLD_MIN_DAYS,
        min_count_by_year=GOLD_MIN_DAYS_BY_YEAR,
    )

    key_numeric_cols = ["Close", "Volume"]
    for col in key_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    require(df[key_numeric_cols].notna().all().all(), "Gold clean file has missing key values")


def validate_macro_stock_merged() -> None:
    path = DATA_DIR / "macro_stock_merged.csv"
    df = pd.read_csv(path)

    required_columns = {
        "Year",
        "SP500",
        "Dividend",
        "Earnings",
        "Long Interest Rate",
        "PE10",
        "gdp_growth",
        "inflation_rate",
        "gold_close",
        "dividend_yield",
        "cape_yield",
        "cape_yield_minus_10y_yield",
        "sp500_annual_avg_yoy_change",
        "gold_annual_avg_yoy_change",
        "gdp_growth_lag1",
    }
    require_columns(df, required_columns, path)

    retired_columns = {
        "earnings_yield",
        "equity_risk_premium",
        "sp500_annual_return",
        "gold_annual_return",
    }
    require(not retired_columns.intersection(df.columns), "Merged output still contains ambiguous retired column names")

    require(df["Year"].is_monotonic_increasing, "Merged output is not sorted by Year")
    require(df["Year"].is_unique, "Merged output has duplicate Year values")
    require_expected_years(df["Year"], "Merged output")
    require(
        df[
            [
                "SP500",
                "Dividend",
                "Earnings",
                "Long Interest Rate",
                "PE10",
                "gdp_growth",
                "inflation_rate",
                "gold_close",
                "dividend_yield",
            ]
        ]
        .notna()
        .all()
        .all(),
        "Merged output has missing required analytical values",
    )
    require(pd.isna(df.loc[df["Year"] == START_YEAR, "sp500_annual_avg_yoy_change"]).all(), "First S&P YoY change should be NA")
    require(pd.isna(df.loc[df["Year"] <= START_YEAR + 1, "gold_annual_avg_yoy_change"]).all(), "Gold YoY change for 2000 and 2001 should be NA due to partial 2000 data")
    require(pd.isna(df.loc[df["Year"] == START_YEAR, "gdp_growth_lag1"]).all(), "First GDP lag should be NA")
    require(df.loc[df["Year"] > START_YEAR, "sp500_annual_avg_yoy_change"].notna().all(), "Non-boundary S&P YoY change values should not be NaN")
    require(df.loc[df["Year"] > START_YEAR + 1, "gold_annual_avg_yoy_change"].notna().all(), "Non-boundary gold YoY change values should not be NaN")
    require(df.loc[df["Year"] > START_YEAR, "gdp_growth_lag1"].notna().all(), "Non-boundary GDP lag values should not be NaN")


def validate_sqlite_store() -> None:
    path = DATA_DIR / "macro_stock_data.db"
    require(path.exists(), f"SQLite data store is missing: {path}")

    expected_tables = {
        "sp500_shiller_raw",
        "world_bank_gdp_raw",
        "world_bank_inflation_raw",
        "gold_prices_raw",
        "sp500_shiller_clean",
        "world_bank_clean",
        "gold_prices_clean",
        "macro_stock_merged",
    }
    with sqlite3.connect(path) as conn:
        actual_tables = {
            row[0]
            for row in conn.execute(
                "select name from sqlite_master where type='table'"
            ).fetchall()
        }
        require(
            expected_tables.issubset(actual_tables),
            f"SQLite data store is missing tables: {sorted(expected_tables - actual_tables)}",
        )

        csv_row_counts = {
            "sp500_shiller_raw": len(pd.read_csv(DATA_DIR / "sp500_shiller_raw.csv")),
            "gold_prices_raw": len(pd.read_csv(DATA_DIR / "gold_prices_raw.csv")),
            "sp500_shiller_clean": len(pd.read_csv(DATA_DIR / "sp500_shiller_clean.csv")),
            "world_bank_clean": len(pd.read_csv(DATA_DIR / "world_bank_clean.csv")),
            "gold_prices_clean": len(pd.read_csv(DATA_DIR / "gold_prices_clean.csv")),
            "macro_stock_merged": len(pd.read_csv(DATA_DIR / "macro_stock_merged.csv")),
        }
        for table, expected_count in csv_row_counts.items():
            actual_count = conn.execute(f'select count(*) from "{table}"').fetchone()[0]
            require(
                actual_count == expected_count,
                f"SQLite table {table} has {actual_count} rows; expected {expected_count}",
            )

        raw_gdp_rows = len(json.loads((DATA_DIR / "world_bank_gdp_raw.json").read_text())[1])
        raw_inflation_rows = len(
            json.loads((DATA_DIR / "world_bank_inflation_raw.json").read_text())[1]
        )
        for table, expected_count in {
            "world_bank_gdp_raw": raw_gdp_rows,
            "world_bank_inflation_raw": raw_inflation_rows,
        }.items():
            actual_count = conn.execute(f'select count(*) from "{table}"').fetchone()[0]
            iso_codes = {
                row[0]
                for row in conn.execute(
                    f'select distinct countryiso3code from "{table}"'
                ).fetchall()
            }
            require(
                actual_count == expected_count,
                f"SQLite table {table} has {actual_count} rows; expected {expected_count}",
            )
            require(iso_codes == {"USA"}, f"SQLite table {table} should contain USA raw rows only")


def main() -> None:
    validate_sp500_clean()
    validate_world_bank_clean()
    validate_gold_clean()
    validate_macro_stock_merged()
    validate_sqlite_store()
    print("All data output validation checks passed.")


if __name__ == "__main__":
    main()

"""Validate committed raw, cleaned, merged, and SQLite data outputs.

These checks are intentionally lightweight and focus on issues that can silently
distort the notebook's analysis: duplicate merge keys, blank World Bank ISO codes,
unexpected country rows, missing required values, ambiguous return labeling, and
stale database tables.
"""


import json
import hashlib
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
SP500_RAW_SHA256 = "3fe682b8dd593beb2548092d2a5e9b8844c2adc0f512da200dc5725d390ecfc9"
SP500_EXPECTED_2023_NON_NULL_COUNTS = {
    "SP500": 9,
    "Dividend": 6,
    "Earnings": 6,
    "Consumer Price Index": 9,
    "Long Interest Rate": 9,
    "Real Price": 9,
    "Real Dividend": 6,
    "Real Earnings": 6,
    "PE10": 9,
}


def require(condition: bool, message: str) -> None:
    """Raise an AssertionError with a readable message when a check fails."""
    if not condition:
        raise AssertionError(message)


def require_columns(df: pd.DataFrame, required_columns: set[str], path: Path) -> None:
    missing_columns = sorted(required_columns - set(df.columns))
    require(not missing_columns, f"{path} is missing required columns: {missing_columns}")


def prepare_sqlite_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert nested raw API objects to JSON strings before SQLite comparison."""
    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda value: isinstance(value, (dict, list))).any():
            out[col] = out[col].map(
                lambda value: json.dumps(value) if isinstance(value, (dict, list)) else value
            )
    return out


def require_table_matches_frame(
    conn: sqlite3.Connection,
    table: str,
    expected: pd.DataFrame,
) -> None:
    """Validate that a SQLite table is not just present, but matches the stored source."""
    actual = pd.read_sql_query(f'select * from "{table}"', conn)
    actual = normalize_date_like_columns(actual)
    expected = normalize_date_like_columns(expected)
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )
    except AssertionError as exc:
        raise AssertionError(f"SQLite table {table} content does not match source data: {exc}") from exc


def normalize_date_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize date-like storage strings so CSV and SQLite can be compared."""
    out = df.copy()
    for col in out.columns:
        col_name = str(col).lower()
        if col_name == "index" or "date" in col_name:
            parsed = pd.to_datetime(out[col], errors="coerce")
            if parsed.notna().all():
                out[col] = parsed.dt.strftime("%Y-%m-%d")
    return out


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
    raw_path = DATA_DIR / "sp500_shiller_raw.csv"
    actual_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    require(
        actual_hash == SP500_RAW_SHA256,
        (
            "S&P 500 raw cache hash changed; update the frozen source hash "
            "only after reviewing upstream data changes"
        ),
    )

    path = DATA_DIR / "sp500_shiller_clean.csv"
    df = pd.read_csv(path)

    required_columns = {
        "Date",
        "SP500",
        "Dividend",
        "Earnings",
        "Consumer Price Index",
        "Long Interest Rate",
        "Real Price",
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

    key_numeric_cols = ["SP500", "Consumer Price Index", "Long Interest Rate", "Real Price", "PE10"]
    for col in key_numeric_cols + ["Dividend", "Earnings", "Real Dividend", "Real Earnings"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    require(df[key_numeric_cols].notna().all().all(), "S&P 500 clean file has missing key values")

    placeholder_cols = [
        "Dividend",
        "Earnings",
        "Consumer Price Index",
        "Long Interest Rate",
        "Real Price",
        "Real Dividend",
        "Real Earnings",
        "PE10",
    ]
    require(
        not (df[placeholder_cols] == 0).any().any(),
        "S&P 500 clean file still contains zero placeholders",
    )
    df_2023 = df[df["Date"].dt.year == END_YEAR]
    actual_counts = df_2023[list(SP500_EXPECTED_2023_NON_NULL_COUNTS)].count().to_dict()
    require(
        actual_counts == SP500_EXPECTED_2023_NON_NULL_COUNTS,
        (
            "S&P 500 clean file has unexpected 2023 non-null counts: "
            f"{actual_counts}; expected {SP500_EXPECTED_2023_NON_NULL_COUNTS}"
        ),
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
        "sp500_months_used",
        "dividend_months_used",
        "earnings_months_used",
        "sp500_coverage_label",
        "sp500_annual_avg_yoy_change",
        "gold_annual_avg_yoy_change",
        "gdp_growth_lag1",
        "gdp_growth_lead1",
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
        df.loc[df["Year"] == END_YEAR, "sp500_months_used"].eq(9).all(),
        "Merged output should label 2023 S&P metrics as using 9 aligned months",
    )
    require(
        df.loc[df["Year"] == END_YEAR, "dividend_months_used"].eq(6).all(),
        "Merged output should label 2023 dividend metrics as using 6 aligned months",
    )
    require(
        df.loc[df["Year"] == END_YEAR, "earnings_months_used"].eq(6).all(),
        "Merged output should label 2023 earnings metrics as using 6 aligned months",
    )
    require(
        df.loc[df["Year"] == END_YEAR, "sp500_coverage_label"]
        .eq("aligned_9_month_partial_year")
        .all(),
        "Merged output should label 2023 S&P metrics as aligned 9-month partial-year values",
    )
    require(
        df.loc[df["Year"] < END_YEAR, "sp500_coverage_label"].eq("full_year").all(),
        "Merged output should label pre-2023 S&P metrics as full-year values",
    )
    require(
        df.loc[
            df["Year"] < END_YEAR,
            ["sp500_months_used", "dividend_months_used", "earnings_months_used"],
        ]
        .eq(DEFAULT_SP500_MIN_MONTHS)
        .all()
        .all(),
        "Merged output should label pre-2023 S&P, dividend, and earnings metrics as 12-month values",
    )
    sp500_clean = pd.read_csv(DATA_DIR / "sp500_shiller_clean.csv")
    sp500_clean["Date"] = pd.to_datetime(sp500_clean["Date"], errors="coerce")
    expected_dividend_yield = (
        sp500_clean.dropna(subset=["SP500", "Dividend"])
        .assign(Year=lambda frame: frame["Date"].dt.year)
        .groupby("Year", as_index=False)
        .agg(dividend_mean=("Dividend", "mean"), dividend_sp500_mean=("SP500", "mean"))
    )
    expected_dividend_yield["expected_dividend_yield"] = (
        expected_dividend_yield["dividend_mean"]
        / expected_dividend_yield["dividend_sp500_mean"]
        * 100
    )
    dividend_yield_check = pd.merge(
        df[["Year", "dividend_yield"]],
        expected_dividend_yield[["Year", "expected_dividend_yield"]],
        on="Year",
        how="inner",
        validate="one_to_one",
    )
    require(
        len(dividend_yield_check) == len(df),
        "Dividend yield validation should cover every merged year",
    )
    max_yield_diff = (
        dividend_yield_check["dividend_yield"]
        - dividend_yield_check["expected_dividend_yield"]
    ).abs().max()
    require(
        max_yield_diff < 1e-10,
        "Merged dividend_yield should use months where both Dividend and SP500 are available",
    )
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
    require(pd.isna(df.loc[df["Year"] == END_YEAR, "sp500_annual_avg_yoy_change"]).all(), "Partial-year 2023 S&P YoY change should be NA")
    require(pd.isna(df.loc[df["Year"] <= START_YEAR + 1, "gold_annual_avg_yoy_change"]).all(), "Gold YoY change for 2000 and 2001 should be NA due to partial 2000 data")
    require(pd.isna(df.loc[df["Year"] == START_YEAR, "gdp_growth_lag1"]).all(), "First GDP lag should be NA")
    require(pd.isna(df.loc[df["Year"] == END_YEAR, "gdp_growth_lead1"]).all(), "Last GDP lead should be NA")
    require(
        df.loc[
            df["Year"].between(START_YEAR + 1, END_YEAR - 1),
            "sp500_annual_avg_yoy_change",
        ]
        .notna()
        .all(),
        "Full-year non-boundary S&P YoY change values should not be NaN",
    )
    require(df.loc[df["Year"] > START_YEAR + 1, "gold_annual_avg_yoy_change"].notna().all(), "Non-boundary gold YoY change values should not be NaN")
    require(df.loc[df["Year"] > START_YEAR, "gdp_growth_lag1"].notna().all(), "Non-boundary GDP lag values should not be NaN")
    require(df.loc[df["Year"] < END_YEAR, "gdp_growth_lead1"].notna().all(), "Non-boundary GDP lead values should not be NaN")


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

        expected_frames = {
            "sp500_shiller_raw": pd.read_csv(DATA_DIR / "sp500_shiller_raw.csv"),
            "gold_prices_raw": pd.read_csv(DATA_DIR / "gold_prices_raw.csv"),
            "sp500_shiller_clean": pd.read_csv(DATA_DIR / "sp500_shiller_clean.csv"),
            "world_bank_clean": pd.read_csv(DATA_DIR / "world_bank_clean.csv"),
            "gold_prices_clean": pd.read_csv(DATA_DIR / "gold_prices_clean.csv"),
            "macro_stock_merged": pd.read_csv(DATA_DIR / "macro_stock_merged.csv"),
            "world_bank_gdp_raw": prepare_sqlite_frame(
                pd.DataFrame(json.loads((DATA_DIR / "world_bank_gdp_raw.json").read_text())[1])
            ),
            "world_bank_inflation_raw": prepare_sqlite_frame(
                pd.DataFrame(
                    json.loads((DATA_DIR / "world_bank_inflation_raw.json").read_text())[1]
                )
            ),
        }
        for table, expected_frame in expected_frames.items():
            expected_count = len(expected_frame)
            actual_count = conn.execute(f'select count(*) from "{table}"').fetchone()[0]
            require(
                actual_count == expected_count,
                f"SQLite table {table} has {actual_count} rows; expected {expected_count}",
            )
            require_table_matches_frame(conn, table, expected_frame)

        for table in ["world_bank_gdp_raw", "world_bank_inflation_raw"]:
            iso_codes = {
                row[0]
                for row in conn.execute(
                    f'select distinct countryiso3code from "{table}"'
                ).fetchall()
            }
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

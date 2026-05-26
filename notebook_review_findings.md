# Notebook Review Findings
**Repository:** `luisjfeliu/data-analysis-wrangling-macro-stocks`  
**Files reviewed:** `Data_Wrangling_Project_Starter.ipynb`, `scripts/validate_outputs.py`, `scripts/rebuild_outputs.py`, `requirements.txt`  
**Review date:** 2026-05-26  
**Method:** 3-angle independent scan (line-by-line diff, removed-behavior audit, cross-file trace) + per-candidate verification

---

## Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | 🔴 High | Notebook cells 8 / 16 | `PE10=0` → `dropna` silently drops Oct–Dec of `END_YEAR`, biasing annual averages |
| 2 | 🔴 High | Notebook cells 13–14 | `GC=F` starts Aug 2000; year-2000 gold average is a 5-month partial-year figure |
| 3 | 🔴 High | Notebook cells 16 / 18 | `pct_change()` gap-year risk — no row-count assertion |
| 4 | 🔴 High | `validate_outputs.py:88` | `sp500_shiller_clean.csv` and `gold_prices_clean.csv` are never validated |
| 5 | 🟠 Medium | Notebook cells 18 / 26 | `gdp_growth_lag1` engineered but never visualized; `lag_data` is same-year, not lagged |
| 6 | 🟠 Medium | Notebook cell 0 | "Assesses quality issues" stated in intro but no assessment cell exists |
| 7 | 🟠 Medium | Notebook cell 8 | `Long Interest Rate` and `CPI` zeros not replaced — only `PE10` is sanitized |
| 8 | 🟠 Medium | Notebook cell 16 | `notna` assertion omits `Long Interest Rate` |
| 9 | 🟠 Medium | Notebook cell 11 | World Bank merge key includes `country_name`; whitespace mismatch silently drops USA |
| 10 | 🟠 Medium | `validate_outputs.py:83` | NaN sentinels only verify year 2000; an all-NaN column still passes |
| 11 | 🟠 Medium | `validate_outputs.py:41,75` | Year bounds hard-coded as literals, not shared with notebook constants |
| 12 | 🟠 Medium | Notebook cells 27–28 | Outlier check covers only `PE10`; `gdp_growth`, `inflation_rate`, `gold_close` unchecked |
| 13 | 🟡 Low | `validate_outputs.py:27` | `date` dtype not pinned on CSV re-read; null year passes `2000.0 == 2000` |
| 14 | 🟡 Low | `validate_outputs.py:49` | `required_columns` missing `Dividend`, `Earnings`, `Consumer Price Index`, `gold_volume` |
| 15 | 🟡 Low | `rebuild_outputs.py:24` | Bare `Path("data")` breaks when script is run outside repo root |

---

## Detailed Findings

---

### 🔴 Finding 1 — `PE10 = 0` silently drops Oct–Dec of `END_YEAR`, biasing the annual S&P average

**Files:** `Data_Wrangling_Project_Starter.ipynb` cells 8 & 16; `scripts/rebuild_outputs.py` lines 102, 113

**What happens:**  
The Shiller CSV publishes `PE10 = 0.0` for months where the CAPE ratio has not yet been computed (typically the trailing 2–3 months of the most recent year). The notebook correctly converts those zeros to `NaN`:

```python
df_sp500_clean["PE10"] = df_sp500_clean["PE10"].replace(0, np.nan)
```

But `PE10` is also included in `required_sp500_cols`, so the subsequent `dropna` silently discards those months:

```python
required_sp500_cols = ["Date", "SP500", "Dividend", "Earnings",
                       "Consumer Price Index", "Long Interest Rate", "PE10"]
df_sp500_clean = df_sp500_clean.dropna(subset=required_sp500_cols)
```

**Impact:**  
Oct–Dec 2023 — rows with valid `SP500`, `Dividend`, `Earnings`, `CPI`, and `Long Interest Rate` values — are dropped entirely. The 2023 annual average is computed on 9 months instead of 12, underestimating the full-year S&P 500 level by approximately 59 points (~1.4%). All existing assertions pass silently: `year.max() == END_YEAR` is satisfied by the September row, and `Year.max() == END_YEAR` is satisfied in the merge.

**Recommended fix:**  
Add a per-year row-count warning after the `dropna` step:

```python
monthly_counts = df_sp500_clean.groupby(df_sp500_clean["Date"].dt.year).size()
thin_years = monthly_counts[monthly_counts < 10]
if not thin_years.empty:
    print(f"WARNING: Fewer than 10 months of S&P data for: {thin_years.to_dict()}")
```

Alternatively, remove `PE10` from `required_sp500_cols` and forward-fill trailing `NaN` PE10 values (since CAPE changes slowly month to month), preserving the full-year equity data.

---

### 🔴 Finding 2 — `GC=F` gold futures data starts 2000-08-30; year-2000 annual average is a 5-month partial figure

**Files:** `Data_Wrangling_Project_Starter.ipynb` cells 13–14; `scripts/rebuild_outputs.py` line 195

**What happens:**  
Yahoo Finance's `GC=F` continuous gold futures contract has no data before approximately 2000-08-30. The year filter retains 2000 rows without checking per-year coverage:

```python
df_gold_clean = df_gold_clean[
    df_gold_clean["Date"].dt.year.between(START_YEAR, END_YEAR)
].copy()
```

**Impact:**  
The year-2000 `gold_close` annual average (~270) is computed on only ~84 trading days (August–December) instead of ~252. This makes `gold_annual_avg_yoy_change` for 2001 a biased comparison: a full-year 2001 average divided by a partial-year 2000 baseline. No assertion anywhere checks minimum per-year trading day counts.

**Recommended fix:**  
Add a minimum-coverage guard after filtering:

```python
gold_year_counts = df_gold_clean.groupby(df_gold_clean["Date"].dt.year).size()
assert (gold_year_counts >= 200).all(), \
    f"Gold data has partial-year coverage: {gold_year_counts[gold_year_counts < 200].to_dict()}"
```

Or set `START_YEAR = 2001` to avoid the partial-year 2000 baseline entirely (and update `validate_outputs.py` accordingly).

---

### 🔴 Finding 3 — `pct_change()` silently produces wrong YoY values if any year is missing from the merge

**Files:** `Data_Wrangling_Project_Starter.ipynb` cells 16 & 18

**What happens:**  
After the three-way inner join, assertions only check:

```python
assert df_merged["Year"].min() == START_YEAR
assert df_merged["Year"].max() == END_YEAR
assert df_merged["Year"].is_unique
```

There is no check that all 24 years are present. In cell 18:

```python
df_merged["sp500_annual_avg_yoy_change"] = df_merged["SP500"].pct_change() * 100
df_merged["gold_annual_avg_yoy_change"]  = df_merged["gold_close"].pct_change() * 100
```

`pct_change()` computes changes between consecutive *rows*, not between consecutive *years*. A gap year causes the two rows flanking the gap to produce a two-year return reported as a one-year return, with no error or warning.

**Recommended fix:**  
Add immediately after the sort in cell 16:

```python
expected_years = set(range(START_YEAR, END_YEAR + 1))
actual_years   = set(df_merged["Year"])
missing_years  = expected_years - actual_years
assert not missing_years, f"Merged data is missing years: {sorted(missing_years)}"
```

---

### 🔴 Finding 4 — `validate_outputs.py` never validates `sp500_shiller_clean.csv` or `gold_prices_clean.csv`

**File:** `scripts/validate_outputs.py` lines 88–91

**What happens:**  
`main()` calls only two validators:

```python
def main() -> None:
    validate_world_bank_clean()
    validate_macro_stock_merged()
    print("All data output validation checks passed.")
```

The notebook produces three cleaned intermediate files (`sp500_shiller_clean.csv`, `world_bank_clean.csv`, `gold_prices_clean.csv`) that are all critical inputs to the annual merge. Only the World Bank file is validated at the source; the S&P and gold files are only indirectly checked through the merged output.

**Impact:**  
A corrupted `sp500_shiller_clean.csv` (wrong year range, duplicate rows, all-zero PE10) or a bad `gold_prices_clean.csv` (wrong `Close` column, incorrect year span) passes validation silently and propagates errors into every downstream chart and table.

**Recommended fix:**  
Add `validate_sp500_clean()` and `validate_gold_clean()` functions with at minimum: required column checks, date-range assertions, row-count-per-year checks, and `notna` assertions on key numeric columns. Call both from `main()`.

---

### 🟠 Finding 5 — `gdp_growth_lag1` is engineered but never visualized; Visual 5 uses same-year data despite the variable name `lag_data`

**Files:** `Data_Wrangling_Project_Starter.ipynb` cells 18 & 26

**What happens:**  
Cell 18 computes:

```python
df_merged["gdp_growth_lag1"] = df_merged["gdp_growth"].shift(1)
```

`validate_outputs.py` even confirms it is `NaN` for year 2000. But none of the five visualizations (cells 22–26) use `gdp_growth_lag1`. Cell 26 is named `lag_data` and titled *"S&P 500 Annual-Average YoY Change vs. Same-Year GDP Growth"*, yet plots `gdp_growth` (same-year), not `gdp_growth_lag1`:

```python
lag_data = df_merged.dropna(subset=["sp500_annual_avg_yoy_change", "gdp_growth"]).copy()
sns.regplot(data=lag_data, x="sp500_annual_avg_yoy_change", y="gdp_growth")
```

The project description explicitly promises a lead-lag analysis, and `gdp_growth_lag1` is the only feature that could deliver it.

**Recommended fix:**  
Either replace `gdp_growth` with `gdp_growth_lag1` on the y-axis in cell 26 to produce the intended lead-lag scatter, or rename the variable to `same_year_data` and update the section heading to remove the lead-lag implication.

---

### 🟠 Finding 6 — Intro claims "assesses quality issues" but no assessment cell exists in the notebook

**File:** `Data_Wrangling_Project_Starter.ipynb` cell 0

**What happens:**  
Cell 0 describes the workflow as:

> "The workflow gathers raw data, **assesses quality issues**, cleans and validates the data, merges annual observations, stores outputs, and produces exploratory visualizations."

No cell in the 32-cell notebook calls `.isnull().sum()`, `.info()`, or `.describe()` on any raw DataFrame before cleaning. Loading and cleaning happen in the same sections with no diagnostic checkpoint between them.

**Impact:**  
The stated workflow is not implemented. For a data wrangling project, the assess step is typically required to document (a) quality issues (missing values, wrong types, duplicates) and (b) tidiness issues, each identified both visually and programmatically, before any cleaning code is written. Omitting it misrepresents what the notebook does and would fail a typical data wrangling rubric's "assess" criterion.

**Recommended fix:**  
Add a **Section 2.5 – Data Quality Assessment** (or insert sub-cells after each raw load) showing, for each source:
- `df_raw.info()` — column types and non-null counts
- `df_raw.isnull().sum()` — null counts per column
- `df_raw.describe()` — numeric distributions
- A brief written description of each identified quality or tidiness issue

---

### 🟠 Finding 7 — `Long Interest Rate` and `Consumer Price Index` zeros are not replaced; only `PE10` is sanitized

**File:** `Data_Wrangling_Project_Starter.ipynb` cell 8

**What happens:**  
The Shiller dataset uses `0.0` as a placeholder for unavailable values in multiple columns for pre-modern rows. The notebook only sanitizes `PE10`:

```python
# PE10 appears as 0.0 in early rows where CAPE is not available.
df_sp500_clean["PE10"] = df_sp500_clean["PE10"].replace(0, np.nan)
```

`Long Interest Rate` and `Consumer Price Index` are passed through without a zero-replacement step. The year filter (`between(2000, 2023)`) removes most historic zero-placeholder rows, but if the upstream source ever introduces a zero-valued entry in the 2000–2023 window, it would silently bias the annual means used in every visualization. The `notna()` check in `validate_macro_stock_merged` does not catch zeros.

**Recommended fix:**  
Apply the same guard to all three columns (or to all numeric columns in the dataset):

```python
for col in ["PE10", "Long Interest Rate", "Consumer Price Index"]:
    df_sp500_clean[col] = df_sp500_clean[col].replace(0, np.nan)
```

---

### 🟠 Finding 8 — Cell 16's `notna` assertion omits `Long Interest Rate`

**File:** `Data_Wrangling_Project_Starter.ipynb` cell 16

**What happens:**  
The post-merge integrity assertion is:

```python
assert df_merged[["SP500", "PE10", "gdp_growth", "inflation_rate", "gold_close"]].notna().all().all()
```

`Long Interest Rate` is absent from this list, even though it is used directly in `cape_yield_minus_10y_yield` (cell 18) and is included in `validate_outputs.py`'s own `notna` check (line 77). A null `Long Interest Rate` for any year produces a null `cape_yield_minus_10y_yield` that flows silently into the correlation heatmap and Visual 2.

**Recommended fix:**  
Add `"Long Interest Rate"` to the column list:

```python
assert df_merged[["SP500", "PE10", "Long Interest Rate",
                   "gdp_growth", "inflation_rate", "gold_close"]].notna().all().all()
```

---

### 🟠 Finding 9 — World Bank inner merge uses `country_name` as a key; API whitespace differences silently drop the USA row

**Files:** `Data_Wrangling_Project_Starter.ipynb` cell 11; `scripts/rebuild_outputs.py` line 133

**What happens:**  
The GDP and inflation DataFrames are merged on three columns:

```python
pd.merge(df_gdp_clean, df_inf_clean,
         on=["country_name", "countryiso3code", "date"],
         how="inner", validate="one_to_one")
```

`country_name` is extracted from the `"country"."value"` field of the World Bank API response — a free-text string from two separate HTTP calls. If the API returns `"United States"` with a trailing non-breaking space in one response but not the other (a known World Bank API inconsistency), the `inner` merge silently drops the USA row before the ISO filter is applied. The subsequent assertion `df_world_bank_clean["date"].min() == START_YEAR` then raises a confusing `TypeError` (comparing `NAType` to `int`) rather than a clear domain error.

**Recommended fix:**  
Remove `country_name` from the merge key, since `countryiso3code` already uniquely identifies the country:

```python
pd.merge(df_gdp_clean, df_inf_clean,
         on=["countryiso3code", "date"],
         how="inner", validate="one_to_one")
```

---

### 🟠 Finding 10 — NaN sentinel checks only verify year-2000 is NaN; an all-NaN column still passes all checks

**File:** `scripts/validate_outputs.py` lines 83–85

**What happens:**  
```python
require(pd.isna(df.loc[df["Year"] == 2000, "sp500_annual_avg_yoy_change"]).all(),
        "First S&P YoY change should be NA")
require(pd.isna(df.loc[df["Year"] == 2000, "gold_annual_avg_yoy_change"]).all(),
        "First gold YoY change should be NA")
require(pd.isna(df.loc[df["Year"] == 2000, "gdp_growth_lag1"]).all(),
        "First GDP lag should be NA")
```

These checks confirm the first-row behaviour of `pct_change()` and `shift()`, but they do not verify that post-2000 rows are non-NaN. If a dtype bug caused `pct_change()` to return all-NaN (e.g., the `SP500` column was inadvertently cast to string), every row would be NaN, all three checks would still pass, and the analysis charts would render as blank regression lines.

**Recommended fix:**  
Add a complementary assertion for each column:

```python
require(df.loc[df["Year"] > 2000, "sp500_annual_avg_yoy_change"].notna().all(),
        "Non-2000 S&P YoY change values should not be NaN")
require(df.loc[df["Year"] > 2000, "gold_annual_avg_yoy_change"].notna().all(),
        "Non-2000 gold YoY change values should not be NaN")
require(df.loc[df["Year"] > 2001, "gdp_growth_lag1"].notna().all(),
        "Non-2001 GDP lag values should not be NaN")
```

---

### 🟠 Finding 11 — Year range bounds hard-coded in `validate_outputs.py`, not shared with notebook constants

**File:** `scripts/validate_outputs.py` lines 41, 75

**What happens:**  
```python
require(df["date"].min() == 2000 and df["date"].max() == 2023,
        "Unexpected World Bank year range")
```

The notebook and `rebuild_outputs.py` both use `START_YEAR = 2000` / `END_YEAR = 2023` as named constants. `validate_outputs.py` hard-codes the integer literals. If the date range is extended or narrowed in either script, the validator will incorrectly raise `AssertionError` without indicating that constants have drifted.

**Recommended fix:**  
Define `START_YEAR = 2000` and `END_YEAR = 2023` at the top of `validate_outputs.py` and reference them in all assertions, or extract them to a shared `config.py` module imported by all three files.

---

### 🟠 Finding 12 — Outlier check covers only `PE10`; `gdp_growth`, `inflation_rate`, and `gold_close` are not checked

**File:** `Data_Wrangling_Project_Starter.ipynb` cells 27–28

**What happens:**  
Section 10 runs IQR-based outlier detection on `PE10` only. The original notebook applied the same method across all four key series. Notable outliers that go unflagged:

- **2009 GDP:** −2.58% (financial crisis contraction)
- **2021 GDP:** +5.95% (post-pandemic rebound)
- **2022 CPI inflation:** 8.0% (highest since 1981)

Without explicit documentation that these were inspected and retained as genuine economic extremes, an accidental bad data row (e.g., a wrong country mixed into the World Bank fetch for one year) would produce an equally extreme value that passes all existing checks and distorts every regression chart.

**Recommended fix:**  
Extend Section 10 to run the IQR check on all four series:

```python
for col in ["PE10", "gdp_growth", "inflation_rate", "gold_close"]:
    q1, q3 = df_merged[col].quantile(0.25), df_merged[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df_merged[
        (df_merged[col] < q1 - 1.5 * iqr) | (df_merged[col] > q3 + 1.5 * iqr)
    ][["Year", col]]
    print(f"\n{col} outliers (IQR method):")
    display(outliers)
    print("  → Retained as genuine economic observations.")
```

---

### 🟡 Finding 13 — `date` column dtype not pinned when `validate_outputs.py` re-reads `world_bank_clean.csv`

**File:** `scripts/validate_outputs.py` line 27

**What happens:**  
```python
df = pd.read_csv(path, dtype={"countryiso3code": "string"})
```

The `date` column (written as pandas `Int64`) is not explicitly typed on re-read. If a null year exists (the World Bank API occasionally omits the `date` field for preliminary data), `Int64.to_csv()` writes it as an empty cell, and pandas infers the column as `float64` on re-read. The comparison `df["date"].min() == 2000` then evaluates as `2000.0 == 2000` → `True`, masking the corrupted null-year row. `is_unique` in pandas also ignores `NaN` by default.

**Recommended fix:**  
Pin the dtype on read:

```python
df = pd.read_csv(path, dtype={"countryiso3code": "string", "date": "Int64"})
```

---

### 🟡 Finding 14 — `required_columns` in `validate_outputs.py` is missing several notebook-produced columns

**File:** `scripts/validate_outputs.py` lines 49–62

**What happens:**  
The validator checks for `SP500`, `Long Interest Rate`, `PE10`, `gdp_growth`, `inflation_rate`, `gold_close`, and the five engineered columns. The following columns that the notebook writes into `macro_stock_merged.csv` are not in `required_columns` and would not be caught if accidentally removed or renamed:

- `Dividend`
- `Earnings`
- `Consumer Price Index`
- `gold_volume`

**Recommended fix:**  
Add the missing column names to `required_columns`:

```python
required_columns = {
    "Year", "SP500", "Dividend", "Earnings", "Consumer Price Index",
    "Long Interest Rate", "PE10", "gdp_growth", "inflation_rate",
    "gold_close", "gold_volume", "cape_yield", "cape_yield_minus_10y_yield",
    "sp500_annual_avg_yoy_change", "gold_annual_avg_yoy_change", "gdp_growth_lag1",
}
```

---

### 🟡 Finding 15 — `rebuild_outputs.py` uses bare `Path("data")`, breaks when run outside repo root

**File:** `scripts/rebuild_outputs.py` lines 24–25

**What happens:**  
```python
DATA_DIR  = Path("data")    # resolved relative to cwd, not to __file__
IMAGE_DIR = Path("images")
```

Compare with `validate_outputs.py` line 15, which correctly anchors to the script's own location:

```python
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
```

Running `python scripts/rebuild_outputs.py` from inside the `scripts/` subdirectory writes outputs to `scripts/data/`, while `validate_outputs.py` reads from the repo-root `data/` — producing a `FileNotFoundError` that looks like missing data rather than a path issue.

**Recommended fix:**  
Apply the same pattern as `validate_outputs.py`:

```python
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
IMAGE_DIR = ROOT / "images"
```

---

## How findings were surfaced

Three independent review angles ran in parallel:

- **Angle A — Line-by-line diff scan:** Read every hunk and the enclosing cell/function, asking what input or state makes each changed line wrong.
- **Angle B — Removed-behavior auditor:** For every section deleted from the 93-cell original, identified what invariant it enforced and verified whether the new code re-establishes it.
- **Angle C — Cross-file tracer:** Checked callers and callees across `validate_outputs.py`, `rebuild_outputs.py`, and the notebook for precondition breaks, return-shape changes, and path/constant drift.

Each candidate was independently verified (CONFIRMED / PLAUSIBLE / REFUTED) before inclusion. All 15 findings above are CONFIRMED or PLAUSIBLE with a concrete failure scenario.

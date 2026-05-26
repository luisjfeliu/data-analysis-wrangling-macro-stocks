# Real-World Data Wrangling: Macroeconomic Trends & Stock Market Valuations

This project explores how macroeconomic growth and inflation trends in the United States relate to historical S&P 500 stock market valuation, dividend yields, long-term interest rates, and a gold futures price proxy. It uses two large rubric-qualifying raw datasets, plus supplemental annual World Bank macroeconomic data, to perform exploratory analysis of relationships between macroeconomic health and financial asset valuation/price measures.

## Project Overview

We investigate the relationships between:
1. **Economic Growth**: Annual US GDP growth rate (%).
2. **Inflation**: Annual CPI inflation rate (%), and its relationship to long-term rates and yield proxies.
3. **Equity Valuations**: S&P 500 price level, dividends, corporate earnings, and Shiller's CAPE ratio (PE10).
4. **Interest Rates**: US 10-year Treasury bond yield (`Long Interest Rate`).
5. **Alternative Assets**: Gold futures prices (`GC=F`) from Yahoo Finance, used as a practical proxy for gold exposure rather than a true spot-gold series.

## Data Sources & Gathering Methods

- **S&P 500 Shiller Dataset (CSV)**: Programmatically downloaded from GitHub, containing more than 500 monthly records for equity metrics, CPI, interest rates, and PE10.
- **Gold Futures Price Proxy (API)**: Programmatically retrieved from Yahoo Finance via the `yfinance` library using `GC=F`, containing more than 500 daily market records. `GC=F` is a futures-based Yahoo Finance symbol, so the project describes it as a gold futures/proxy series, not as physical spot gold.
- **Supplemental US Macroeconomic Data (JSON)**: Fetched from the World Bank Indicators API for annual US GDP growth and CPI inflation from 2000 to 2023. The corrected pipeline queries `country/USA` directly instead of `country/all`.

## Important Data Notes

- `data/world_bank_clean.csv` is expected to contain one row per US year.
- The notebook and rebuild script first attempt the programmatic download/API steps, then use the committed raw files as cache fallbacks if a live source is unavailable during review.
- The S&P 500 cleaning step treats `0.0` values in unavailable-data fields as missing placeholders, including `Dividend`, `Earnings`, `Consumer Price Index`, `Long Interest Rate`, `Real Price`, `Real Dividend`, `Real Earnings`, and `PE10`. Rows missing core analytical fields are dropped after this conversion. Annual averages are then computed per column, so unavailable dividend/earnings fields do not cause otherwise valid aligned monthly price, CPI, rate, or PE10 rows to be dropped.
- The validation script enforces full year coverage from 2000-2023, with documented partial-year exceptions: the 2023 S&P source rows contain placeholder values after September, so S&P annual metrics are aligned to the 9 valid months with available price/CPI/rate/PE10 fields, while dividend and earnings averages use the 6 months with available values. The merged dataset labels this explicitly with `sp500_months_used` and `sp500_coverage_label`. Cached S&P rows after 2023 are excluded from the analysis window but also contain unavailable-data placeholders in the same non-price fields. Gold futures data begins partway through 2000, so gold annual-average YoY changes are suppressed for 2000 and 2001.
- `sp500_annual_avg_yoy_change` and `gold_annual_avg_yoy_change` are year-over-year percentage changes in **annual average price levels**, not true calendar-year total returns. Use month-end/year-end values or total-return series if investment return precision is required.
- Correlations use pairwise complete observations. Most variables have 24 annual observations, `sp500_annual_avg_yoy_change` has 23 non-null observations, and `gold_annual_avg_yoy_change` has 22 non-null observations after documented boundary-year suppression.
- Inflation-regime summaries report both total years in each regime and the non-null observation counts used for each YoY average, because S&P and gold YoY series have different boundary-year missing values.
- Seaborn regression plots use a fixed bootstrap seed so the committed figures can be reproduced consistently when the notebook or rebuild script is rerun.
- `cape_yield` is computed as `100 / PE10`, so it is a **CAPE yield proxy**, not a conventional current earnings yield.
- `cape_yield_minus_10y_yield` is a rough **CAPE yield minus 10-year yield spread**, not a full expected equity risk premium model.
- The merged annual dataset contains 24 observations (2000-2023). It stores both `gdp_growth_lag1` and `gdp_growth_lead1`; the lead-lag visualization uses `gdp_growth_lead1`. Treat correlations, regression, and lead-lag conclusions as exploratory rather than causal.

## Repository Structure

```
├── .gitignore                           # Standard git ignore definitions
├── LICENSE                              # MIT License
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies for the notebook and validation script
├── Data_Wrangling_Project_Starter.ipynb # Full restored data-wrangling notebook
├── Data_Wrangling_Project_Starter.html  # Exported HTML version of the prior executed notebook
│
├── scripts/
│   ├── rebuild_outputs.py               # Corrected reproducible pipeline for data, database, and figures
│   └── validate_outputs.py              # Sanity checks for committed cleaned/merged outputs
│
├── data/                                # Data files (raw, cleaned, and database)
│   ├── gold_prices_raw.csv              # Raw daily gold futures/proxy prices
│   ├── sp500_shiller_raw.csv            # Raw monthly S&P 500 historical metrics
│   ├── world_bank_gdp_raw.json          # Raw World Bank API response for GDP growth
│   ├── world_bank_inflation_raw.json    # Raw World Bank API response for inflation
│   ├── gold_prices_clean.csv            # Cleaned daily gold futures/proxy prices
│   ├── sp500_shiller_clean.csv          # Cleaned monthly S&P 500 dataset
│   ├── world_bank_clean.csv             # Cleaned annual US GDP & inflation metrics
│   ├── macro_stock_merged.csv           # Final combined annual dataset
│   └── macro_stock_data.db              # SQLite database storing raw, cleaned, and merged tables
│
└── images/                              # Visualization files (PNGs)
    ├── visual1_pe10_vs_gdp.png          # Shiller PE10 vs. Annual GDP Growth
    ├── visual2_inflation_vs_yields.png  # CPI Inflation vs. Long-Term Interest Rates & CAPE Yield
    ├── visual3_correlation_heatmap.png  # Correlation heatmap across macro and market indicators
    ├── visual4_inflation_regimes.png    # Gold proxy vs. S&P 500 annual-average changes by inflation regime
    ├── visual5_lagged_lead_gdp.png      # Exploratory lagged relationship visualization
    └── visual6_inflation_vs_yields_trends.png # US Inflation Rate vs. S&P 500 Dividend Yield & 10Y Interest Rate (2000-2023)
```

## Key Findings & Visualizations

The project produces several exploratory plots and analyses:

1. **Valuations vs. Growth**: Shiller PE10 levels may be elevated during economic expansions, but the relationship should be interpreted cautiously because valuations and GDP growth are measured at different frequencies and the merged sample is small.
2. **Inflation vs. Yields**: CPI inflation and long-term Treasury yields show a positive but modest relationship in the sample, while CAPE yield and dividend yield measures behave differently.
3. **Gold Proxy vs. Equities**: Inflation-regime analysis compares gold futures/proxy price changes with S&P 500 annual-average price changes. Gold price changes are highest in the moderate-inflation group rather than the high-inflation group, but the group-level averages use the non-null YoY counts documented in the notebook, so this is exploratory and should not be read as a full inflation-hedge test.
4. **Lead-Lag Relationships**: The lagged GDP visualization is suggestive only. A formal cross-correlation, Granger-causality, or out-of-sample forecasting test would be needed before claiming leading-indicator behavior.
5. **Multiple Linear Regression (OLS)**: A programmatic OLS model regressing PE10 against GDP growth, inflation, and interest rates explains about 31.1% of PE10 variance. Because the sample contains only 24 annual observations and residual autocorrelation is present, the coefficients are interpreted as exploratory rather than causal. 
6. **Chronological Time-Series Trends (Visual 6)**: A dual-axis time-series visualization tracing US inflation, stock dividend yields, and long-term interest rates chronologically. It shows that long-term rates rose during the 2021-2023 inflation surge even though the full-period correlation is modest, while S&P 500 dividend yields remained relatively flat or compressed.

## Installation and Requirements

**Requires Python 3.11+**

To run the project locally, ensure you have a Python environment with the required libraries installed:

```bash
# Clone the repository
git clone https://github.com/luisjfeliu/data-analysis-wrangling-macro-stocks.git
cd data-analysis-wrangling-macro-stocks

# Set up virtual environment and install packages
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open the restored full notebook using Jupyter:

```bash
jupyter notebook Data_Wrangling_Project_Starter.ipynb
```

You can also rebuild the committed outputs with the script:

```bash
python scripts/rebuild_outputs.py
```

Run the validation checks:

```bash
python scripts/validate_outputs.py
```

## License

This project is licensed under the [MIT License](LICENSE).

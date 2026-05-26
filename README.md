# Real-World Data Wrangling: Macroeconomic Trends & Stock Market Valuations

This project explores how macroeconomic growth and inflation trends in the United States relate to historical S&P 500 stock market valuation, dividend yields, long-term interest rates, and a gold futures price proxy. By programmatically gathering, cleaning, and merging three distinct datasets, the project performs exploratory analysis of relationships between macroeconomic health and financial asset valuation/price measures.

## Project Overview

We investigate the relationships between:
1. **Economic Growth**: Annual US GDP growth rate (%).
2. **Inflation**: Annual CPI inflation rate (%), and its relationship to long-term rates and yield proxies.
3. **Equity Valuations**: S&P 500 price level, dividends, corporate earnings, and Shiller's CAPE ratio (PE10).
4. **Interest Rates**: US 10-year Treasury bond yield (`Long Interest Rate`).
5. **Alternative Assets**: Gold futures prices (`GC=F`) from Yahoo Finance, used as a practical proxy for gold exposure rather than a true spot-gold series.

## Data Sources & Gathering Methods

- **S&P 500 Shiller Dataset (CSV)**: Programmatically downloaded from GitHub, containing monthly records since 1871 for equity metrics, CPI, interest rates, and PE10.
- **US Macroeconomic Data (JSON)**: Fetched from the World Bank Indicators API for annual US GDP growth and CPI inflation from 2000 to 2023. The corrected pipeline queries `country/USA` directly instead of `country/all`.
- **Gold Futures Price Proxy (API)**: Programmatically retrieved from Yahoo Finance via the `yfinance` library using `GC=F`. `GC=F` is a futures-based Yahoo Finance symbol, so the project describes it as a gold futures/proxy series, not as physical spot gold.

## Important Data Notes

- `data/world_bank_clean.csv` is expected to contain one row per US year.
- `sp500_annual_avg_yoy_change` and `gold_annual_avg_yoy_change` are year-over-year percentage changes in **annual average price levels**, not true calendar-year total returns. Use month-end/year-end values or total-return series if investment return precision is required.
- `cape_yield` is computed as `100 / PE10`, so it is a **CAPE yield proxy**, not a conventional current earnings yield.
- `cape_yield_minus_10y_yield` is a rough **CAPE yield minus 10-year yield spread**, not a full expected equity risk premium model.
- The merged annual dataset contains 24 observations (2000-2023). Treat correlations, regression, and lead-lag conclusions as exploratory rather than causal.

## Repository Structure

```
├── .gitignore                           # Standard git ignore definitions
├── LICENSE                              # MIT License
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies for the notebook and validation script
├── Data_Wrangling_Project_Starter.ipynb # Thin notebook wrapper around the corrected pipeline script
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
│   └── macro_stock_data.db              # SQLite database storing the cleaned tables
│
└── images/                              # Visualization files (PNGs)
    ├── visual1_pe10_vs_gdp.png          # Shiller PE10 vs. Annual GDP Growth
    ├── visual2_inflation_vs_yields.png  # CPI Inflation vs. Long-Term Interest Rates & Dividend Yields
    ├── visual3_correlation_heatmap.png  # Correlation heatmap across macro and market indicators
    ├── visual4_inflation_regimes.png    # Gold proxy vs. S&P 500 annual-average changes by inflation regime
    └── visual5_lagged_lead_gdp.png      # Exploratory lagged relationship visualization
```

## Key Findings & Visualizations

The project produces several exploratory plots:

1. **Valuations vs. Growth**: Shiller PE10 levels may be elevated during economic expansions, but the relationship should be interpreted cautiously because valuations and GDP growth are measured at different frequencies and the merged sample is small.
2. **Inflation vs. Yields**: CPI inflation and long-term Treasury yields show a positive relationship in the sample, while dividend yields track inflation less directly.
3. **Gold Proxy vs. Equities**: Inflation-regime analysis compares gold futures/proxy price changes with S&P 500 annual-average price changes. This is exploratory and should not be read as a full inflation-hedge test.
4. **Lead-Lag Relationships**: The lagged GDP visualization is suggestive only. A formal cross-correlation, Granger-causality, or out-of-sample forecasting test would be needed before claiming leading-indicator behavior.

## Installation and Requirements

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

Run the corrected pipeline:

```bash
python scripts/rebuild_outputs.py
```

Open the notebook wrapper using Jupyter:

```bash
jupyter notebook Data_Wrangling_Project_Starter.ipynb
```

Run the validation checks:

```bash
python scripts/validate_outputs.py
```

## License

This project is licensed under the [MIT License](LICENSE).

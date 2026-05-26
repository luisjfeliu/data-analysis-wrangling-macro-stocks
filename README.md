# Real-World Data Wrangling: Macroeconomic Trends & Stock Market Valuations

This project explores how macroeconomic growth and inflation trends in the United States relate to historical S&P 500 stock market valuation, dividend yields, long-term interest rates, and gold spot prices. By programmatically gathering, cleaning, and merging three distinct datasets, we analyze the correlations and dynamics between macroeconomic health and financial asset returns.

## Project Overview

We investigate the relationships between:
1. **Economic Growth**: Annual US GDP Growth rate (%).
2. **Inflation**: Annual CPI inflation rate (%), and its impact on risk-free rates and equity yields.
3. **Equity Valuations**: S&P 500 close price, dividend yield, corporate earnings, and Shiller's CAPE ratio (PE10).
4. **Interest Rates**: US 10-year Treasury bond yield (Long Interest Rate).
5. **Alternative Assets**: Gold spot prices (GC=F) as a traditional safe-haven hedge against inflation.

## Data Sources & Gathering Methods

- **S&P 500 Shiller Dataset (CSV)**: Programmatically downloaded from GitHub, containing monthly records since 1871 for equity metrics, CPI, interest rates, and PE10.
- **US Macroeconomic Data (JSON)**: Fetched directly from the **World Bank Indicators API** for annual GDP growth and CPI inflation from 2000 to 2023.
- **Gold Spot Prices (API)**: Programmatically retrieved from Yahoo Finance via the `yfinance` library, providing daily spot prices from 2000 to 2023.

## Repository Structure

```
├── .gitignore                          # Standard git ignore definitions
├── LICENSE                             # MIT License
├── README.md                           # Project documentation (this file)
├── Data_Wrangling_Project_Starter.ipynb # Main Jupyter Notebook containing all wrangling steps
├── Data_Wrangling_Project_Starter.html  # Exported HTML version of the executed notebook
│
├── data/                               # Data files (raw, cleaned, and database)
│   ├── gold_prices_raw.csv             # Raw daily gold spot prices
│   ├── sp500_shiller_raw.csv           # Raw monthly S&P 500 historical metrics
│   ├── world_bank_gdp_raw.json         # Raw World Bank API response for GDP growth
│   ├── world_bank_inflation_raw.json   # Raw World Bank API response for inflation
│   ├── gold_prices_clean.csv           # Cleaned daily/monthly gold spot prices
│   ├── sp500_shiller_clean.csv         # Cleaned monthly S&P 500 dataset
│   ├── world_bank_clean.csv            # Cleaned annual US GDP & inflation metrics
│   ├── macro_stock_merged.csv          # Final combined dataset
│   └── macro_stock_data.db             # SQLite database storing the cleaned tables
│
└── images/                             # Visualization files (PNGs)
    ├── visual1_pe10_vs_gdp.png         # Shiller PE10 vs. Annual GDP Growth
    ├── visual2_inflation_vs_yields.png # CPI Inflation vs. Long-Term Interest Rates & Dividend Yields
    ├── visual3_correlation_heatmap.png # Correlation heatmap across macro and market indicators
    ├── visual4_inflation_regimes.png   # Gold vs. S&P 500 returns across inflation regimes
    └── visual5_lagged_lead_gdp.png     # Lead-lag correlation analysis of GDP growth and PE10
```

## Key Findings & Visualizations

The Jupyter Notebook produces several analytical plots demonstrating key findings:
1. **Valuations vs. Growth**: Shiller PE10 levels tend to peak during sustained GDP expansion but can lead or lag actual growth inflections.
2. **Inflation vs. Yields**: A strong positive correlation exists between CPI inflation and 10-year Treasury yields, while dividend yields track inflation less tightly.
3. **Safe-Havens**: Analysis of inflation regimes shows gold outperforming equities during high-inflation periods, demonstrating its traditional role as a hedge.
4. **Lead-Lag Relationships**: Cross-correlation shows that equity valuations (PE10) can sometimes act as a leading indicator of macroeconomic growth inflections.

## Installation and Requirements

To run the project notebook locally, ensure you have a Python environment with the required libraries installed:

```bash
# Clone the repository
git clone <repository-url>
cd data-analysis-wrangling

# Set up virtual environment and install packages
python -m venv .venv
source .venv/bin/activate
pip install pandas requests matplotlib seaborn yfinance
```

Open the notebook using Jupyter:
```bash
jupyter notebook Data_Wrangling_Project_Starter.ipynb
```

## License

This project is licensed under the [MIT License](LICENSE).

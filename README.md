# Turtle Breakout Strategy Backtester

Interactive Streamlit app that reproduces the Turtle-style breakout system using daily Yahoo Finance data. The app compares:

- Turtle breakout strategy trading the trigger ticker directly
- Turtle breakout strategy using trigger ticker signals but executing on a second ticker (e.g., SPY triggers, UPRO trades)
- Buy & hold baselines for both instruments

## Quickstart

```bash
pip install streamlit yfinance pandas numpy plotly scipy
streamlit run streamlit_app.py
```

The UI lets you adjust dates, tickers, breakout/exit windows, ATR period, annual risk-free rate, transaction costs, and long/short behaviour. Results include normalized equity curves, drawdown plots, performance statistics (CAGR, volatility, skewness, kurtosis, Sharpe, Sortino, max drawdown), and downloadable CSV outputs.

## Creating the GitHub Repository

1. Create a new repository on GitHub (no README, .gitignore, or license).
2. From this project directory, run:
   ```bash
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git add .
   git commit -m "Initial commit: Turtle breakout Streamlit backtester"
   git push -u origin main
   ```

Replace `<your-username>` and `<repo-name>` with your GitHub details.


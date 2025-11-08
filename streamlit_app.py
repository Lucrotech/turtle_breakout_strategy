# Instructions
# ------------
# 1. Optional: create and activate a virtual environment.
# 2. Install requirements:
#       pip install streamlit yfinance pandas numpy plotly scipy
# 3. Run the app:
#       streamlit run streamlit_app.py
#
# The application backtests a Turtle-style breakout strategy on Yahoo Finance
# daily data using Streamlit for the UI and Plotly for visualization.

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from scipy.stats import skew, kurtosis


TRADING_DAYS_PER_YEAR = 252


@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker from Yahoo Finance."""
    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        actions=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.xs(ticker, axis=1, level=-1)
        except KeyError:
            # Fallback: use the first level if ticker slicing fails.
            data.columns = data.columns.get_level_values(0)

    if data.empty:
        raise ValueError(f"No price data returned for ticker {ticker}.")

    required_cols = {"Adj Close", "High", "Low", "Close"}
    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(f"{ticker} missing expected columns: {missing}")

    data = data[list(required_cols)]
    data.sort_index(inplace=True)
    return data


def calculate_atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Compute the Average True Range (ATR)."""
    high_low = data["High"] - data["Low"]
    high_close_prev = (data["High"] - data["Close"].shift(1)).abs()
    low_close_prev = (data["Low"] - data["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()
    return atr


@dataclass
class StrategyResult:
    returns: pd.Series
    equity_curve: pd.Series
    drawdown: pd.Series
    positions: pd.Series
    trades: pd.Series
    atr: pd.Series


def compute_strategy_positions(
    trigger_data: pd.DataFrame,
    entry_window: int,
    exit_window: int,
    atr_period: int,
    include_shorts: bool,
) -> Tuple[pd.Series, pd.Series]:
    """Generate daily positions (+1, -1, 0) based on Turtle breakout rules."""
    df = trigger_data.copy()

    entry_high = df["High"].rolling(window=entry_window, min_periods=entry_window).max().shift(1)
    entry_low = df["Low"].rolling(window=entry_window, min_periods=entry_window).min().shift(1)
    exit_low = df["Low"].rolling(window=exit_window, min_periods=exit_window).min().shift(1)
    exit_high = df["High"].rolling(window=exit_window, min_periods=exit_window).max().shift(1)

    atr = calculate_atr(df, atr_period)

    positions = []
    position = 0

    for idx, row in df.iterrows():
        long_entry = row["Close"] > entry_high.loc[idx] if not pd.isna(entry_high.loc[idx]) else False
        short_entry = row["Close"] < entry_low.loc[idx] if not pd.isna(entry_low.loc[idx]) else False
        long_exit = row["Close"] < exit_low.loc[idx] if not pd.isna(exit_low.loc[idx]) else False
        short_exit = row["Close"] > exit_high.loc[idx] if not pd.isna(exit_high.loc[idx]) else False

        if position == 1 and long_exit:
            position = 0
        elif position == -1 and short_exit:
            position = 0

        if position == 0:
            if long_entry:
                position = 1
            elif include_shorts and short_entry:
                position = -1

        positions.append(position)

    position_series = pd.Series(positions, index=df.index, name="position")
    return position_series, atr


def apply_transaction_costs(position_series: pd.Series, cost_pct: float) -> pd.Series:
    """Compute per-day transaction cost adjustments based on position changes."""
    if cost_pct <= 0:
        return pd.Series(0.0, index=position_series.index)

    position_change = position_series.diff().fillna(position_series)
    per_leg_cost = (cost_pct / 100.0) / 2.0
    trade_cost = position_change.abs() * per_leg_cost
    return trade_cost


def run_turtle_strategy(
    trigger_data: pd.DataFrame,
    execution_data: pd.DataFrame,
    entry_window: int,
    exit_window: int,
    atr_period: int,
    include_shorts: bool,
    trade_cost_pct: float,
) -> StrategyResult:
    """Run the Turtle breakout strategy and return performance components."""
    positions, atr = compute_strategy_positions(
        trigger_data=trigger_data,
        entry_window=entry_window,
        exit_window=exit_window,
        atr_period=atr_period,
        include_shorts=include_shorts,
    )

    exec_prices = execution_data["Adj Close"].copy()
    returns = exec_prices.pct_change().fillna(0.0)

    aligned_positions = positions.reindex(exec_prices.index).ffill().fillna(0)
    # Zero out positions when execution data is missing.
    valid_prices = exec_prices.notna()
    aligned_positions = aligned_positions.where(valid_prices, 0)

    effective_positions = aligned_positions.shift(1).fillna(0)

    raw_returns = effective_positions * returns

    trade_costs = apply_transaction_costs(effective_positions, trade_cost_pct)
    returns_after_costs = raw_returns - trade_costs

    equity_curve = (1 + returns_after_costs).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0

    trades = aligned_positions.diff().fillna(aligned_positions.abs())

    atr_aligned = atr.reindex(exec_prices.index)

    return StrategyResult(
        returns=returns_after_costs,
        equity_curve=equity_curve,
        drawdown=drawdown,
        positions=aligned_positions,
        trades=trades,
        atr=atr_aligned,
    )


def buy_and_hold(data: pd.DataFrame) -> StrategyResult:
    """Compute buy-and-hold performance for reference."""
    exec_prices = data["Adj Close"]
    returns = exec_prices.pct_change().fillna(0.0)
    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    zero_series = pd.Series(0, index=data.index)

    return StrategyResult(
        returns=returns,
        equity_curve=equity_curve,
        drawdown=drawdown,
        positions=zero_series,
        trades=zero_series,
        atr=pd.Series(np.nan, index=data.index),
    )


def compute_cagr(equity_curve: pd.Series) -> float:
    """Compound annual growth rate."""
    if equity_curve.empty or equity_curve.iloc[0] <= 0:
        return np.nan
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    num_periods = len(equity_curve)
    if num_periods <= 1 or total_return <= 0:
        return np.nan
    return total_return ** (TRADING_DAYS_PER_YEAR / (num_periods - 1)) - 1


def compute_annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of returns."""
    daily_std = returns.dropna().std(ddof=0)
    return daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)


def compute_skewness(returns: pd.Series) -> float:
    """Skewness of daily returns."""
    valid_returns = returns.dropna()
    if valid_returns.empty:
        return np.nan
    return skew(valid_returns, bias=False)


def compute_kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis of daily returns."""
    valid_returns = returns.dropna()
    if valid_returns.empty:
        return np.nan
    return kurtosis(valid_returns, bias=False, fisher=True)


def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """Annualized Sharpe ratio (excess returns over risk-free rate)."""
    valid_returns = returns.dropna()
    if valid_returns.empty:
        return np.nan

    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess_returns = valid_returns - rf_daily
    mean_excess = excess_returns.mean() * TRADING_DAYS_PER_YEAR
    vol = valid_returns.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)

    if vol == 0:
        return np.nan
    return mean_excess / vol


def compute_sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """Annualized Sortino ratio using downside deviation."""
    valid_returns = returns.dropna()
    if valid_returns.empty:
        return np.nan

    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess_returns = valid_returns - rf_daily
    mean_excess = excess_returns.mean() * TRADING_DAYS_PER_YEAR

    downside_diff = np.minimum(excess_returns, 0)
    downside_variance = (downside_diff ** 2).mean()
    downside_dev = math.sqrt(downside_variance) * math.sqrt(TRADING_DAYS_PER_YEAR)

    if downside_dev == 0:
        return np.nan
    return mean_excess / downside_dev


def compute_max_drawdown(drawdown: pd.Series) -> float:
    """Maximum observed drawdown (as a negative number)."""
    if drawdown.empty:
        return np.nan
    return drawdown.min()


def summarise_metrics(result: StrategyResult, risk_free_rate: float) -> Dict[str, float]:
    """Aggregate all performance metrics for a strategy."""
    metrics = {
        "CAGR": compute_cagr(result.equity_curve),
        "Annualized Volatility": compute_annualized_volatility(result.returns),
        "Skewness": compute_skewness(result.returns),
        "Kurtosis": compute_kurtosis(result.returns),
        "Sharpe Ratio": compute_sharpe_ratio(result.returns, risk_free_rate),
        "Sortino Ratio": compute_sortino_ratio(result.returns, risk_free_rate),
        "Max Drawdown": compute_max_drawdown(result.drawdown),
    }
    return metrics


def build_equity_plot(equity_series: Dict[str, pd.Series]) -> go.Figure:
    """Plot normalized equity curves."""
    fig = go.Figure()
    for label, series in equity_series.items():
        if series.empty:
            continue
        normalized = series / series.iloc[0]
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized, mode="lines", name=label))

    fig.update_layout(
        title="Normalized Equity Curves",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        hovermode="x unified",
        legend_title="Strategy",
    )
    return fig


def build_drawdown_plot(drawdown_series: Dict[str, pd.Series]) -> go.Figure:
    """Plot drawdown curves and annotate maximum drawdowns."""
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    for label, series in drawdown_series.items():
        if series.empty:
            continue
        fig.add_trace(
            go.Scatter(x=series.index, y=series, mode="lines", name=label),
            row=1,
            col=1,
        )
        min_value = series.min()
        min_date = series.idxmin()
        if pd.notna(min_value) and pd.notna(min_date):
            fig.add_annotation(
                x=min_date,
                y=min_value,
                text=f"{label} min: {min_value:.2%}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
            )

    fig.update_layout(
        title="Drawdown (%)",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        hovermode="x unified",
        legend_title="Strategy",
    )

    return fig


def prepare_download_df(
    equity: Dict[str, pd.Series], drawdowns: Dict[str, pd.Series], returns: Dict[str, pd.Series]
) -> pd.DataFrame:
    """Combine equity, drawdown, and return series for CSV download."""
    combined = pd.DataFrame(index=next(iter(equity.values())).index)
    for label, series in equity.items():
        combined[f"{label}_equity"] = series
    for label, series in drawdowns.items():
        combined[f"{label}_drawdown"] = series
    for label, series in returns.items():
        combined[f"{label}_returns"] = series
    return combined


def main() -> None:
    st.set_page_config(page_title="Turtle Breakout Strategy Backtester", layout="wide")
    st.title("Turtle Breakout Strategy Backtester")
    st.markdown(
        "Backtest the classic Turtle-style breakout system using Yahoo Finance daily data."
    )

    default_start = pd.to_datetime("2000-01-01")
    default_end = pd.to_datetime("2025-10-31")

    with st.sidebar:
        st.header("Configuration")
        start_date, end_date = st.date_input(
            "Date Range",
            value=(default_start.date(), default_end.date()),
            min_value=pd.to_datetime("1990-01-01").date(),
            max_value=pd.Timestamp.today().date(),
        )

        trigger_ticker = st.text_input("Trigger Ticker", value="SPY").strip().upper()
        execution_ticker = st.text_input("Execution Ticker", value="UPRO").strip().upper()

        entry_window = st.number_input("Entry Window (days)", min_value=2, max_value=252, value=20, step=1)
        exit_window = st.number_input("Exit Window (days)", min_value=1, max_value=252, value=10, step=1)
        atr_period = st.number_input("ATR Period (days)", min_value=2, max_value=252, value=14, step=1)

        risk_free_rate = st.number_input("Annual Risk-Free Rate (decimal)", value=0.0, step=0.01, format="%.4f")
        trade_cost_pct = st.number_input(
            "Per-trade roundtrip cost (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.05, format="%.2f"
        )

        include_shorts = st.checkbox("Allow Short Positions", value=True)
        run_button = st.button("Run Backtest", type="primary")

    if not run_button:
        st.info("Set your parameters and click **Run Backtest** to begin.")
        return

    progress = st.progress(0)
    status_text = st.empty()

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    try:
        status_text.text(f"Fetching data for {trigger_ticker}...")
        progress.progress(10)
        trigger_data = load_price_data(trigger_ticker, start_ts, end_ts)

        status_text.text(f"Fetching data for {execution_ticker}...")
        progress.progress(25)
        execution_data = load_price_data(execution_ticker, start_ts, end_ts)

        status_text.text("Fetching SPY and UPRO data for buy & hold benchmarks...")
        progress.progress(40)
        spy_data = load_price_data("SPY", start_ts, end_ts)
        upro_data = load_price_data("UPRO", start_ts, end_ts)

        status_text.text("Running strategy on trigger & execution ticker...")
        progress.progress(60)
        strategy_same = run_turtle_strategy(
            trigger_data=trigger_data,
            execution_data=trigger_data,
            entry_window=entry_window,
            exit_window=exit_window,
            atr_period=atr_period,
            include_shorts=include_shorts,
            trade_cost_pct=trade_cost_pct,
        )

        status_text.text(f"Running strategy: {trigger_ticker} signals on {execution_ticker} returns...")
        progress.progress(75)
        strategy_cross = run_turtle_strategy(
            trigger_data=trigger_data,
            execution_data=execution_data,
            entry_window=entry_window,
            exit_window=exit_window,
            atr_period=atr_period,
            include_shorts=include_shorts,
            trade_cost_pct=trade_cost_pct,
        )

        status_text.text("Calculating buy & hold benchmarks...")
        progress.progress(85)
        buy_hold_spy = buy_and_hold(spy_data)
        buy_hold_upro = buy_and_hold(upro_data)

        status_text.text("Compiling results and visuals...")
        progress.progress(95)

        metrics = {
            f"Turtle {trigger_ticker}": summarise_metrics(strategy_same, risk_free_rate),
            f"Turtle {trigger_ticker}->{execution_ticker}": summarise_metrics(strategy_cross, risk_free_rate),
            "Buy & Hold SPY": summarise_metrics(buy_hold_spy, risk_free_rate),
            "Buy & Hold UPRO": summarise_metrics(buy_hold_upro, risk_free_rate),
        }
        metrics_df = pd.DataFrame(metrics).T

        equity_curves = {
            f"Turtle {trigger_ticker}": strategy_same.equity_curve,
            f"Turtle {trigger_ticker}->{execution_ticker}": strategy_cross.equity_curve,
            "Buy & Hold SPY": buy_hold_spy.equity_curve,
            "Buy & Hold UPRO": buy_hold_upro.equity_curve,
        }
        drawdowns = {
            f"Turtle {trigger_ticker}": strategy_same.drawdown,
            f"Turtle {trigger_ticker}->{execution_ticker}": strategy_cross.drawdown,
            "Buy & Hold SPY": buy_hold_spy.drawdown,
            "Buy & Hold UPRO": buy_hold_upro.drawdown,
        }
        return_series = {
            f"Turtle {trigger_ticker}": strategy_same.returns,
            f"Turtle {trigger_ticker}->{execution_ticker}": strategy_cross.returns,
            "Buy & Hold SPY": buy_hold_spy.returns,
            "Buy & Hold UPRO": buy_hold_upro.returns,
        }

        equity_fig = build_equity_plot(equity_curves)
        drawdown_fig = build_drawdown_plot(drawdowns)

        download_df = prepare_download_df(equity_curves, drawdowns, return_series)
        csv_bytes = download_df.to_csv().encode("utf-8")

        progress.progress(100)
        status_text.success("Backtest complete.")

    except Exception as exc:
        progress.empty()
        status_text.error(f"Error: {exc}")
        st.stop()

    st.plotly_chart(equity_fig, use_container_width=True)
    st.plotly_chart(drawdown_fig, use_container_width=True)

    st.subheader("Performance Metrics")
    st.dataframe(metrics_df.style.format({
        "CAGR": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Skewness": "{:.2f}",
        "Kurtosis": "{:.2f}",
        "Sharpe Ratio": "{:.2f}",
        "Sortino Ratio": "{:.2f}",
        "Max Drawdown": "{:.2%}",
    }))

    st.download_button(
        label="Download time series CSV",
        data=csv_bytes,
        file_name="turtle_strategy_timeseries.csv",
        mime="text/csv",
    )

    st.caption(
        "Strategy trades are based on prior-day breakout signals per the Turtle methodology. "
        "Returns are computed with daily frequency using adjusted close prices."
    )


if __name__ == "__main__":
    main()



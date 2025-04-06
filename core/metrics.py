import numpy as np
import pandas as pd


def calculate_total_return(price_data: pd.DataFrame, column: str) -> float:
    """Calculate total return"""

    non_zero = price_data[column].loc[price_data[column] > 0]
    if non_zero.empty:
        return np.nan
    return (non_zero.iloc[-1] / non_zero.iloc[0]) - 1


def calculate_sharpe_ratio(price_data: pd.DataFrame, column: str, periods_per_year: int = 252) -> float:
    """Calculate Sharpe ratio (assuming daily returns)"""

    daily_returns = price_data[column].pct_change().dropna()
    if daily_returns.std() == 0:
        return np.nan
    return daily_returns.mean() / daily_returns.std() * np.sqrt(periods_per_year)


def calculate_max_drawdown(price_data: pd.DataFrame, column: str) -> float:
    """Calculate max drawdown"""

    rolling_max = price_data[column].cummax()
    drawdown = (price_data[column] - rolling_max) / rolling_max
    return drawdown.min()


def calculate_win_rate(price_data: pd.DataFrame, column: str) -> float:
    """Calculate win rate (percentage of positive returns)"""

    daily_returns = price_data[column].pct_change().dropna()
    wins = daily_returns[daily_returns > 0].count()
    total_trades = daily_returns.count()
    return wins / total_trades if total_trades != 0 else np.nan


def calculate_expectancy(price_data: pd.DataFrame, column: str) -> float:
    """Calculate expectancy (expected profit per trade)"""

    daily_returns = price_data[column].pct_change().dropna()
    win_rate = calculate_win_rate(price_data, column)
    avg_profit = daily_returns.mean()
    avg_loss = daily_returns[daily_returns < 0].mean()
    return (win_rate * avg_profit) - ((1 - win_rate) * abs(avg_loss))


def calculate_exposure_time(price_data: pd.DataFrame, position_column: str) -> float:
    """Calculate exposure time (percentage of time in the market)"""

    in_market = price_data[position_column].notna() & (price_data[position_column] != 0)
    return in_market.sum() / len(price_data) * 100
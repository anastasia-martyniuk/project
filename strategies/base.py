from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class StrategyBase(ABC):
    def __init__(self, price_data: pd.DataFrame):
        """
        Initialize the strategy with price data.

        :param price_data: historical price data for backtesting, represented as a pd.DataFrame.
        """
        self.price_data = price_data

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on the strategy logic.
        This method should implement the logic for generating buy, sell, or hold signals
        based on the price data.

        :return: A pd.DataFrame containing the generated signals and relevant information.
        """
        pass

    @abstractmethod
    def run_backtest(self) -> pd.DataFrame:
        """
        Run the backtest of the strategy using the price data.
        This method should execute the trading strategy, apply risk management,
        and simulate trading results over the historical data.

        :return: A pd.DataFrame with backtest results, such as portfolio_value, positions.
        """
        pass

    def calculate_total_return(self, column: str) -> float:
        """Calculate total return"""

        non_zero = self.price_data[column].loc[self.price_data[column] > 0]
        if non_zero.empty:
            return np.nan
        return (non_zero.iloc[-1] / non_zero.iloc[0]) - 1

    def calculate_sharpe_ratio(self, column: str, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio (assuming daily returns)"""

        daily_returns = self.price_data[column].pct_change().dropna()
        if daily_returns.std() == 0:
            return np.nan
        return daily_returns.mean() / daily_returns.std() * np.sqrt(periods_per_year)

    def calculate_max_drawdown(self, column: str) -> float:
        """Calculate max drawdown"""

        rolling_max = self.price_data[column].cummax()
        drawdown = (self.price_data[column] - rolling_max) / rolling_max
        return drawdown.min()

    def calculate_win_rate(self, column: str) -> float:
        """Calculate win rate (percentage of positive returns)"""

        daily_returns = self.price_data[column].pct_change().dropna()
        wins = daily_returns[daily_returns > 0].count()
        total_trades = daily_returns.count()
        return wins / total_trades if total_trades != 0 else np.nan

    def calculate_expectancy(self, column: str) -> float:
        """Calculate expectancy (expected profit per trade)"""

        daily_returns = self.price_data[column].pct_change().dropna()
        win_rate = self.calculate_win_rate(column)
        avg_profit = daily_returns.mean()
        avg_loss = daily_returns[daily_returns < 0].mean()
        return (win_rate * avg_profit) - ((1 - win_rate) * abs(avg_loss))

    def calculate_exposure_time(self, position_column: str) -> float:
        """Calculate exposure time (percentage of time in the market)"""

        in_market = self.price_data[position_column].notna() & (self.price_data[position_column] != 0)
        return in_market.sum() / len(self.price_data) * 100

    @abstractmethod
    def get_metrics(self, column: str, position_column: str) -> dict:
        """
        Calculate and return key performance metrics for the strategy.
        This method should return important metrics like:
        total_return, Sharpe ratio, max drawdown, win rate, expectancy, exposure_time.

        :return: A dictionary containing calculated metrics.
        """
        return {
            "total_return": self.calculate_total_return(column),
            "sharpe_ratio": self.calculate_sharpe_ratio(column),
            "max_drawdown": self.calculate_max_drawdown(column),
            "win_rate": self.calculate_win_rate(column),
            "expectancy": self.calculate_expectancy(column),
            "exposure_time": self.calculate_exposure_time(column)
        }

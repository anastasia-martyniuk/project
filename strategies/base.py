from abc import ABC, abstractmethod

import pandas as pd

from core.metrics import (
    calculate_total_return, calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate, calculate_expectancy,
    calculate_exposure_time
)


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

    @abstractmethod
    def get_metrics(self, column: str, position_column: str) -> dict:
        """
        Calculate and return key performance metrics for the strategy.
        This method should return important metrics like:
        total_return, Sharpe ratio, max drawdown, win rate, expectancy, exposure_time.

        :return: A dictionary containing calculated metrics.
        """
        return {
            "total_return": calculate_total_return(self.price_data, column),
            "sharpe_ratio": calculate_sharpe_ratio(self.price_data, column),
            "max_drawdown": calculate_max_drawdown(self.price_data, column),
            "win_rate": calculate_win_rate(self.price_data, column),
            "expectancy": calculate_expectancy(self.price_data, column),
            "exposure_time": calculate_exposure_time(self.price_data, column)
        }

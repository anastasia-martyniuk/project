import pandas as pd
import numpy as np

from strategies.base import StrategyBase
from core.data_loader import get_liquid_pairs, fetch_binance_data


class SMACrossover(StrategyBase):
    def __init__(self, price_data: pd.DataFrame, short_window: int = 10, long_window: int = 50):
        super().__init__(price_data)
        self.price_data = price_data
        self.short_window = short_window
        self.long_window = long_window
        self.initial_capital = 10000
        self.commission = 0.001
        self.slippage = 0.002
        self.time_drift_delay = 1

    def generate_signals(self) -> pd.DataFrame:
        self.price_data['SMA_short'] = self.price_data['close'].rolling(window=self.short_window).mean()
        self.price_data['SMA_long'] = self.price_data['close'].rolling(window=self.long_window).mean()

        # Signal generation: 1 buy, -1 sell, 0 hold
        self.price_data['sma_signal'] = np.where(
            self.price_data['SMA_short'] > self.price_data['SMA_long'], 1, np.where(
                self.price_data['SMA_short'] < self.price_data['SMA_long'], -1, 0
            )
        )

        # Position: when to buy or sell
        self.price_data['sma_position'] = self.price_data['sma_signal'].diff().shift(-1)

        return self.price_data

    def run_backtest(self) -> pd.DataFrame:
        self.generate_signals()

        # Apply time drift
        self.price_data['shifted_close'] = self.price_data['close'].shift(self.time_drift_delay)

        # Apply slippage
        slippage_adjusted_price = self.price_data['shifted_close'] * (1 + self.slippage)

        # Calculate portfolio value considering commission and slippage
        self.price_data['sma_portfolio_value'] = (
            self.initial_capital + (self.price_data['sma_position'].shift(1) * slippage_adjusted_price.diff()).cumsum()
        )

        # Apply commission
        self.price_data['sma_portfolio_value'] *= (1 - self.commission)

        return self.price_data[['timestamp', 'sma_portfolio_value']]

    def get_metrics(self, column: str, position_column: str) -> dict:
        self.run_backtest()

        return super().get_metrics(column=column, position_column=position_column)


if __name__ == "__main__":
    liquid_pairs = get_liquid_pairs()
    for liquid_pair in liquid_pairs:
        print(f"Start working with {liquid_pair}")
        print(
            SMACrossover(
                price_data=fetch_binance_data(pair=liquid_pair)).get_metrics(
                column='sma_portfolio_value', position_column='sma_position'
            )
        )

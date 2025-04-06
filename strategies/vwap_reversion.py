import pandas as pd
import numpy as np

from strategies.base import StrategyBase
from core.data_loader import get_liquid_pairs, fetch_binance_data


class VWAPReversion(StrategyBase):
    def __init__(self, price_data: pd.DataFrame):
        super().__init__(price_data)
        self.price_data = price_data
        self.threshold = 0.02
        self.capital = 10000
        self.commission = 0.001
        self.slippage = 0.002
        self.time_drift_delay = 1

    def generate_signals(self) -> pd.DataFrame:
        self.price_data['vwap'] = (
            (self.price_data['close'] * self.price_data['volume']).cumsum() / self.price_data['volume'].cumsum()
        )

        # Determine levels for opening and closing positions
        self.price_data['vwap_buy_signal'] = np.where(
            self.price_data['close'] < self.price_data['vwap'] * (1 - self.threshold), 1, 0
        )
        self.price_data['vwap_sell_signal'] = np.where(
            self.price_data['close'] > self.price_data['vwap'] * (1 + self.threshold), -1, 0
        )

        # Signal: buy or sell
        self.price_data['vwap_position'] = self.price_data['vwap_buy_signal'] + self.price_data['vwap_sell_signal']

        return self.price_data

    def run_backtest(self) -> pd.DataFrame:
        self.generate_signals()

        # Position shifted by 1 to account for time offset
        self.price_data['shifted_close'] = self.price_data['close'].shift(self.time_drift_delay)

        # Position with slippage
        slippage_adjusted_price = self.price_data['shifted_close'] * (1 + self.slippage)

        # Calculation of portfolio value taking into account commission and slippage
        self.price_data['vwap_portfolio_value'] = (
            self.capital + (self.price_data['vwap_position'].shift(1) * slippage_adjusted_price.diff()).cumsum()
        )
        # Apply a commission
        self.price_data['vwap_portfolio_value'] *= (1 - self.commission)

        return self.price_data[['timestamp', 'vwap_portfolio_value']]

    def get_metrics(self, column: str, position_column: str) -> dict:
        self.run_backtest()

        return super().get_metrics(column=column, position_column=position_column)

if __name__ == "__main__":
    liquid_pairs = get_liquid_pairs()
    for liquid_pair in liquid_pairs:
        print(f"Start working with {liquid_pair}")
        print(
            VWAPReversion(
                price_data=fetch_binance_data(pair=liquid_pair)).get_metrics(
                column='vwap_portfolio_value', position_column='vwap_position'
            )
        )

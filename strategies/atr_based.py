import pandas as pd
import vectorbt as vbt

from strategies.base import StrategyBase


class ATRBasedTrailingBreakout(StrategyBase):
    def __init__(self, price_data: pd.DataFrame):
        super().__init__(price_data)
        self.price_data = price_data
        self.multiplier = 1.5

    def generate_signals(self) -> pd.DataFrame:
        # high, low and close for all pairs
        high_columns = self.price_data.filter(like="high").copy()
        low_columns  = self.price_data.filter(like="low").copy()
        close_columns = self.price_data.filter(like="close").copy()

        # Calculating ATR
        atr = vbt.IndicatorFactory.from_pandas_ta('atr').run(high=high_columns, low=low_columns, close=close_columns)

        # Calculating breakout level
        breakout_level = close_columns.shift(1) + self.multiplier * atr

        # Buy
        entries = close_columns > breakout_level

        # Sell
        exits = close_columns < breakout_level

        return entries, exits
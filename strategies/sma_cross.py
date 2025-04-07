import pandas as pd

from strategies.base import StrategyBase


class SMACrossover(StrategyBase):
    def __init__(self, price_data: pd.DataFrame, short_window: int = 10, long_window: int = 50):
        super().__init__(price_data)
        self.price_data = price_data
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self) -> pd.DataFrame:
        # close_columns = self.price_data.filter(like='close').copy()
        # calculating SMA
        short_sma = self.price_data.rolling(window=self.short_window).mean()
        long_sma = self.price_data.rolling(window=self.long_window).mean()

        # entries: signal to buy
        entries = (short_sma > long_sma) & (short_sma.shift() <= long_sma.shift())

        # exits: signal to sell
        exits = (short_sma < long_sma) & (short_sma.shift() >= long_sma.shift())

        return entries, exits

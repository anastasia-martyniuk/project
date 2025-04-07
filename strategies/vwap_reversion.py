import pandas as pd
from strategies.base import StrategyBase


class VWAPReversionIntraday(StrategyBase):
    def __init__(self, price_data: pd.DataFrame):
        super().__init__(price_data)
        self.price_data = price_data
        self.vwap_window = 14
        self.threshold = 0.02

    def generate_signals(self) -> (pd.DataFrame, pd.DataFrame):
        entries = pd.DataFrame(index=self.price_data.index)
        exits = pd.DataFrame(index=self.price_data.index)

        for symbol in self.price_data.columns:
            if 'close' in symbol and 'volume' in symbol:
                close_col = f'close {symbol.split(" ")[1]}'
                volume_col = f'volume {symbol.split(" ")[1]}'

                vwap = self.price_data[[close_col, volume_col]].rolling(window=self.vwap_window).apply(
                    lambda x: (x[close_col] * x[volume_col]).sum() / x[volume_col].sum(),
                    raw=False
                )

                deviation = (self.price_data[close_col] - vwap) / vwap
                entries[symbol] = deviation < -self.threshold
                exits[symbol] = deviation > self.threshold

        entries = entries.dropna(axis=1, how='all')
        exits = exits.dropna(axis=1, how='all')
        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits

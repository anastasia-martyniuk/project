import pandas as pd
from strategies.base import StrategyBase
from core.data_loader import get_liquid_pairs, fetch_binance_data



class ATRBasedTrailingBreakout(StrategyBase):
    def __init__(
        self, price_data: pd.DataFrame, atr_period: int = 14, breakout_window: int = 20, atr_multiplier: float = 3.0
    ):
        super().__init__(price_data)
        self.atr_period = atr_period
        self.breakout_window = breakout_window
        self.atr_multiplier = atr_multiplier
        self.initial_capital = 10000
        self.slippage = 0.001
        self.commission = 0.001

    def calculate_atr(self) -> pd.Series:
        high, low, close = self.price_data['high'], self.price_data['low'], self.price_data['close']
        prev_close = close.shift(1)
        atr = pd.concat(
            [high - low, abs(high - prev_close), abs(low - prev_close)], axis=1
        ).max(axis=1).rolling(window=self.atr_period).mean()
        return atr

    def generate_signals(self) -> pd.DataFrame:
        # Calculate ATR
        high, low, close = self.price_data['high'], self.price_data['low'], self.price_data['close']
        prev_close = close.shift(1)

        # Add atr column
        self.price_data['atr'] = pd.concat(
            [high - low, abs(high - prev_close), abs(low - prev_close)], axis=1
        ).max(axis=1).rolling(window=self.atr_period).mean()

        # Breakout of local high
        self.price_data['breakout_entry'] = (
            self.price_data['close'] > self.price_data['high'].shift(1).rolling(self.breakout_window).max()
        )

        # Initialize position column
        self.price_data['atr_position'] = 0
        in_position, trailing_stop = False, None

        # Signal: buy or sell
        for i in range(len(self.price_data)):
            if not in_position:
                if self.price_data['breakout_entry'].iloc[i]:
                    in_position = True
                    entry_price = self.price_data['close'].iloc[i]
                    trailing_stop = entry_price - self.atr_multiplier * self.price_data['atr'].iloc[i]
                    self.price_data.at[self.price_data.index[i], 'atr_position'] = 1
            else:
                current_price = self.price_data['close'].iloc[i]
                trailing_stop = max(trailing_stop, current_price - self.atr_multiplier * self.price_data['atr'].iloc[i])
                if current_price < trailing_stop:
                    in_position = False
                    self.price_data.at[self.price_data.index[i], 'atr_position'] = -1
                else:
                    self.price_data.at[self.price_data.index[i], 'atr_position'] = 1

        return self.price_data

    def run_backtest(self) -> pd.DataFrame:
        self.generate_signals()

        # Position shifted by 1 to account for time offset
        self.price_data['shifted_close'] = self.price_data['close'].shift(1)

        # Position with slippage
        slippage_price = self.price_data['shifted_close'] * (1 + self.slippage)

        # Calculation of portfolio value taking into account
        position_shifted = self.price_data['atr_position'].shift(1).fillna(0)
        self.price_data['atr_portfolio_value'] = self.initial_capital + (position_shifted * slippage_price.diff()).cumsum()
        self.price_data['atr_portfolio_value'] *= (1 - self.commission)

        return self.price_data[['timestamp', 'atr_portfolio_value']]

    def get_metrics(self, column: str, position_column: str) -> dict:
        self.run_backtest()

        return super().get_metrics(column=column, position_column=position_column)

if __name__ == "__main__":
    liquid_pairs = get_liquid_pairs()
    for liquid_pair in liquid_pairs:
        print(f"Start working with {liquid_pair}")
        print(
            ATRBasedTrailingBreakout(
                price_data=fetch_binance_data(pair=liquid_pair)).get_metrics(
                column='atr_portfolio_value', position_column='atr_position'
            )
        )

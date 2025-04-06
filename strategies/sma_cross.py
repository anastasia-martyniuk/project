import pandas as pd
import numpy as np

from strategies.base import StrategyBase
from core.data_loader import get_liquid_pairs, download_ohlcv_data_for_liquid_pairs


class SMACrossover(StrategyBase):
    def __init__(self, price_data, short_window: int = 10, long_window: int = 50):
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
        self.price_data['signal'] = np.where(
            self.price_data['SMA_short'] > self.price_data['SMA_long'], 1, np.where(
                self.price_data['SMA_short'] < self.price_data['SMA_long'], -1, 0
            )
        )

        # Position: when to buy or sell
        self.price_data['position'] = self.price_data['signal'].diff().shift(-1)

        return self.price_data

    def run_backtest(self) -> pd.DataFrame:
        self.generate_signals()

        # Apply time drift
        self.price_data['shifted_close'] = self.price_data['close'].shift(self.time_drift_delay)

        # Apply slippage
        slippage_adjusted_price = self.price_data['shifted_close'] * (1 + self.slippage)

        # Calculate portfolio value considering commission and slippage
        self.price_data['portfolio_value'] = (
            self.initial_capital + (self.price_data['position'].shift(1) * slippage_adjusted_price.diff()).cumsum()
        )

        # Apply commission
        self.price_data['portfolio_value'] *= (1 - self.commission)

        return self.price_data

    def get_metrics(self) -> dict:
        self.run_backtest()

        # Calculate total return
        first_nonzero = self.price_data['portfolio_value'].loc[self.price_data['portfolio_value'] > 0].iloc[0]
        last_nonzero = self.price_data['portfolio_value'].loc[self.price_data['portfolio_value'] > 0].iloc[-1]

        if pd.isna(first_nonzero) or pd.isna(last_nonzero):
            total_return = np.nan
        else:
            total_return = (last_nonzero / first_nonzero) - 1

        # Calculate max drawdown
        rolling_max = self.price_data['portfolio_value'].cummax()
        max_drawdown = ((self.price_data['portfolio_value'] - rolling_max) / rolling_max).min()

        # Calculate Sharpe ratio (assuming daily returns)
        daily_returns = self.price_data['portfolio_value'].pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # 252 - count of trading days in year

        # Calculate win rate (percentage of positive returns)
        wins = daily_returns[daily_returns > 0].count()
        total_trades = daily_returns.count()
        win_rate = wins / total_trades if total_trades != 0 else 0

        # Calculate expectancy (expected profit per trade)
        avg_profit_per_trade = daily_returns.mean()
        loss_per_trade = daily_returns[daily_returns < 0].mean()
        expectancy = (win_rate * avg_profit_per_trade) - ((1 - win_rate) * abs(loss_per_trade))

        # Calculate exposure time (percentage of time in the market)
        in_market = self.price_data['position'].notna() & (self.price_data['position'] != 0)
        exposure_time = in_market.sum() / len(self.price_data) * 100

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "expectancy": expectancy,
            "exposure_time": exposure_time,
        }


if __name__ == '__main__':
    data_frames = download_ohlcv_data_for_liquid_pairs(get_liquid_pairs())
    for data_frame in data_frames:
        print(SMACrossover(price_data=data_frame).get_metrics())

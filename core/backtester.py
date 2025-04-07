import pandas as pd
import vectorbt as vbt
import seaborn as sns
import os
import plotly.graph_objects as go

from matplotlib import pyplot as plt

from strategies import StrategyBase

DATA_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')), 'results')


class Backtester:
    def __init__(self, strategy: StrategyBase):
        self.strategy = strategy
        self.price_data = strategy.price_data
        self.results_dir = DATA_DIR

    @staticmethod
    def calculate_metrics(pf: vbt.Portfolio) -> pd.DataFrame:
        stats = pd.DataFrame(index=pf.wrapper.columns)

        stats['Total Return [%]'] = pf.total_return() * 100
        stats['Sharpe Ratio'] = pf.returns().mean() / pf.returns().std() * (252 * 60)**0.5
        stats['Max Drawdown [%]'] = pf.get_drawdowns().drawdown.min() * 100
        stats['Win Rate [%]'] = pf.trades.win_rate() * 100
        stats['Expectancy'] = pf.trades.expectancy()
        # stats['Exposure Time [%]'] = pf.stats()['Exposure Time [%]'] * 100

        return stats.round(2)

    def save_results(self, stats: pd.DataFrame, pf: vbt.Portfolio):
        strategy_name = self.strategy.__class__.__name__.lower()
        stats_path = os.path.join(self.results_dir, f"backtest_metrics_{strategy_name}.csv")
        stats.to_csv(stats_path)

        for symbol in self.price_data.columns:
            fig = pf[symbol].plot()
            fig.write_image(
                os.path.join(
                    os.path.join(self.results_dir, "screenshots"), f"{symbol}_{strategy_name}_equity_curve.png"
                )
            )

        # Create heatmap for performance
        self.generate_performance_heatmap(pf)

    def generate_performance_heatmap(self, pf: vbt.Portfolio):
        # Extract performance data for all symbols
        performance_data = pd.DataFrame(
            {symbol: pf[symbol].total_return() for symbol in self.price_data.columns}, index=['Total Return']
        )

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(performance_data.T, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Total Return [%]'})
        heatmap_path = os.path.join(self.results_dir, "screenshots", "performance_heatmap.png")
        plt.savefig(heatmap_path)

        # Show heatmap in a Jupyter Notebook or directly in code
        plt.show()

    def generate_equity_curve(self, pf: vbt.Portfolio, strategy_name: str):
        # Plot equity curve
        equity_curves = {symbol: pf[symbol].equity_curve() for symbol in self.price_data.columns}

        for symbol, equity_curve in equity_curves.items():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name=f'{symbol} Equity Curve'))
            fig.update_layout(title=f'{strategy_name} Equity Curve for {symbol}',
                              xaxis_title='Time',
                              yaxis_title='Equity',
                              template='plotly_dark')
            fig.write_html(
                os.path.join(self.results_dir, f"equity_curve_{strategy_name}_{symbol}.html")
            )

    def run(self):
        entries, exits = self.strategy.generate_signals()

        pf = vbt.Portfolio.from_signals(
            close=self.price_data,
            entries=entries,
            exits=exits,
            direction='longonly',
            fees=0.001,
            slippage=0.001
        )

        stats = self.calculate_metrics(pf)
        self.save_results(stats, pf)
        self.generate_equity_curve(pf, self.strategy.__class__.__name__.lower())

        return stats

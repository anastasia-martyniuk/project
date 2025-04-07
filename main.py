from core.backtester import Backtester
from core.data_loader import load_price_data
from strategies import SMACrossover



if __name__ == "__main__":
    price_data = load_price_data()
    sma_backtester = Backtester(strategy=SMACrossover(price_data=price_data)).run()

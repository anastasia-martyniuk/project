from core.data_loader import get_liquid_pairs, fetch_binance_data
from strategies import SMACrossover, VWAPReversion



if __name__ == "__main__":
    liquid_pairs = get_liquid_pairs()
    for liquid_pair in liquid_pairs:
        print(f"Start working with {liquid_pair}")
        
        sma_cross = SMACrossover(
            price_data=fetch_binance_data(pair=liquid_pair)).get_metrics(
            column='sma_portfolio_value', position_column='sma_position'
        )
        print(sma_cross)

        vwap_cross = VWAPReversion(
                price_data=fetch_binance_data(pair=liquid_pair)).get_metrics(
                column='vwap_portfolio_value', position_column='vwap_position'
            )
        print(vwap_cross)
        break

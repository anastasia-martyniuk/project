from core.data_loader import get_liquid_pairs, fetch_binance_data
from strategies import SMACrossover, VWAPReversionIntraday, ATRBasedTrailingBreakout



if __name__ == "__main__":
    liquid_pairs = get_liquid_pairs()
    for liquid_pair in liquid_pairs:
        print(f"Start working with {liquid_pair}")

        sma_cross = SMACrossover(
            price_data=fetch_binance_data(pair=liquid_pair)).get_metrics(
            column='sma_portfolio_value', position_column='sma_position'
        )
        print(sma_cross)

        vwap_reversion = VWAPReversionIntraday(
                price_data=fetch_binance_data(pair=liquid_pair)).get_metrics(
                column='vwap_portfolio_value', position_column='vwap_position'
            )
        print(vwap_reversion)

        atr_based = ATRBasedTrailingBreakout(
                price_data=fetch_binance_data(pair=liquid_pair)).get_metrics(
                column='atr_portfolio_value', position_column='atr_position'
            )
        print(atr_based)
        break

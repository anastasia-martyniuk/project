from core.data_loader import get_liquid_pairs, download_ohlcv_data_for_liquid_pairs
from strategies.sma_cross import SMACrossover


if __name__ == '__main__':
    data_frames = download_ohlcv_data_for_liquid_pairs(get_liquid_pairs())
    for data_frame in data_frames:
        print(SMACrossover(price_data=data_frame).get_metrics())

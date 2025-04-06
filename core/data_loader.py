import io
import json
import os

from pandas import DataFrame
from tqdm import tqdm


import ccxt
import pandas as pd
import requests
import zipfile


def get_liquid_pairs(base_asset: str = "BTC", liquid_number: int = 100, cache_file: str = "liquid_pairs.json") -> dict:
    cache_file_path = os.path.join('../data', cache_file)
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r') as f:
            print(f"Downloaded liquid pairs from {cache_file}.")
            return json.load(f)

    binance = ccxt.binance()
    markets = binance.load_markets()
    pairs = [market for market in markets if market.endswith(f"/{base_asset}")]

    volumes = {}
    for pair in tqdm(pairs):
        ticker = binance.fetch_ticker(pair)
        volumes[pair.replace('/', '')] = ticker.get("quoteVolume", 0)

    sorted_pairs = sorted(volumes.items(), key=lambda x: x[1], reverse=True)
    liquid_pairs = dict(sorted_pairs[:liquid_number])

    with open(cache_file_path, 'w') as f:
        json.dump(liquid_pairs, f)
        print(f"Liquid pairs saved to {cache_file}.")

    return liquid_pairs


def fetch_binance_data(
    pair: str, start_date: str, interval: str = "1m", cache_dir: str = "../data"
) -> DataFrame | None:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_path = os.path.join(cache_dir, f"{pair}_{interval}_feb25.parquet")

    if os.path.exists(cache_file_path):
        print(f"Download from cache: {pair}")
        return pd.read_parquet(cache_file_path)

    base_url = "https://data.binance.vision/data/spot/monthly/klines/"
    path = f"{pair}/{interval}/{pair}-{interval}-{start_date}.zip"
    url = base_url + path
    response = requests.get(url)

    if response.status_code == 404:
        print(f"Zip file for {pair} does not exist.")
        return None

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        csv_filename = zip_file.namelist()[0]
        with zip_file.open(csv_filename) as csv_file_obj:
            ohlcv_df = pd.read_csv(csv_file_obj, header=None)
            ohlcv_df.columns = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
            ]

    ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['open_time'], unit='us')
    ohlcv_df.to_parquet(cache_file_path, compression='snappy')

    return ohlcv_df


def download_ohlcv_data_for_liquid_pairs(liquid_pairs: dict, start_date: str = "2025-02") -> list:
    result = []
    for pair in tqdm(liquid_pairs.keys()):
        data = fetch_binance_data(pair, start_date)
        if data is None:
            print(f"Failed to load data for {pair}.")
        else:
            print(f"Data for {pair} loaded.")
        result += data
    return result


if __name__ == "__main__":
    download_ohlcv_data_for_liquid_pairs(get_liquid_pairs())

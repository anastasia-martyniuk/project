# Trading Strategy Backtester

This project implements a backtesting framework for various trading strategies using historical market data. 
It supports strategies such as:

- **SMA Crossover**  
- **ATR-Based Trailing Breakout**  
- **VWAP Reversion Intraday**

## Setup

### 1. Clone the Repository

>> git clone https://github.com/yourusername/project.git

### 2. Setup the Virtual Environment
>> chmod +x setup.sh
>> ./setup.sh

### 3. Data Loading
You can find parquet files for all 100 pairs in directory data. 
But if you want check script, you can clean this directory and start script:
>> python core/data_loader.py

### 4. Start backtest to see the result
You can find all results in directory results. 
But you can clean this directory and start module:
>> python core/main.py

Strategies:

1.SMA Crossover(now the only working strategy)

Description:
This strategy calculates two simple moving averages (SMA) – a short-term and a long-term SMA.

Entry Signal: When the short SMA crosses above the long SMA.
Exit Signal: When the short SMA crosses below the long SMA.

Parameters:
short_window: Number of periods for the short SMA (default: 10)
long_window: Number of periods for the long SMA (default: 50)

2.ATR-Based Trailing Breakout

Description:
This strategy uses the Average True Range (ATR) to determine breakout levels.

Entry Signal: A buy signal is generated when the price exceeds the previous close plus a multiple of ATR.
Exit Signal: A sell signal is triggered when the price falls below this breakout level.

Parameters:
atr_window: Window size for calculating ATR (default: 14)
multiplier: ATR multiplier for defining the breakout level (default: 1.5)

3.VWAP Reversion Intraday

Description:
This strategy is based on the Volume Weighted Average Price (VWAP).

Entry Signal: When the current price is significantly below the VWAP, indicating a potential reversion upward.
Exit Signal: When the price rises above the VWAP.

Parameters:
vwap_window: Window size for calculating VWAP (default: 14)
Note: For strategies like VWAP and ATR, ensure that your data loader provides the required columns (e.g., close, volume, high, and low).

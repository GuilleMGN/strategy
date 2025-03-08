# BingX Perpetual Futures Trading Strategy

This project is an automated trading bot built to trade cryptocurrency perpetual futures on the BingX exchange. The bot uses technical analysis indicators (e.g., Supertrend, SMA, EMA, ATR) to identify trading opportunities and manages risk with predefined stop-loss and take-profit levels. It operates on 1-hour and 5-minute timeframes, executing long and short trades based on trend conditions, and provides real-time updates via the console.

## Features

- **Asset Selection**: Prompts the user to input a cryptocurrency asset (e.g., "BTC", "ETH") and validates it against available assets on BingX using the `openApi/swap/v2/quote/contracts` endpoint.
- **Technical Analysis**:
  - 1-hour timeframe: Uses Supertrend and SMA to determine the overall trend (Up, Down, or None).
  - 5-minute timeframe: Uses EMA (5, 8, 13) and ATR to identify short-term trends and volatility.
- **Trade Execution**:
  - Executes market orders with a 1:3 risk-reward ratio.
  - Sets stop-loss and take-profit levels based on recent candle data and ATR.
  - Supports leverage up to 125x with a 1% risk per trade.
- **Real-Time Monitoring**:
  - Displays initial trends and price after startup.
  - Provides live updates on trade status, balance, and profit/loss.
  - Waits for 5-minute candle closes to ensure accurate data.
- **Risk Management**:
  - Enforces minimum trade amounts based on `tradeMinUSDT` from BingX API.
  - Rounds position sizes and prices to exchange-specified precision.
- **Error Handling**: Includes retry mechanisms for API calls and graceful exit on errors or keyboard interrupt.

## Configuration

* **RISK_PER_TRADE** : Set to 0.01 (1%) in the script; adjust this constant to change risk per trade.
* **SUPER_TREND_PERIOD/MULTIPLIER** : Configurable for 1h trend detection.
* **SMA_PERIOD** : Set to 200 for 1h trend confirmation.
* **ATR_PERIOD** : Set to 14 for volatility calculation.
* **STOP_LOSS_CANDLES** : Set to 3 for stop-loss calculation based on recent highs/lows.

## Requirements

- **Python 3.13.0+**
- Required Libraries:
  - `ccxt` (for interacting with BingX API)
  - `pandas` (for data manipulation)
  - `talib` (for technical indicators)
  - `pandas_ta` (for additional indicators like Supertrend)
  - `requests` (for fetching contract details)
  - `python-dotenv` (for environment variable management)
  - `pytz` (for timezone handling)
- **API Credentials**: BingX API key and secret stored in a `.env` file.
  - Replace **your_api_key** and **your_secret_key** with your BingX API credentials.

```.env
BINGX_API_KEY=your_api_key
BINGX_SECRET_KEY=your_secret_key
```

## Run the Script:

```
python live.py
```

## Usage

1. **Start the Bot** :

* Run the script, and it will prompt you to enter a cryptocurrency asset (e.g., "ADA").
* The bot validates the asset against BingX’s available contracts and initializes with the corresponding symbol (e.g., "ADA-USDT").

1. **Initial Output** :

* Displays initialization progress, balance, and initial trends/price based on fetched data.
* Waits for the next 5-minute candle close before checking for trades.

1. **Live Trading** :

* Checks for trade opportunities every 5 minutes after candle close.
* Executes trades (Long or Short) when conditions are met, displaying entry price, stop-loss, take-profit, and position size.
* Monitors open trades and closes them when stop-loss or take-profit is hit.

1. **Exit** :

* Press **Ctrl+C** to stop the bot manually, which triggers a summary of the trading session (initial balance, final balance, profit/loss).

## Example Output

```
Enter a cryptocurrency asset: xrp

Initializing data for XRP...
Initializing 1h data...
Fetched 700 1h candles for XRP
Initializing 5m data...
Fetched 60 5m candles for XRP

XRP is trading live on BingX Perpetual Futures
BingX Perpetual Futures Balance: 1234.56 USDT
1h trend: Down, 5m trend: Up, Price: 2.3456 USDT
Waiting 0s until 2025-01-23 12:30:00 EST...

Checking for trades at 2025-01-23 12:30:00 EST...
1h trend: Down, 5m trend: Up, Price: 2.3499 USDT
No trades found in this 5m candle for XRP
Waiting 0s until 2025-01-23 12:35:00 EST...

Checking for trades at 2025-01-23 12:35:00 EST...
1h trend: Down, 5m trend: Down, Price: 2.3456 USDT

Trade Opened for XRP:
Enter Short: 2.3456
Take Profit: 2.0858
Stop Loss: 2.4321
Position Size: 1.2345
```

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

Coded by Matthew Guillen © 2025

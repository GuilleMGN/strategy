# BingX Perpetual Futures Trading Strategy

This project is an automated trading bot built to trade cryptocurrency perpetual futures on the BingX exchange. The bot uses technical analysis indicators (e.g., Supertrend, SMA, EMA, ATR) to identify trading opportunities and manages risk with predefined stop-loss and take-profit levels. It operates on 1-hour and 5-minute timeframes, executing long and short trades based on trend conditions, and provides real-time updates via the console.

## Features

- **Asset Selection**: Prompts the user to input a cryptocurrency asset (e.g., "BTC", "ADA") and validates it against available assets on BingX using the `openApi/swap/v2/quote/contracts` endpoint.
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

## Requirements

- **Python 3.x**
- Required Libraries:
  - `ccxt` (for interacting with BingX API)
  - `pandas` (for data manipulation)
  - `talib` (for technical indicators)
  - `pandas_ta` (for additional indicators like Supertrend)
  - `requests` (for fetching contract details)
  - `python-dotenv` (for environment variable management)
  - `pytz` (for timezone handling)
- **API Credentials**: BingX API key and secret stored in a `.env` file.

```.env
BINGX_API_KEY=INSERT PUBLIC KEY
BINGX_SECRET_KEY=INSERT SECRET KEY
```

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

Coded by Matthew Guillen © 2025

import ccxt
import pandas as pd
import talib as ta
import numpy as np
import pandas_ta as pta
import requests
import time
import os
import sys
from dotenv import load_dotenv
from pytz import timezone

# Load API keys from .env file
load_dotenv()
api_key = os.getenv('BINGX_API_KEY')
secret_key = os.getenv('BINGX_SECRET_KEY')

# Initialize exchange with explicit Futures context
exchange = ccxt.bingx({
    'enableRateLimit': True,
    'apiKey': api_key,
    'secret': secret_key,
    'options': {'defaultType': 'swap'}  # Set default to Perpetual Futures (swap)
})

def initialize_crypto_data():
    """Prompt user for a cryptocurrency asset, validate it using BingX API, and fetch all contract details."""
    url = "https://open-api.bingx.com/openApi/swap/v2/quote/contracts"
    try:
        response = requests.get(url)
        response.raise_for_status()
        contracts = response.json().get('data', [])

        # Filter for USDT pairs only
        usdt_contracts = [contract for contract in contracts if contract.get('currency') == 'USDT']
        if not usdt_contracts:
            print("Error: No USDT cryptocurrency pairs found.")
            sys.exit(1)

        valid_assets = {}
        for contract in usdt_contracts:
            asset = contract['asset'].upper()
            symbol = contract['symbol']
            valid_assets[asset] = {
                'symbol': symbol,
                'currency': contract.get('currency', 'USDT'),
                'tradeMinUSDT': float(contract.get('tradeMinUSDT', 5.0)),
                'quantityPrecision': int(contract.get('quantityPrecision', 2)),
                'makerFeeRate': float(contract.get('makerFeeRate', 0.0)),  # Default to 0 if not present
                'takerFeeRate': float(contract.get('takerFeeRate', 0.0)),  # Default to 0 if not present
                'pricePrecision': int(contract.get('pricePrecision', 2))   # Default to 2 if not present
            }

        while True:
            try:
                crypto = input("\nEnter a cryptocurrency symbol: ").strip().upper()
                if crypto in valid_assets:
                    contract_data = valid_assets[crypto]
                    return (crypto, contract_data['symbol'], contract_data['currency'],
                            contract_data['tradeMinUSDT'], contract_data['quantityPrecision'],
                            contract_data['makerFeeRate'], contract_data['takerFeeRate'],
                            contract_data['pricePrecision'])
                else:
                    print("Error: Invalid cryptocurrency asset or not available with USDT pairing. Please try again.")
            except KeyboardInterrupt:
                print("\nLive Trading cancelled.")
                sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching cryptocurrency data: {e}")
        sys.exit(1)

# Use the function to get all required data
global asset, symbol, currency, trade_min_usdt, quantity_precision, maker_fee_rate, taker_fee_rate, price_precision
(asset, symbol, currency, trade_min_usdt, quantity_precision,
 maker_fee_rate, taker_fee_rate, price_precision) = initialize_crypto_data()

# Configuration constants
RISK_PER_TRADE = 0.01  # 1% risk per trade
SUPER_TREND_PERIOD = 10
SUPER_TREND_MULTIPLIER = 3
SMA_PERIOD = 200
ATR_PERIOD = 14
STOP_LOSS_CANDLES = 3

print(f'Asset: {asset}')
print(f'Symbol: {symbol}')
print(f'Currency: {currency}')
print(f'trade_min_usdt: {trade_min_usdt}')
print(f'quantity_precision: {quantity_precision}')
print(f'maker_fee_rate: {maker_fee_rate}')
print(f'taker_fee_rate: {taker_fee_rate}')
print(f'price_precision: {price_precision}')


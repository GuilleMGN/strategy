import ccxt
import pandas as pd
import talib as ta
import numpy as np
import pandas_ta as pta
import time
import os
from dotenv import load_dotenv  # For secure API key storage

# Load API keys from .env file
load_dotenv()
api_key = os.getenv('BINGX_API_KEY')
secret_key = os.getenv('BINGX_SECRET_KEY')

# Initialize exchange (BingX Perpetual Futures)
exchange = ccxt.bingx({
    'enableRateLimit': True,
    'apiKey': api_key,
    'secret': secret_key
})
symbol = 'XRP/USDT:USDT'  # Perpetual Futures symbol for XRP/USDT

# Configuration
INITIAL_BALANCE = 10000.00  # Starting with $10,000 VST
RISK_PER_TRADE = 0.01      # 1% risk per trade
SUPER_TREND_PERIOD = 10
SUPER_TREND_MULTIPLIER = 3
SMA_PERIOD = 200
ATR_PERIOD = 14
STOP_LOSS_CANDLES = 3
LEVERAGE = 3               # 3x leverage for Perpetual Futures

balance = INITIAL_BALANCE
# Use DataFrames instead of lists for ohlcv_5m and ohlcv_1h
ohlcv_5m = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
ohlcv_1h = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

def fetch_ohlcv(timeframe, limit=1000):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # Convert timestamp from milliseconds to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    # Ensure numeric columns and fill NaN with forward fill or drop
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    df = df.ffill().dropna()  # Forward fill NaN and drop any remaining NaN rows
    return df

def calculate_indicators(df, timeframe):
    if df.empty or len(df) < (SUPER_TREND_PERIOD + 1 if timeframe == '1h' else (max(5, 8, 13) + 1)):
        return 'None' if timeframe == '1h' else pd.Series({'EMA5': 0, 'EMA8': 0, 'EMA13': 0, 'ATR': 0, 'close': 0, 'high': 0, 'low': 0})

    if timeframe == '1h':
        try:
            supertrend_result = pta.supertrend(df['high'], df['low'], df['close'], length=SUPER_TREND_PERIOD, multiplier=SUPER_TREND_MULTIPLIER)
            if supertrend_result is None or supertrend_result.empty:
                return 'None'
            df['SuperTrend'] = supertrend_result['SUPERT_10_3.0']
            df['SuperTrend_Direction'] = supertrend_result['SUPERTd_10_3.0']
            df['SMA'] = ta.SMA(df['close'], timeperiod=SMA_PERIOD)
            df['Trend'] = np.where((df['SuperTrend_Direction'] == 1) & (df['close'] > df['SMA']), 'Up',
                                 np.where((df['SuperTrend_Direction'] == -1) & (df['close'] < df['SMA']), 'Down', 'None'))
            return df['Trend'].iloc[-1]
        except Exception as e:
            print(f"Error calculating 1h indicators: {e}")
            return 'None'
    else:  # 5m
        try:
            df = df.copy()  # Avoid SettingWithCopyWarning by working with a copy
            df['EMA5'] = ta.EMA(df['close'], timeperiod=5)
            df['EMA8'] = ta.EMA(df['close'], timeperiod=8)
            df['EMA13'] = ta.EMA(df['close'], timeperiod=13)
            df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_PERIOD)
            return df[['EMA5', 'EMA8', 'EMA13', 'ATR', 'close', 'high', 'low']].iloc[-1]
        except Exception as e:
            print(f"Error calculating 5m indicators: {e}")
            return pd.Series({'EMA5': 0, 'EMA8': 0, 'EMA13': 0, 'ATR': 0, 'close': 0, 'high': 0, 'low': 0})

def execute_trade(trend, ema_data, balance, last_3_candles, leverage=LEVERAGE):
    if trend == 'None' or pd.isna(ema_data['close']) or ema_data['close'] == 0:
        return None, None, None, None, None

    current_close, ema5, ema8, ema13, atr = ema_data['close'], ema_data['EMA5'], ema_data['EMA8'], ema_data['EMA13'], ema_data['ATR']
    high_3 = max(last_3_candles['high'])
    low_3 = min(last_3_candles['low'])

    if trend == 'Up' and current_close > ema5 > ema8 > ema13 and current_close > ema13:
        entry_price = current_close
        stop_loss = low_3 - (2 * atr)
        stop_distance = entry_price - stop_loss
        if stop_distance > 0:
            risk_amount = balance * RISK_PER_TRADE
            # Apply leverage to position size
            position_size = (risk_amount / stop_distance) * leverage
            take_profit = entry_price + (3 * stop_distance)
            # Place order in demo mode (simulate for now)
            print(f"Demo Trade: Entered Long at {entry_price}, SL: {stop_loss}, TP: {take_profit}, Size: {position_size} VST")
            # In live trading, uncomment and use:
            # order = exchange.create_market_order(symbol, 'buy', position_size, {'leverage': leverage})
            # sl_order = exchange.create_order(symbol, 'STOP_LOSS_LIMIT', 'sell', position_size, stop_loss, params={'stopPrice': stop_loss, 'leverage': leverage})
            # tp_order = exchange.create_order(symbol, 'LIMIT', 'sell', position_size, take_profit, params={'leverage': leverage})
            return 'Long', entry_price, stop_loss, take_profit, position_size

    elif trend == 'Down' and ema13 > ema8 > ema5 > current_close and current_close < ema5:
        entry_price = current_close
        stop_loss = high_3 + (2 * atr)
        stop_distance = stop_loss - entry_price
        if stop_distance > 0:
            risk_amount = balance * RISK_PER_TRADE
            # Apply leverage to position size
            position_size = (risk_amount / stop_distance) * leverage
            take_profit = entry_price - (3 * stop_distance)
            # Place order in demo mode (simulate for now)
            print(f"Demo Trade: Entered Short at {entry_price}, SL: {stop_loss}, TP: {take_profit}, Size: {position_size} VST")
            # In live trading, uncomment and use:
            # order = exchange.create_market_order(symbol, 'sell', position_size, {'leverage': leverage})
            # sl_order = exchange.create_order(symbol, 'STOP_LOSS_LIMIT', 'buy', position_size, stop_loss, params={'stopPrice': stop_loss, 'leverage': leverage})
            # tp_order = exchange.create_order(symbol, 'LIMIT', 'buy', position_size, take_profit, params={'leverage': leverage})
            return 'Short', entry_price, stop_loss, take_profit, position_size
    return None, None, None, None, None

def main():
    global balance, ohlcv_5m, ohlcv_1h
    in_position = False
    last_3_candles = pd.DataFrame(columns=['high', 'low'])

    while True:
        # Fetch 5m data (adjust for real-time WebSocket if needed)
        df_5m = fetch_ohlcv('5m', limit=50)
        # Keep ohlcv_5m as a DataFrame, not a list
        ohlcv_5m = df_5m.iloc[-50:]  # Keep last 50 candles as DataFrame

        # Aggregate 5m to 1h
        df_1h = df_5m.resample('1H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        # Keep ohlcv_1h as a DataFrame, not a list
        ohlcv_1h = df_1h.iloc[-24:]  # Keep last 24 hours as DataFrame

        # Calculate trends and EMAs
        trend = calculate_indicators(ohlcv_1h, '1h')
        ema_data = calculate_indicators(ohlcv_5m, '5m')

        # Update last 3 candles
        if len(ohlcv_5m) >= 3:
            last_3_candles = ohlcv_5m.iloc[-3:][['high', 'low']]

        if not in_position:
            position, entry_price, stop_loss, take_profit, position_size = execute_trade(trend, ema_data, balance, last_3_candles)
            if position:
                in_position = True

        # Check for exits (simplified for demo; use WebSocket for real orders)
        if in_position:
            current_price = exchange.fetch_ticker(symbol)['last']
            if position == 'Long' and current_price <= stop_loss:
                pnl = (stop_loss - entry_price) * position_size
                balance += pnl
                print(f"Demo Trade: Exited Long at {stop_loss}, PnL: {pnl} VST")
                in_position = False
            elif position == 'Long' and current_price >= take_profit:
                pnl = (take_profit - entry_price) * position_size
                balance += pnl
                print(f"Demo Trade: Exited Long at {take_profit}, PnL: {pnl} VST")
                in_position = False
            elif position == 'Short' and current_price >= stop_loss:
                pnl = (entry_price - stop_loss) * position_size
                balance += pnl
                print(f"Demo Trade: Exited Short at {stop_loss}, PnL: {pnl} VST")
                in_position = False
            elif position == 'Short' and current_price <= take_profit:
                pnl = (entry_price - take_profit) * position_size
                balance += pnl
                print(f"Demo Trade: Exited Short at {take_profit}, PnL: {pnl} VST")
                in_position = False

        time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    main()
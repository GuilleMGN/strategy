import ccxt
import pandas as pd
import talib as ta
import numpy as np
import pandas_ta as pta
import time
import os
from dotenv import load_dotenv
from pytz import timezone

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
symbol = 'SUI/USDT:USDT'  # Perpetual Futures symbol
print(f'{symbol} is now trading LIVE on BingX Perpetual Futures...\n')

# Configuration
INITIAL_BALANCE = 10000.00  # VST
RISK_PER_TRADE = 0.01
SUPER_TREND_PERIOD = 10
SUPER_TREND_MULTIPLIER = 3
SMA_PERIOD = 200
ATR_PERIOD = 14
STOP_LOSS_CANDLES = 3
LEVERAGE = 3

balance = INITIAL_BALANCE
ohlcv_5m = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
ohlcv_1h = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

def fetch_ohlcv(timeframe, limit=1000, since=None):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Convert timestamp from milliseconds to UTC datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        # Convert to EST (UTC-5) to match the chart's timezone
        df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
        df.set_index('timestamp', inplace=True)
        # Ensure numeric columns and handle NaN values
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        # Check for gaps or invalid data and log
        if df.isnull().any().any():
            print(f"Warning: NaN values found in {timeframe} data before filling. Gaps: {df.isnull().sum()}")
        df = df.ffill().dropna()  # Forward fill NaN and drop any remaining NaN rows
        print(f"Fetched {len(df)} {timeframe} candles for {symbol} (since {since if since else 'latest'})")
        # Debug: Print last few candles for verification
        print(f"Last 3 {timeframe} candles: {df.tail(3)[['open', 'high', 'low', 'close']]}")
        return df
    except Exception as e:
        print(f"Error fetching {timeframe} data: {e}")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

def initialize_historical_1h_data():
    # Fetch historical 1h data (e.g., last 700 candles, about 29 days) from Feb 1, 2025
    since = exchange.parse8601('2025-02-01T00:00:00Z')  # Start from Feb 1, 2025, to match recent chart data
    df_1h = fetch_ohlcv('1h', limit=700, since=since)
    return df_1h

def calculate_indicators(df, timeframe):
    if df.empty or len(df) < (SUPER_TREND_PERIOD + 1 if timeframe == '1h' else (max(5, 8, 13) + 1)):
        print(f"Insufficient data for {timeframe} indicators - returning default")
        return 'None' if timeframe == '1h' else pd.Series({'EMA5': 0, 'EMA8': 0, 'EMA13': 0, 'ATR': 0, 'close': 0, 'high': 0, 'low': 0})

    if timeframe == '1h':
        try:
            supertrend_result = pta.supertrend(df['high'], df['low'], df['close'], length=SUPER_TREND_PERIOD, multiplier=SUPER_TREND_MULTIPLIER)
            if supertrend_result is None or supertrend_result.empty:
                print(f"No SuperTrend result for {timeframe} - returning 'None'")
                return 'None'
            df['SuperTrend'] = supertrend_result['SUPERT_10_3.0']
            df['SuperTrend_Direction'] = supertrend_result['SUPERTd_10_3.0']
            df['SMA'] = ta.SMA(df['close'], timeperiod=SMA_PERIOD)
            df['Trend'] = np.where((df['SuperTrend_Direction'] == 1) & (df['close'] > df['SMA']), 'Up',
                                   np.where((df['SuperTrend_Direction'] == -1) & (df['close'] < df['SMA']), 'Down', 'None'))
            print(f"200 SMA: {df['SMA'].iloc[-1]:.4f}, SuperTrend: {df['SuperTrend'].iloc[-1]:.4f}, Price: {df['close'].iloc[-1]:.4f}")
            print(f"Calculated 1h trend: {df['Trend'].iloc[-1]}")
            return df['Trend'].iloc[-1]
        except Exception as e:
            print(f"Error calculating 1h indicators: {e}")
            return 'None'
    else:  # 5m
        try:
            df = df.copy()
            df['EMA5'] = ta.EMA(df['close'], timeperiod=5)
            df['EMA8'] = ta.EMA(df['close'], timeperiod=8)
            df['EMA13'] = ta.EMA(df['close'], timeperiod=13)
            df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_PERIOD)
            print(f"Calculated 5m trend: {'Up' if df['close'].iloc[-1] > df['EMA13'].iloc[-1] else 'Down'}\nCalculated 5m price: {df['close'].iloc[-1]:.4f}")
            return df[['EMA5', 'EMA8', 'EMA13', 'ATR', 'close', 'high', 'low']].iloc[-1]
        except Exception as e:
            print(f"Error calculating 5m indicators: {e}")
            return pd.Series({'EMA5': 0, 'EMA8': 0, 'EMA13': 0, 'ATR': 0, 'close': 0, 'high': 0, 'low': 0})

def execute_trade(trend, ema_data, balance, last_3_candles, leverage=LEVERAGE):
    if trend == 'None' or pd.isna(ema_data['close']) or ema_data['close'] == 0:
        print("No valid trend or data for trade execution - skipping")
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
            position_size = risk_amount / stop_distance  # Cap risk at 100 VST
            take_profit = entry_price + (3 * stop_distance)
            print(f"\nNew Trade Opened: Entered Long at {entry_price:.4f}, SL: {stop_loss:.4f}, TP: {take_profit:.4f}, Size: {position_size:.4f} VST")
            return 'Long', entry_price, stop_loss, take_profit, position_size

    elif trend == 'Down' and ema13 > ema8 > ema5 > current_close and current_close < ema5:
        entry_price = current_close
        stop_loss = high_3 + (2 * atr)
        stop_distance = stop_loss - entry_price
        if stop_distance > 0:
            risk_amount = balance * RISK_PER_TRADE
            position_size = risk_amount / stop_distance  # Cap risk at 100 VST
            take_profit = entry_price - (3 * stop_distance)
            print(f"\nNew Trade Opened: Entered Short at {entry_price:.4f}, SL: {stop_loss:.4f}, TP: {take_profit:.4f}, Size: {position_size:.4f} VST")
            return 'Short', entry_price, stop_loss, take_profit, position_size
    print("\nNo trades found in this 5m candle...")
    return None, None, None, None, None

def main():
    global balance, ohlcv_5m, ohlcv_1h
    in_position = False
    last_3_candles = pd.DataFrame(columns=['high', 'low'])

    print("Initializing with historical 1h data...")
    ohlcv_1h = initialize_historical_1h_data()

    while True:
        print(f"\nChecking for trades at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        df_5m = fetch_ohlcv('5m', limit=60)
        ohlcv_5m = df_5m.iloc[-60:]

        df_1h = df_5m.resample('1H', closed='right', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        ohlcv_1h = pd.concat([ohlcv_1h, df_1h]).sort_index().iloc[-700:]  # Keep last 700 hours (about 29 days) for SMA 200

        trend = calculate_indicators(ohlcv_1h, '1h')
        ema_data = calculate_indicators(ohlcv_5m, '5m')

        if len(ohlcv_5m) >= 3:
            last_3_candles = ohlcv_5m.iloc[-3:][['high', 'low']]

        if not in_position:
            position, entry_price, stop_loss, take_profit, position_size = execute_trade(trend, ema_data, balance, last_3_candles)
            if position:
                in_position = True

        if in_position:
            while in_position:
                current_price = exchange.fetch_ticker(symbol)['last']
                if position == 'Long':
                    if current_price <= stop_loss:
                        pnl = (stop_loss - entry_price) * position_size
                        balance += pnl
                        print(f"Trade Closed: Exited Long at {stop_loss:.4f}, PnL: {pnl:.2f} VST")
                        in_position = False
                    elif current_price >= take_profit:
                        pnl = (take_profit - entry_price) * position_size
                        balance += pnl
                        print(f"Trade Closed: Exited Long at {take_profit:.4f}, PnL: {pnl:.2f} VST")
                        in_position = False
                elif position == 'Short':
                    if current_price >= stop_loss:
                        pnl = (entry_price - stop_loss) * position_size
                        balance += pnl
                        print(f"Trade Closed: Exited Short at {stop_loss:.4f}, PnL: {pnl:.2f} VST")
                        in_position = False
                    elif current_price <= take_profit:
                        pnl = (entry_price - take_profit) * position_size
                        balance += pnl
                        print(f"Trade Closed: Exited Short at {take_profit:.4f}, PnL: {pnl:.2f} VST")
                        in_position = False
                time.sleep(1)  # Check every second

        time.sleep(300)  # Check for new trades every 5 minutes

if __name__ == "__main__":
    main()
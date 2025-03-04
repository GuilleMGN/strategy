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

# Initialize exchange
exchange = ccxt.bingx({
    'enableRateLimit': True,
    'apiKey': api_key,
    'secret': secret_key
})
symbol = 'ADA/USDT:USDT'

# Configuration constants
RISK_PER_TRADE = 0.01  # 1% risk per trade
SUPER_TREND_PERIOD = 10
SUPER_TREND_MULTIPLIER = 3
SMA_PERIOD = 200
ATR_PERIOD = 14
STOP_LOSS_CANDLES = 3

# Global variables
balance = None
ohlcv_5m = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'EMA5', 'EMA8', 'EMA13', 'ATR'])
ohlcv_1h = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
previous_5m_trend = 'None'

def get_account_balance():
    """Fetch the current available margin balance from BingX Perpetual Futures."""
    try:
        account_info = exchange.fetch_balance(params={'type': 'future'})
        if 'USDT' in account_info and 'free' in account_info['USDT']:
            available_margin = float(account_info['USDT']['free'])
            if available_margin > 0.01:
                return available_margin
        if 'info' in account_info:
            if 'freeMargin' in account_info['info']:
                available_margin = float(account_info['info']['freeMargin'])
                if available_margin > 0.01:
                    return available_margin
            if 'availableBalance' in account_info['info']:
                available_margin = float(account_info['info']['availableBalance'])
                if available_margin > 0.01:
                    return available_margin
            if 'totalMargin' in account_info['info'] and 'usedMargin' in account_info['info']:
                available_margin = float(account_info['info']['totalMargin']) - float(account_info['info']['usedMargin'])
                if available_margin > 0.01:
                    return available_margin
        if 'info' in account_info and 'data' in account_info['info'] and 'balances' in account_info['info']['data']:
            for balance in account_info['info']['data']['balances']:
                if balance['asset'] == 'USDT':
                    available_margin = float(balance['free'])
                    if available_margin > 0.01:
                        return available_margin
        print(f"Could not find valid available margin field for {symbol}")
        return 0.0
    except Exception as e:
        print(f"Error fetching account balance for {symbol}: {e}")
        return 0.0

def check_open_positions():
    """Check if there are any open positions for the symbol."""
    try:
        positions = exchange.fetch_positions([symbol], params={'type': 'future'})
        for position in positions:
            if position['info'].get('positionAmt', 0) != 0:
                print(f"Open position detected for {symbol}: Amount = {position['info']['positionAmt']}")
                return True
        return False
    except Exception as e:
        print(f"Error checking open positions for {symbol}: {e}")
        return False

def calculate_trade_parameters(entry_price, stop_loss_price, current_balance):
    """Calculate position size, leverage, and margin for 1% risk with 1:3 risk-reward."""
    risk_amount = current_balance * RISK_PER_TRADE
    stop_loss_distance = abs(entry_price - stop_loss_price)
    
    if stop_loss_distance <= 0:
        print(f"Invalid stop-loss distance for {symbol}: {stop_loss_distance}")
        return None, None, None

    position_size = risk_amount / stop_loss_distance
    notional_value = position_size * entry_price
    required_leverage = max(1, (notional_value / current_balance) + 1)
    leverage = min(125, round(required_leverage))
    adjusted_margin = notional_value / leverage

    if adjusted_margin > current_balance:
        print(f"Insufficient margin for {symbol}: Required {adjusted_margin:.2f} USDT, Available {current_balance:.2f} USDT")
        return None, None, None

    position_size = round(position_size, 1)  # Adjust precision as needed
    return position_size, leverage, adjusted_margin

def fetch_ohlcv(timeframe, limit=1000, since=None, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit, params={'type': 'future'})
            if not ohlcv:
                print(f"No data fetched for {timeframe} with limit {limit}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('America/New_York')
            df.set_index('timestamp', inplace=True)
            df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            df = df.ffill().dropna()
            return df
        except ccxt.NetworkError as e:
            print(f"Network error fetching {timeframe} data: {e}. Retrying ({retry_count + 1}/{max_retries})...")
            retry_count += 1
            time.sleep(1)
        except ccxt.ExchangeError as e:
            print(f"Exchange error fetching {timeframe} data: {e}. Retrying ({retry_count + 1}/{max_retries})...")
            retry_count += 1
            time.sleep(1)
    print(f"Max retries reached for {timeframe} data.")
    return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

def initialize_historical_1h_data():
    df_1h = fetch_ohlcv('1h', limit=700)
    print(f"Fetched {len(df_1h)} 1h candles for {symbol}")
    return df_1h

def initialize_5m_data():
    df_5m = fetch_ohlcv('5m', limit=60)
    print(f"Fetched {len(df_5m)} 5m candles for {symbol}")
    return df_5m

def wait_for_candle_close(current_time_est, timeframe='5m'):
    est_timezone = timezone('America/New_York')
    if not isinstance(current_time_est, pd.Timestamp):
        current_time_est = pd.Timestamp(current_time_est, tz=est_timezone)
    elif current_time_est.tz is None:
        current_time_est = current_time_est.tz_localize(est_timezone)
    
    if timeframe == '5m':
        minutes = current_time_est.minute
        seconds = current_time_est.second + current_time_est.microsecond / 1e6
        next_5m = current_time_est.replace(minute=((minutes // 5 + 1) * 5) % 60, second=0, microsecond=0)
        if next_5m.minute == 0 and minutes >= 55:
            next_5m += pd.Timedelta(hours=1)
        wait_seconds = (next_5m - current_time_est).total_seconds()
        if wait_seconds < 0:
            wait_seconds += 300
        print(f"Waiting {wait_seconds:.1f} seconds for 5m candle at {next_5m.strftime('%Y-%m-%d %H:%M:%S')} EST...")
        time.sleep(wait_seconds)
        return next_5m
    else:
        next_hour = current_time_est.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
        wait_seconds = (next_hour - current_time_est).total_seconds()
        print(f"Waiting {wait_seconds:.1f} seconds for 1h candle at {next_hour.strftime('%Y-%m-%d %H:%M:%S')} EST...")
        time.sleep(wait_seconds)
        return next_hour

def calculate_indicators(df, timeframe, other_df=None):
    global ohlcv_5m, ohlcv_1h
    min_length = SUPER_TREND_PERIOD + 1 if timeframe == '1h' else max(5, 8, 13) + 1
    if df.empty or len(df) < min_length:
        print(f"Insufficient data for {timeframe} indicators.")
        return ('None', 'Unknown', np.nan, np.nan, np.nan) if timeframe == '1h' else (pd.Series({'EMA5': 0, 'EMA8': 0, 'EMA13': 0, 'ATR': 0, 'close': 0, 'high': 0, 'low': 0}), 'Unknown')

    if timeframe == '1h':
        try:
            supertrend_result = pta.supertrend(df['high'], df['low'], df['close'], length=SUPER_TREND_PERIOD, multiplier=SUPER_TREND_MULTIPLIER)
            df['SuperTrend'] = supertrend_result[f'SUPERT_{SUPER_TREND_PERIOD}_{SUPER_TREND_MULTIPLIER}.0']
            df['SuperTrend_Direction'] = supertrend_result[f'SUPERTd_{SUPER_TREND_PERIOD}_{SUPER_TREND_MULTIPLIER}.0']
            df['SMA'] = ta.SMA(df['close'], timeperiod=SMA_PERIOD)
            df['Trend'] = np.where((df['SuperTrend_Direction'] == 1) & (df['close'] > df['SMA']), 'Up',
                                   np.where((df['SuperTrend_Direction'] == -1) & (df['close'] < df['SMA']), 'Down', 'None'))
            ohlcv_1h = df.copy()
            return df['Trend'].iloc[-1], 'Unknown', df['SMA'].iloc[-1], df['SuperTrend'].iloc[-1], df['close'].iloc[-1]
        except Exception as e:
            print(f"Error calculating 1h indicators: {e}")
            return 'None', 'Unknown', np.nan, np.nan, np.nan
    else:  # 5m
        try:
            df = df.copy()
            df['EMA5'] = ta.EMA(df['close'], timeperiod=5)
            df['EMA8'] = ta.EMA(df['close'], timeperiod=8)
            df['EMA13'] = ta.EMA(df['close'], timeperiod=13)
            df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_PERIOD)
            ohlcv_5m = df.tail(60)
            price = df['close'].iloc[-1]
            ema5 = df['EMA5'].iloc[-1]
            ema8 = df['EMA8'].iloc[-1]
            ema13 = df['EMA13'].iloc[-1]
            if price > ema5 > ema8 > ema13:
                five_m_trend = 'Up'
            elif ema13 > ema8 > ema5 > price:
                five_m_trend = 'Down'
            else:
                five_m_trend = 'None'
            return df[['EMA5', 'EMA8', 'EMA13', 'ATR', 'close', 'high', 'low']].iloc[-1], five_m_trend
        except Exception as e:
            print(f"Error calculating 5m indicators: {e}")
            return pd.Series({'EMA5': 0, 'EMA8': 0, 'EMA13': 0, 'ATR': 0, 'close': 0, 'high': 0, 'low': 0}), 'Unknown'

def execute_trade(trend, ema_data, current_balance, last_3_candles, previous_5m_trend):
    if trend == 'None' or pd.isna(ema_data['close']) or ema_data['close'] == 0:
        print("No trades found in this 5m candle...")
        return None, None, None, None, None, None

    current_close, ema5, ema8, ema13, atr = (ema_data['close'], ema_data['EMA5'], ema_data['EMA8'], ema_data['EMA13'], ema_data['ATR'])
    high_3 = max(last_3_candles['high']) if not last_3_candles.empty else current_close
    low_3 = min(last_3_candles['low']) if not last_3_candles.empty else current_close

    position_size, leverage, margin = calculate_trade_parameters(current_close, low_3 if trend == 'Up' else high_3, current_balance)
    if position_size is None or leverage is None or margin is None:
        return None, None, None, None, None, None

    if trend == 'Up' and current_close > ema5 > ema8 > ema13 and previous_5m_trend in ['Down', 'None']:
        entry_price = current_close
        stop_loss = low_3 - (2 * atr)
        stop_distance = entry_price - stop_loss
        if stop_distance <= 0:
            print(f"Invalid stop-loss distance for {symbol}: {stop_distance}")
            return None, None, None, None, None, None
        take_profit = entry_price + (3 * stop_distance)
        try:
            exchange.set_leverage(leverage, symbol, params={"marginMode": "isolated", 'side': 'BOTH'})
            order = exchange.create_order(
                symbol,
                'market',
                'buy',
                position_size,
                params={
                    'stopLoss': {'type': 'STOP_MARKET', 'stopPrice': stop_loss},
                    'takeProfit': {'type': 'TAKE_PROFIT_MARKET', 'stopPrice': take_profit},
                    'marginMode': 'isolated'
                }
            )
            if not order or 'id' not in order:
                raise Exception("Failed to open position")
            print(f"\nTrade Opened:\nEnter Long: {entry_price:.4f}\nTake Profit: {take_profit:.4f}\nStop Loss: {stop_loss:.4f}\nPosition Size: {position_size:.1f}\nLeverage: {leverage}x\ntime: {order['timestamp']}")
            return 'Long', entry_price, stop_loss, take_profit, position_size, margin
        except Exception as e:
            print(f"Error placing long order: {e}")
            return None, None, None, None, None, None
    elif trend == 'Down' and ema13 > ema8 > ema5 > current_close and previous_5m_trend in ['Up', 'None']:
        entry_price = current_close
        stop_loss = high_3 + (2 * atr)
        stop_distance = stop_loss - entry_price
        if stop_distance <= 0:
            print(f"Invalid stop-loss distance for {symbol}: {stop_distance}")
            return None, None, None, None, None, None
        take_profit = entry_price - (3 * stop_distance)
        try:
            exchange.set_leverage(leverage, symbol, params={"marginMode": "isolated", 'side': 'BOTH'})
            order = exchange.create_order(
                symbol,
                'market',
                'sell',
                position_size,
                params={
                    'stopLoss': {'type': 'STOP_MARKET', 'stopPrice': stop_loss},
                    'takeProfit': {'type': 'TAKE_PROFIT_MARKET', 'stopPrice': take_profit},
                    'marginMode': 'isolated'
                }
            )
            if not order or 'id' not in order:
                raise Exception("Failed to open position")
            print(f"\nTrade Opened:\nEnter Short: {entry_price:.4f}\nTake Profit: {take_profit:.4f}\nStop Loss: {stop_loss:.4f}\nPosition Size: {position_size:.1f}\nLeverage: {leverage}x\ntime: {order['timestamp']}")
            return 'Short', entry_price, stop_loss, take_profit, position_size, margin
        except Exception as e:
            print(f"Error placing short order: {e}")
            return None, None, None, None, None, None
    else:
        print("No trades found in this 5m candle due to trend conditions...")
        return None, None, None, None, None, None

def print_summary(final_balance, initial_balance):
    if initial_balance <= 0:
        print("\nTrading Session Summary:")
        print(f"Initial Balance: Unknown (API error)")
        print(f"Final Balance: {final_balance:.2f} USDT")
        print("Profit/Loss: Cannot calculate due to initial balance error")
        print("\nLive Trading Finished.")
        return
    
    profit_loss = final_balance - initial_balance
    profit_percentage = (profit_loss / initial_balance) * 100 if initial_balance > 0 else 0.0
    print("\nTrading Session Summary:")
    print(f"Initial Balance: {initial_balance:.2f} USDT")
    print(f"Final Balance: {final_balance:.2f} USDT")
    print(f"Profit/Loss: {profit_loss:.2f} USDT ({profit_percentage:.2f}%)")
    print("\nLive Trading Finished.")

def main():
    global balance, ohlcv_5m, ohlcv_1h, previous_5m_trend
    in_position = False
    position = None
    entry_price = None
    stop_loss = None
    take_profit = None
    position_size = None
    margin = None

    try:
        balance = get_account_balance()
        if balance <= 0:
            raise ValueError("No available balance; check API connection or account.")
        initial_balance = balance
        print(f"Starting live trading for {symbol} with initial balance {balance:.2f} USDT")

        print("Fetching initial 1h data...")
        ohlcv_1h = initialize_historical_1h_data()
        print("Fetching initial 5m data...")
        ohlcv_5m = initialize_5m_data()
        print(f"\n{symbol} is now trading LIVE on BingX Perpetual Futures.")
        print(f"Account Balance: {balance:.4f} USDT")

        while True:
            est_timezone = timezone('America/New_York')
            current_est = pd.Timestamp.now(tz=est_timezone)
            next_5m_close = wait_for_candle_close(current_est, '5m')
            print(f"\nChecking status at {next_5m_close.strftime('%Y-%m-%d %H:%M:%S')} EST")

            balance = get_account_balance()
            if balance <= 0:
                print(f"No available balance; stopping trading.")
                break

            ohlcv_5m = fetch_ohlcv('5m', limit=60)
            ema_data, current_5m_trend = calculate_indicators(ohlcv_5m, '5m')

            current_hour = current_est.replace(minute=0, second=0, microsecond=0)
            if ohlcv_1h.empty or current_hour > ohlcv_1h.index[-1]:
                ohlcv_1h = fetch_ohlcv('1h', limit=700)

            current_trend, _, sma, supertrend, price = calculate_indicators(ohlcv_1h, '1h')
            print(f"200 SMA: {sma:.4f}, SuperTrend: {supertrend:.4f}, Price: {price:.4f}")
            print(f"1h trend: {current_trend}, 5m trend: {current_5m_trend}")

            last_3_candles = ohlcv_5m.tail(STOP_LOSS_CANDLES)[['high', 'low']] if len(ohlcv_5m) >= STOP_LOSS_CANDLES else pd.DataFrame()

            if not in_position and not check_open_positions():
                position, entry_price, stop_loss, take_profit, position_size, margin = execute_trade(current_trend, ema_data, balance, last_3_candles, previous_5m_trend)
                if position:
                    in_position = True
                    balance -= margin
                    print(f"Trade opened: {position}, Entry Price: {entry_price:.4f}, Balance after trade: {balance:.2f} USDT, Margin Used: {margin:.2f} USDT")

            previous_5m_trend = current_5m_trend

    except KeyboardInterrupt:
        print_summary(balance, initial_balance)
    except Exception as e:
        print(f"\nError encountered: {e}")
        print_summary(balance, initial_balance)

if __name__ == "__main__":
    main()
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
symbol = 'BTC/USDT'
bingx_symbol = symbol + ':USDT'

# Configuration constants
RISK_PER_TRADE = 0.01  # 1% risk per trade
SUPER_TREND_PERIOD = 10
SUPER_TREND_MULTIPLIER = 3
SMA_PERIOD = 200
ATR_PERIOD = 14
STOP_LOSS_CANDLES = 3

# Minimum amount precision for symbols
MIN_AMOUNT_PRECISION = {
    'BTC/USDT:USDT': 0.0001,
    'ETH/USDT:USDT': 0.01,
    'SOL/USDT:USDT': 1.0,
    'XRP/USDT:USDT': 1.0,
    'ADA/USDT:USDT': 1.0,
    'SUI/USDT:USDT': 2.0,
    'LTC/USDT:USDT': 0.1
}

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
        positions = exchange.fetch_positions([bingx_symbol], params={'type': 'future'})
        for position in positions:
            if float(position['info'].get('positionAmt', 0)) != 0:
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
    
    # Check minimum amount precision
    min_amount = MIN_AMOUNT_PRECISION.get(bingx_symbol, 0.001)
    if position_size < min_amount:
        print(f"Position size {position_size:.4f} below minimum {min_amount} for {symbol}. Skipping trade.")
        return None, None, None

    # Start with 1x leverage, add +1 buffer to cover fees
    required_leverage = notional_value / current_balance
    base_leverage = max(1, round(required_leverage))
    leverage = min(125, base_leverage + 1)
    margin = notional_value / leverage

    if margin > current_balance:
        base_leverage = round(notional_value / current_balance)
        leverage = min(125, base_leverage + 1)
        margin = notional_value / leverage
        if margin > current_balance:
            print(f"Insufficient margin for {symbol}: Required {margin:.2f} USDT, Available {current_balance:.2f} USDT")
            return None, None, None

    position_size = round(position_size, 1)  # Adjust precision as needed
    return position_size, leverage, margin

def fetch_ohlcv(timeframe, limit=1000, since=None, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            ohlcv = exchange.fetch_ohlcv(bingx_symbol, timeframe, since=since, limit=limit, params={'type': 'future'})
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

def wait_for_candle_close(current_time_est):
    est_timezone = timezone('America/New_York')
    if not isinstance(current_time_est, pd.Timestamp):
        current_time_est = pd.Timestamp(current_time_est, tz=est_timezone)
    elif current_time_est.tz is None:
        current_time_est = current_time_est.tz_localize(est_timezone)
    
    minutes = current_time_est.minute
    seconds = current_time_est.second + current_time_est.microsecond / 1e6
    next_5m = current_time_est.replace(minute=((minutes // 5 + 1) * 5) % 60, second=0, microsecond=0)
    if next_5m.minute == 0 and minutes >= 55:
        next_5m += pd.Timedelta(hours=1)
    wait_seconds = (next_5m - current_time_est).total_seconds()
    if wait_seconds < 0:
        wait_seconds += 300
    return wait_seconds, next_5m

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
        print("No trades found in this 5m candle")
        return None, None, None, None, None, None

    current_close, ema5, ema8, ema13, atr = (ema_data['close'], ema_data['EMA5'], ema_data['EMA8'], ema_data['EMA13'], ema_data['ATR'])
    high_3 = max(last_3_candles['high']) if not last_3_candles.empty else current_close
    low_3 = min(last_3_candles['low']) if not last_3_candles.empty else current_close

    # Calculate actual stop loss price before determining position size
    stop_loss = low_3 - (2 * atr) if trend == 'Up' else high_3 + (2 * atr)
    position_size, leverage, margin = calculate_trade_parameters(current_close, stop_loss, current_balance)
    if position_size is None or leverage is None or margin is None:
        return None, None, None, None, None, None

    if trend == 'Up' and current_close > ema5 > ema8 > ema13 and previous_5m_trend in ['Down', 'None']:
        entry_price = current_close
        stop_distance = entry_price - stop_loss
        if stop_distance <= 0:
            print(f"Invalid stop-loss distance for {symbol}: {stop_distance}")
            return None, None, None, None, None, None
        take_profit = entry_price + (3 * stop_distance)
        try:
            exchange.set_leverage(leverage, bingx_symbol, params={"marginMode": "isolated", 'side': 'BOTH'})
            order = exchange.create_order(
                bingx_symbol,
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
            print(f"\nTrade Opened:\nEnter Long: {entry_price:.4f}\nTake Profit: {take_profit:.4f}\nStop Loss: {stop_loss:.4f}\nPosition Size: {position_size:.1f}\nLeverage: {leverage}x\nMargin: {margin:.2f} USDT")
            return 'Long', entry_price, stop_loss, take_profit, position_size, margin
        except Exception as e:
            print(f"Error placing long order: {e}")
            return None, None, None, None, None, None
    elif trend == 'Down' and ema13 > ema8 > ema5 > current_close and previous_5m_trend in ['Up', 'None']:
        entry_price = current_close
        stop_distance = stop_loss - entry_price
        if stop_distance <= 0:
            print(f"Invalid stop-loss distance for {symbol}: {stop_distance}")
            return None, None, None, None, None, None
        take_profit = entry_price - (3 * stop_distance)
        try:
            exchange.set_leverage(leverage, bingx_symbol, params={"marginMode": "isolated", 'side': 'BOTH'})
            order = exchange.create_order(
                bingx_symbol,
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
            print(f"\nTrade Opened:\nEnter Short: {entry_price:.4f}\nTake Profit: {take_profit:.4f}\nStop Loss: {stop_loss:.4f}\nPosition Size: {position_size:.1f}\nLeverage: {leverage}x\nMargin: {margin:.2f} USDT")
            return 'Short', entry_price, stop_loss, take_profit, position_size, margin
        except Exception as e:
            print(f"Error placing short order: {e}")
            return None, None, None, None, None, None
    else:
        print("No trades found due to trend conditions.")
        return None, None, None, None, None, None

def print_summary(initial_balance):
    final_balance = get_account_balance()
    if initial_balance <= 0 or final_balance <= 0:
        print("\nTrading Session Summary:")
        print(f"Initial Balance: {initial_balance if initial_balance > 0 else 'Unknown (API error)'} USDT")
        print(f"Final Balance: {final_balance if final_balance > 0 else 'Unknown (API error)'} USDT")
        print("Profit/Loss: Cannot calculate due to balance error")
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
        print(f"\nInitializing data for {symbol}...")
        print("Fetching initial 1h data...")
        ohlcv_1h = initialize_historical_1h_data()
        print("Fetching initial 5m data...")
        ohlcv_5m = initialize_5m_data()
        print(f"\n{symbol} is trading live on BingX Perpetual Futures")
        print(f"BingX Perpetual Futures Balance: ${balance:.4f} USDT")

        while True:
            current_est = pd.Timestamp.now(tz=timezone('America/New_York'))
            wait_seconds, next_5m_close = wait_for_candle_close(current_est)
            current_5m_close = next_5m_close  # Check the candle that just closed

            print(f"Waiting {wait_seconds:.1f}s until {next_5m_close.strftime('%Y-%m-%d %H:%M:%S')} EST...")
            time.sleep(wait_seconds)

            balance = get_account_balance()
            if balance <= 0:
                print(f"No available balance; trading stopped.")
                break

            ohlcv_5m = fetch_ohlcv('5m', limit=60)
            ema_data, current_5m_trend = calculate_indicators(ohlcv_5m, '5m')

            if ohlcv_1h.empty or current_est > ohlcv_1h.index[-1]:
                ohlcv_1h = fetch_ohlcv('1h', limit=700)

            current_trend, _, sma, supertrend, _ = calculate_indicators(ohlcv_1h, '1h')
            price = ema_data['close']  # Use 5m candle close for console display
            print(f"\nChecking for trades at {current_5m_close.strftime('%Y-%m-%d %H:%M:%S')} EST...")
            print(f"1h trend: {current_trend}, 5m trend: {current_5m_trend}, Price: {price:.4f} USDT")

            last_3_candles = ohlcv_5m.tail(STOP_LOSS_CANDLES)[['high', 'low']] if len(ohlcv_5m) >= STOP_LOSS_CANDLES else pd.DataFrame()

            if in_position:
                current_price = exchange.fetch_ticker(bingx_symbol)['last']
                if position == 'Long':
                    if current_price <= stop_loss:
                        pnl = (stop_loss - entry_price) * position_size
                        print(f"\nTrade Closed:\nExit Long: {stop_loss:.4f}\nPnL: {pnl:.4f} USDT")
                        in_position = False
                    elif current_price >= take_profit:
                        pnl = (take_profit - entry_price) * position_size
                        print(f"\nTrade Closed:\nExit Long: {take_profit:.4f}\nPnL: {pnl:.4f} USDT")
                        in_position = False
                elif position == 'Short':
                    if current_price >= stop_loss:
                        pnl = (entry_price - stop_loss) * position_size
                        print(f"\nTrade Closed:\nExit Short: {stop_loss:.4f}\nPnL: {pnl:.4f} USDT")
                        in_position = False
                    elif current_price <= take_profit:
                        pnl = (entry_price - take_profit) * position_size
                        print(f"\nTrade Closed:\nExit Short: {take_profit:.4f}\nPnL: {pnl:.4f} USDT")
                        in_position = False
                if not in_position:
                    balance = get_account_balance()
                    print(f"Account Balance: {balance:.4f} USDT")
            elif not check_open_positions():
                position, entry_price, stop_loss, take_profit, position_size, margin = execute_trade(current_trend, ema_data, balance, last_3_candles, previous_5m_trend)
                if position:
                    in_position = True
            previous_5m_trend = current_5m_trend

    except KeyboardInterrupt:
        print_summary(initial_balance)
    except Exception as e:
        print(f"\nError encountered: {e}")
        print_summary(initial_balance)

if __name__ == "__main__":
    main()
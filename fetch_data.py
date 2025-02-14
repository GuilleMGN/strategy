import ccxt
import pandas as pd
import time

# Initialize Binance API
exchange = ccxt.binance()

# Define parameters
symbol = 'BTC/USDT'
timeframes = {'1h': '1h', '5m': '5m'}
since = exchange.parse8601('2025-01-01T00:00:00Z')
limit = 1000  # Max limit per request

def fetch_ohlcv(timeframe):
    print(f'Fetching {timeframe} data...')
    ohlcv = []
    since_local = since  # Use local variable to maintain original start date
    while True:
        new_data = exchange.fetch_ohlcv(symbol, timeframe, since_local, limit)
        if not new_data:
            break
        ohlcv.extend(new_data)
        since_local = new_data[-1][0] + 1  # Move forward in time to fetch more data
        time.sleep(1)  # Respect API rate limits
        
        # Break if we've reached the most recent data
        if since_local >= exchange.milliseconds():
            break
    return ohlcv

# Fetch data
oh1 = fetch_ohlcv(timeframes['1h'])
oh5 = fetch_ohlcv(timeframes['5m'])

# Convert to DataFrame and save to CSV
def save_to_csv(data, timeframe):
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.to_csv(f'BTCUSDT_{timeframe}.csv', index=False)
    print(f'Saved BTCUSDT_{timeframe}.csv')

save_to_csv(oh1, '1h')
save_to_csv(oh5, '5m')
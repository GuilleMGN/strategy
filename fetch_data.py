import ccxt
import pandas as pd
import time
import os

# Initialize API
exchange = ccxt.binance()

# Define parameters
asset = 'XRP'
symbol = asset + '/USDT'
timeframes = {
              '1h': '1h', 
              '5m': '5m'
              }
since = exchange.parse8601('2024-02-22T00:00:00Z')  # Format YYYY-MM-DD
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
    # Create Data directory if it doesn't exist
    data_dir = 'Data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Save to CSV in the Data folder
    data_file_path = os.path.join(data_dir, f'{asset}_data_{timeframe}.csv')
    df.to_csv(data_file_path, index=False)
    print(f'Saved {data_file_path}')

save_to_csv(oh1, '1h')
save_to_csv(oh5, '5m')
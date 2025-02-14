import pandas as pd
import numpy as np

# Load data
ohlcv_1h = pd.read_csv('BTCUSDT_1h.csv', parse_dates=['timestamp'])
oh1 = ohlcv_1h.set_index('timestamp')

oh5 = pd.read_csv('BTCUSDT_5m.csv', parse_dates=['timestamp'])
oh5 = oh5.set_index('timestamp')

# Calculate 1h trend using 8 EMA and 34 EMA
oh1['ema_8'] = oh1['close'].ewm(span=8).mean()
oh1['ema_34'] = oh1['close'].ewm(span=34).mean()
oh1['trend'] = (oh1['close'] > oh1['ema_8']) & (oh1['ema_8'] > oh1['ema_34'])  # Uptrend
oh1['trend'] = np.where((oh1['close'] < oh1['ema_8']) & (oh1['ema_8'] < oh1['ema_34']), False, np.nan)  # Downtrend

oh5['trend'] = oh1['trend'].reindex(oh5.index, method='ffill')

# Calculate 3EMA crossover
oh5['ema_5'] = oh5['close'].ewm(span=5).mean()
oh5['ema_8'] = oh5['close'].ewm(span=8).mean()
oh5['ema_13'] = oh5['close'].ewm(span=13).mean()
oh5['long_signal'] = (oh5['ema_5'] > oh5['ema_8']) & (oh5['ema_8'] > oh5['ema_13'])
oh5['short_signal'] = (oh5['ema_5'] < oh5['ema_8']) & (oh5['ema_8'] < oh5['ema_13'])

# Debugging logs
print(f"Total Long Signals: {oh5['long_signal'].sum()}")
print(f"Total Short Signals: {oh5['short_signal'].sum()}\n")

# ATR for Stop Loss
atr_period = 14
oh5['atr'] = oh5['high'].rolling(atr_period).max() - oh5['low'].rolling(atr_period).min()
oh5.dropna(subset=['atr'], inplace=True)  # Remove rows where ATR is NaN

# Backtesting
initial_balance = 10000
balance = initial_balance
risk_per_trade = 0.01
trade_active = False
trades = []

def execute_trade(row, long=True):
    global balance, trade_active
    risk_amount = balance * risk_per_trade
    entry_price = row['close']
    atr_value = row['atr']
    
    if pd.isna(atr_value):  # Prevent NaN errors
        print(f"Skipping trade at {row.name} due to NaN ATR")
        return
    
    stop_loss = entry_price - 2 * atr_value if long else entry_price + 2 * atr_value
    take_profit = entry_price + 3 * (entry_price - stop_loss) if long else entry_price - 3 * (stop_loss - entry_price)
    position_size = risk_amount / abs(entry_price - stop_loss)
    
    trades.append({'entry': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit, 'size': position_size, 'long': long, 'open_time': row.name, 'close_time': None, 'outcome': None, 'balance_after_trade': None})
    trade_active = True
    print(f"Trade opened: {'Long' if long else 'Short'} at {entry_price}, SL: {stop_loss}, TP: {take_profit}")

for index, row in oh5.iterrows():
    if trade_active:
        last_trade = trades[-1]
        if (last_trade['long'] and row['low'] <= last_trade['stop_loss']) or (not last_trade['long'] and row['high'] >= last_trade['stop_loss']):
            balance -= risk_per_trade * balance  # Stop loss hit
            last_trade['outcome'] = "Stop Loss"
            last_trade['balance_after_trade'] = balance
            last_trade['close_time'] = index
            trade_active = False
            print(f"Trade closed at stop loss: {last_trade['stop_loss']}")
        elif (last_trade['long'] and row['high'] >= last_trade['take_profit']) or (not last_trade['long'] and row['low'] <= last_trade['take_profit']):
            balance += risk_per_trade * balance * 3  # Take profit hit
            last_trade['outcome'] = "Take Profit"
            last_trade['balance_after_trade'] = balance
            last_trade['close_time'] = index
            trade_active = False
            print(f"Trade closed at take profit: {last_trade['take_profit']}")
    else:
        if row['trend'] and row['long_signal']:
            execute_trade(row, long=True)
        elif row['trend'] == False and row['short_signal']:
            execute_trade(row, long=False)

# Performance report
trades_df = pd.DataFrame(trades)
closed_trades = trades_df.dropna(subset=["stop_loss", "take_profit", "outcome", "balance_after_trade", "close_time"])  # Ensure only closed trades are counted

# Export the full DataFrame to an Excel file
closed_trades.to_excel("performance_report.xlsx", index=False)
print("Performance report exported to performance_report.xlsx")

print("Corrected Backtest Summary")
print(f"Total Trades: {len(closed_trades)}")
print(f"Final Balance: ${balance:.2f}")

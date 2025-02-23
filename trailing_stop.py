import pandas as pd
import numpy as np
from datetime import datetime
import xlsxwriter
# from fetch_data import asset
asset = "BTC"

def load_data(hour_file, minute_file):
    """
    Load 1H and 5M data from CSV files
    """
    df_1h = pd.read_csv(hour_file)
    df_5m = pd.read_csv(minute_file)
    
    # Convert timestamps to datetime objects if needed
    if 'timestamp' in df_1h.columns:
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
        df_1h.set_index('timestamp', inplace=True)
    
    if 'timestamp' in df_5m.columns:
        df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
        df_5m.set_index('timestamp', inplace=True)
    
    return df_1h, df_5m

def calculate_indicators(df_1h, df_5m):
    """
    Calculate all required indicators for both timeframes
    """
    # 1H Timeframe Indicators
    df_1h['200_ema'] = df_1h['close'].ewm(span=200, adjust=False).mean()
    df_1h['50_ema'] = df_1h['close'].ewm(span=50, adjust=False).mean()
    
    # Calculate ATR for 1H
    df_1h['tr'] = np.maximum(
        np.maximum(
            df_1h['high'] - df_1h['low'],
            abs(df_1h['high'] - df_1h['close'].shift(1))
        ),
        abs(df_1h['low'] - df_1h['close'].shift(1))
    )
    df_1h['atr_14'] = df_1h['tr'].rolling(window=14).mean()
    df_1h['atr_sma_20'] = df_1h['atr_14'].rolling(window=20).mean()
    
    # Calculate ADX for 1H
    df_1h['plus_dm'] = np.where(
        (df_1h['high'] - df_1h['high'].shift(1)) > (df_1h['low'].shift(1) - df_1h['low']),
        np.maximum(df_1h['high'] - df_1h['high'].shift(1), 0),
        0
    )
    df_1h['minus_dm'] = np.where(
        (df_1h['low'].shift(1) - df_1h['low']) > (df_1h['high'] - df_1h['high'].shift(1)),
        np.maximum(df_1h['low'].shift(1) - df_1h['low'], 0),
        0
    )
    
    df_1h['plus_di_14'] = 100 * (df_1h['plus_dm'].ewm(alpha=1/14, adjust=False).mean() / df_1h['atr_14'])
    df_1h['minus_di_14'] = 100 * (df_1h['minus_dm'].ewm(alpha=1/14, adjust=False).mean() / df_1h['atr_14'])
    df_1h['dx'] = 100 * abs(df_1h['plus_di_14'] - df_1h['minus_di_14']) / (df_1h['plus_di_14'] + df_1h['minus_di_14'])
    df_1h['adx'] = df_1h['dx'].ewm(alpha=1/14, adjust=False).mean()
    
    # 5M Timeframe Indicators
    df_5m['20_ema'] = df_5m['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate RSI for 5M
    delta = df_5m['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df_5m['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD for 5M
    df_5m['ema_12'] = df_5m['close'].ewm(span=12, adjust=False).mean()
    df_5m['ema_26'] = df_5m['close'].ewm(span=26, adjust=False).mean()
    df_5m['macd'] = df_5m['ema_12'] - df_5m['ema_26']
    df_5m['signal'] = df_5m['macd'].ewm(span=9, adjust=False).mean()
    df_5m['macd_histogram'] = df_5m['macd'] - df_5m['signal']
    
    return df_1h, df_5m

def is_optimal_trading_hour(current_time):
    """
    Check if current hour is within optimal trading hours
    """
    hour = current_time.hour
    
    # Optimal hours: 00:00-03:00, 08:00-11:00, 14:00-17:00 UTC
    if (0 <= hour < 3) or (8 <= hour < 11) or (14 <= hour < 17):
        return True
    return False

def check_1h_trend(row, atr_threshold):
    """
    Check if 1H timeframe meets trend conditions
    Returns: 1 for bullish trend, -1 for bearish trend, 0 for no trend
    """
    # Check for bullish trend
    if (row['close'] > row['200_ema'] and 
        row['50_ema'] > row['200_ema'] and
        row['atr_14'] > row['atr_sma_20'] and
        row['atr_14'] > atr_threshold and
        row['adx'] > 25):
        return 1
    
    # Check for bearish trend
    elif (row['close'] < row['200_ema'] and 
          row['50_ema'] < row['200_ema'] and
          row['atr_14'] > row['atr_sma_20'] and
          row['atr_14'] > atr_threshold and
          row['adx'] > 25):
        return -1
    
    # No clear trend or insufficient volatility
    return 0

def check_5m_entry(df_5m, i, trend_direction):
    """
    Check if 5M timeframe provides entry signal
    Returns: True if entry conditions are met, False otherwise
    """
    if i == 0:  # Prevent accessing index -1
        return False  

    row = df_5m.iloc[i]
    prev_row = df_5m.iloc[i - 1]

    if trend_direction == 1:  # Bullish trend
        bullish_candle = row['close'] > row['open']
        price_near_ema = abs(row['low'] - row['20_ema']) / row['20_ema'] < 0.003
        rsi_condition = 40 <= row['rsi'] <= 70
        macd_cross = row['macd_histogram'] > 0 and prev_row['macd_histogram'] <= 0

        return bullish_candle and price_near_ema and rsi_condition and macd_cross

    elif trend_direction == -1:  # Bearish trend
        bearish_candle = row['close'] < row['open']
        price_near_ema = abs(row['high'] - row['20_ema']) / row['20_ema'] < 0.003
        rsi_condition = 30 <= row['rsi'] <= 60
        macd_cross = row['macd_histogram'] < 0 and prev_row['macd_histogram'] >= 0

        return bearish_candle and price_near_ema and rsi_condition and macd_cross

    return False

def calculate_position_size(account_balance, entry_price, stop_price):
    """
    Calculate position size based on 1% risk rule
    """
    risk_amount = account_balance * 0.01
    risk_per_unit = abs(entry_price - stop_price)
    position_size = risk_amount / risk_per_unit
    return position_size, risk_amount

def find_swing_point(df_5m, periods=12, direction='low'):
    """
    Find the most recent swing low/high on 5M chart
    """
    window = 2 * periods + 1
    
    if direction == 'low':
        min_idx = df_5m['low'].rolling(window=window, center=True).apply(
            lambda x: np.argmin(x) == periods, raw=True
        )
        swing_points = df_5m[min_idx == 1].copy()  # Ensure it's a valid boolean filter
        return swing_points['low'].iloc[-1] if not swing_points.empty else None
    else:
        max_idx = df_5m['high'].rolling(window=window, center=True).apply(
            lambda x: np.argmax(x) == periods, raw=True
        )
        swing_points = df_5m[max_idx == 1].copy()
        return swing_points['high'].iloc[-1] if not swing_points.empty else None

def backtest_strategy(df_1h, df_5m, initial_balance=10000, atr_threshold=300):
    """
    Backtest the trading strategy
    """
    account_balance = initial_balance
    balance_history = [initial_balance]
    in_position = False
    position_type = None  # 'long' or 'short'
    entry_price = 0
    stop_loss = 0
    trailing_stop = 0
    position_size = 0
    risk_amount = 0
    trades = []
    
    # Align timestamps between 1H and 5M data
    df_1h_resampled = df_1h.resample('5T').ffill()
    
    # Merge datasets on time index
    combined_df = pd.merge_asof(
        df_5m, 
        df_1h_resampled[['200_ema', '50_ema', 'atr_14', 'atr_sma_20', 'adx']], 
        left_index=True, 
        right_index=True,
        direction='backward'
    )
    
    # Initialize values for tracking highest/lowest since entry
    highest_since_entry = 0
    lowest_since_entry = float('inf')
    
    for i in range(100, len(combined_df)):
        current_row = combined_df.iloc[i]
        current_time = combined_df.index[i]
        
        # Check if in optimal trading hours
        optimal_hour = is_optimal_trading_hour(current_time)
        
        # Check 1H trend direction
        trend_direction = check_1h_trend(current_row, atr_threshold)
        
        # If in position, update tracking values and check for exit
        if in_position:
            if position_type == 'long':
                # Update highest price since entry
                highest_since_entry = max(highest_since_entry, current_row['high'])
                
                # Update trailing stop (2 × 1H ATR below highest high)
                new_trailing_stop = highest_since_entry - (2 * current_row['atr_14'])
                trailing_stop = max(trailing_stop, new_trailing_stop)
                
                # Check for exit conditions
                if current_row['low'] <= trailing_stop or (trend_direction != 1 and optimal_hour):
                    # Exit long position
                    exit_price = min(trailing_stop, current_row['open'])  # Realistic fill
                    profit = (exit_price - entry_price) * position_size
                    account_balance += profit
                    
                    trades.append({
                        'position': 'Long',
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'exit_price': exit_price,
                        'position_size': position_size,
                        'risk_amount': risk_amount,
                        'pnl': profit,
                        'balance_after': account_balance,
                        'exit_reason': 'Trailing Stop' if current_row['low'] <= trailing_stop else 'Trend Reversal'
                    })
                    
                    in_position = False
                    balance_history.append(account_balance)
            
            elif position_type == 'short':
                # Update lowest price since entry
                lowest_since_entry = min(lowest_since_entry, current_row['low'])
                
                # Update trailing stop (2 × 1H ATR above lowest low)
                new_trailing_stop = lowest_since_entry + (2 * current_row['atr_14'])
                trailing_stop = min(trailing_stop if trailing_stop > 0 else float('inf'), new_trailing_stop)
                
                # Check for exit conditions
                if current_row['high'] >= trailing_stop or (trend_direction != -1 and optimal_hour):
                    # Exit short position
                    exit_price = max(trailing_stop, current_row['open'])  # Realistic fill
                    profit = (entry_price - exit_price) * position_size
                    account_balance += profit
                    
                    trades.append({
                        'position': 'Short',
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'exit_price': exit_price,
                        'position_size': position_size,
                        'risk_amount': risk_amount,
                        'pnl': profit,
                        'balance_after': account_balance,
                        'exit_reason': 'Trailing Stop' if current_row['high'] >= trailing_stop else 'Trend Reversal'
                    })
                    
                    in_position = False
                    balance_history.append(account_balance)
        
        # If not in position, check for entry conditions
        elif optimal_hour and not in_position:
            # Check 5M entry signal
            entry_signal = check_5m_entry(combined_df, i, trend_direction)
            
            if entry_signal and trend_direction == 1:  # Long entry
                entry_price = current_row['close']
                
                # Find recent swing low for stop loss
                recent_data = combined_df.iloc[i-20:i+1]
                swing_low = find_swing_point(recent_data, direction='low')
                if swing_low is None:
                    swing_low = recent_data['low'].min()
                
                # Set stop loss (use greater of swing low or 1.5 × ATR)
                atr_stop = entry_price - (1.5 * current_row['atr_14'])
                stop_loss = min(swing_low, atr_stop)
                
                # Calculate position size based on 1% risk
                position_size, risk_amount = calculate_position_size(account_balance, entry_price, stop_loss)
                
                # Set initial trailing stop
                trailing_stop = stop_loss
                highest_since_entry = entry_price
                
                in_position = True
                position_type = 'long'
                entry_time = current_time
            
            elif entry_signal and trend_direction == -1:  # Short entry
                entry_price = current_row['close']
                
                # Find recent swing high for stop loss
                recent_data = combined_df.iloc[i-20:i+1]
                swing_high = find_swing_point(recent_data, direction='high')
                if swing_high is None:
                    swing_high = recent_data['high'].max()
                
                # Set stop loss (use lesser of swing high or 1.5 × ATR)
                atr_stop = entry_price + (1.5 * current_row['atr_14'])
                stop_loss = max(swing_high, atr_stop)
                
                # Calculate position size based on 1% risk
                position_size, risk_amount = calculate_position_size(account_balance, entry_price, stop_loss)
                
                # Set initial trailing stop
                trailing_stop = stop_loss
                lowest_since_entry = entry_price
                
                in_position = True
                position_type = 'short'
                entry_time = current_time
    
    # Calculate strategy metrics
    trade_df = pd.DataFrame(trades)
    
    if len(trade_df) > 0:
        win_rate = len(trade_df[trade_df['pnl'] > 0]) / len(trade_df) * 100
        
        metrics = {
            'Initial Balance': initial_balance,
            'Final Balance': account_balance,
            'Total Return': (account_balance - initial_balance) / initial_balance * 100,
            'Total Trades': len(trade_df),
            'Win Rate': win_rate
        }
    else:
        metrics = {
            'Initial Balance': initial_balance,
            'Final Balance': account_balance,
            'Total Return': 0,
            'Total Trades': 0,
            'Win Rate': 0
        }
    
    return trade_df, metrics, balance_history

def export_to_excel(trades_df, metrics, output_file=f"{asset}_results.xlsx"):
    """
    Export backtest results to Excel
    """
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Create trade results worksheet
        worksheet = workbook.add_worksheet('Trade Results')
        
        # Add header with formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Define headers
        headers = [
            'Position (Long/Short)', 
            'Entry Time', 
            'Exit Time', 
            'Entry Price ($)', 
            'Stop Loss ($)', 
            'Exit Price ($)',
            'Position Size', 
            'Risk Amount ($)', 
            'PnL ($)', 
            'Balance After Trade ($)'
        ]
        
        # Write header row
        for col_num, header in enumerate(headers):
            worksheet.write(0, col_num, header, header_format)
        
        # Format for currency
        currency_format = workbook.add_format({'num_format': '$#,##0.00'})
        coin_format = workbook.add_format({'num_format': '0.00000000'})
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd hh:mm'})
        
        # Conditional formatting for profit/loss
        profit_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        loss_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        
        # Write trade data
        if not trades_df.empty:
            for row_num, trade in enumerate(trades_df.to_dict('records'), 1):
                worksheet.write(row_num, 0, trade['position'])
                worksheet.write_datetime(row_num, 1, trade['entry_time'].to_pydatetime(), date_format)
                worksheet.write_datetime(row_num, 2, trade['exit_time'].to_pydatetime(), date_format)
                worksheet.write_number(row_num, 3, trade['entry_price'], currency_format)
                worksheet.write_number(row_num, 4, trade['stop_loss'], currency_format)
                worksheet.write_number(row_num, 5, trade['exit_price'], currency_format)
                worksheet.write_number(row_num, 6, trade['position_size'], coin_format)
                worksheet.write_number(row_num, 7, trade['risk_amount'], currency_format)
                
                # Apply conditional formatting to PnL
                if trade['pnl'] > 0:
                    worksheet.write_number(row_num, 8, trade['pnl'], profit_format)
                else:
                    worksheet.write_number(row_num, 8, trade['pnl'], loss_format)
                
                worksheet.write_number(row_num, 9, trade['balance_after'], currency_format)
        
        # Auto-size columns for better readability
        for i, _ in enumerate(headers):
            worksheet.set_column(i, i, 15)
        
        # Add summary section with border
        summary_start_row = len(trades_df) + 3 if not trades_df.empty else 3
        
        summary_header_format = workbook.add_format({
            'bold': True,
            'fg_color': '#4F81BD',
            'font_color': 'white',
            'border': 1
        })
        
        summary_cell_format = workbook.add_format({
            'border': 1
        })
        
        summary_currency_format = workbook.add_format({
            'num_format': '$#,##0.00',
            'border': 1
        })
        
        summary_percent_format = workbook.add_format({
            'num_format': '0.00%',
            'border': 1
        })
        
        # Write summary section header
        worksheet.merge_range(summary_start_row, 0, summary_start_row, 9, 'Strategy Summary', summary_header_format)
        
        # Write summary data
        summary_data = [
            ('Initial Balance', metrics['Initial Balance'], summary_currency_format),
            ('Final Balance', metrics['Final Balance'], summary_currency_format),
            ('Total Return', metrics['Total Return']/100, summary_percent_format),
            ('Total Trades', metrics['Total Trades'], summary_cell_format),
            ('Win Rate', metrics['Win Rate']/100, summary_percent_format)
        ]
        
        for i, (label, value, format_obj) in enumerate(summary_data):
            row = summary_start_row + i + 1
            worksheet.write(row, 0, label, summary_cell_format)
            worksheet.write(row, 1, value, format_obj)
            # Merge empty cells to create a border
            worksheet.merge_range(row, 2, row, 9, '', summary_cell_format)
        
        # Create a chart showing the equity curve
        if 'equity_data' in globals() and len(equity_data) > 0:
            chart_sheet = workbook.add_worksheet('Equity Curve')
            
            # Add equity data
            chart_sheet.write(0, 0, 'Trade Number', header_format)
            chart_sheet.write(0, 1, 'Account Balance', header_format)
            
            for i, balance in enumerate(equity_data):
                chart_sheet.write(i+1, 0, i)
                chart_sheet.write(i+1, 1, balance, currency_format)
            
            # Create the chart
            chart = workbook.add_chart({'type': 'line'})
            chart.add_series({
                'name': 'Account Balance',
                'categories': ['Equity Curve', 1, 0, len(equity_data), 0],
                'values': ['Equity Curve', 1, 1, len(equity_data), 1],
                'line': {'color': 'blue', 'width': 2},
            })
            
            chart.set_title({'name': 'Equity Curve'})
            chart.set_x_axis({'name': 'Trade Number'})
            chart.set_y_axis({'name': 'Account Balance ($)'})
            
            chart.set_size({'width': 720, 'height': 400})
            chart_sheet.insert_chart('D2', chart)

def main(hour_file_path, minute_file_path, initial_balance=10000):
    """
    Main function to run the strategy
    """
    # Load data
    print(f"Loading data from {hour_file_path} and {minute_file_path}...")
    df_1h, df_5m = load_data(hour_file_path, minute_file_path)
    
    # Calculate indicators
    print("Calculating indicators...")
    df_1h, df_5m = calculate_indicators(df_1h, df_5m)
    
    # Set ATR threshold (e.g., 0.5% of price)
    avg_price = df_1h['close'].mean()
    atr_threshold = avg_price * 0.005  # 0.5% of average price
    
    # Run backtest
    print(f"Running backtest with ATR threshold of ${atr_threshold:.2f}...")
    trades, metrics, balance_history = backtest_strategy(df_1h, df_5m, initial_balance, atr_threshold)
    
    # Store balance history for chart
    global equity_data
    equity_data = balance_history
    
    # Display summary in console
    print("\n=== Strategy Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Export results to Excel
    output_file = f"{asset}_results.xlsx"
    print(f"\nExporting results to {output_file}...")
    export_to_excel(trades, metrics, output_file)
    print(f"Excel report generated successfully!")

if __name__ == "__main__":
    hour_file_path = f"{asset}USDT_1h.csv"
    minute_file_path = f"{asset}USDT_5m.csv"
    main(hour_file_path, minute_file_path)
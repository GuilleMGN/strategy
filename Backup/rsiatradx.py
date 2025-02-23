import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import os

# Function to calculate indicators (unchanged)
def calculate_indicators(data):
    # Calculate ATR (Average True Range)
    data['atr'] = ta.volatility.average_true_range(high=data['high'], 
                                                 low=data['low'], 
                                                 close=data['close'], 
                                                 window=14)
    
    # Calculate ADX (Average Directional Index)
    adx_indicator = ta.trend.ADXIndicator(high=data['high'], 
                                         low=data['low'], 
                                         close=data['close'], 
                                         window=14)
    data['adx'] = adx_indicator.adx()
    data['+di'] = adx_indicator.adx_pos()
    data['-di'] = adx_indicator.adx_neg()
    
    # Calculate RSI (Relative Strength Index)
    data['rsi'] = ta.momentum.rsi(data['close'], window=9)
    
    # Calculate 10-period SMA
    data['sma_10'] = ta.trend.sma_indicator(data['close'], window=10)
    
    # Volatility filter - ATR value check - MODIFIED from $1000 to $800
    data['atr_filter'] = data['atr'] > 800  # $800 minimum volatility (UPDATED)
    
    return data

# Function to identify trade signals - MODIFIED with adjusted parameters
def identify_signals(data):
    # Initialize columns
    data['long_signal'] = False
    data['short_signal'] = False
    
    # Previous RSI values for crossover detection
    data['prev_rsi'] = data['rsi'].shift(1)
    
    for i in range(1, len(data)):
        # Long entry conditions - MODIFIED
        long_condition1 = data['adx'].iloc[i] > 25  # ADX threshold maintained at 25
        long_condition2 = data['rsi'].iloc[i-1] < 35 and data['rsi'].iloc[i] >= 35  # RSI crosses above 35 (lowered from 40)
        long_condition3 = data['close'].iloc[i] > data['sma_10'].iloc[i]  # Price above 10 SMA
        long_condition4 = data['atr_filter'].iloc[i]  # Volatility filter condition
        
        # Short entry conditions - MODIFIED
        short_condition1 = data['adx'].iloc[i] > 25  # ADX threshold maintained at 25
        short_condition2 = data['rsi'].iloc[i-1] > 65 and data['rsi'].iloc[i] <= 65  # RSI crosses below 65 (raised from 60)
        short_condition3 = data['close'].iloc[i] < data['sma_10'].iloc[i]  # Price below 10 SMA
        short_condition4 = data['atr_filter'].iloc[i]  # Volatility filter condition
        
        # Set signals
        data.loc[data.index[i], 'long_signal'] = long_condition1 and long_condition2 and long_condition3 and long_condition4
        data.loc[data.index[i], 'short_signal'] = short_condition1 and short_condition2 and short_condition3 and short_condition4
    
    return data

# Function to execute backtest - MODIFIED to remove RSI exit
def backtest_strategy(data, initial_balance=10000.00, risk_percent=0.015):  # Maintained 1.5% risk
    trades = []
    balance = initial_balance
    position = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    position_size = 0
    entry_time = None
    risk_amount = 0
    max_balance = initial_balance
    max_drawdown = 0
    drawdown_percent = 0
    
    for i in range(1, len(data)):
        current_data = data.iloc[i]
        prev_data = data.iloc[i-1]
        
        # Update max balance and drawdown
        if balance > max_balance:
            max_balance = balance
        current_drawdown = (max_balance - balance) / max_balance * 100 if max_balance > 0 else 0
        drawdown_percent = max(drawdown_percent, current_drawdown)
        
        # Check if we're in a position
        if position:
            # Check exit conditions
            exit_triggered = False
            exit_price = current_data['close']
            exit_reason = "Other"
            
            # Take profit check - Maintained at 4x ATR
            if (position == "Long" and current_data['high'] >= take_profit) or \
               (position == "Short" and current_data['low'] <= take_profit):
                exit_triggered = True
                exit_price = take_profit
                exit_reason = "Take Profit"
            
            # Stop loss check - Maintained at 1x ATR
            elif (position == "Long" and current_data['low'] <= stop_loss) or \
                 (position == "Short" and current_data['high'] >= stop_loss):
                exit_triggered = True
                exit_price = stop_loss
                exit_reason = "Stop Loss"
            
            # RSI exit condition removed as requested
            
            # If exit is triggered, close the position
            if exit_triggered:
                pnl = 0
                if position == "Long":
                    pnl = (exit_price - entry_price) * position_size
                elif position == "Short":
                    pnl = (entry_price - exit_price) * position_size
                
                balance += pnl
                
                trades.append({
                    'Position': position,
                    'Entry Time': entry_time,
                    'Exit Time': data.index[i],
                    'Entry Price': entry_price,
                    'Stop Loss': stop_loss,
                    'Take Profit': take_profit,
                    'Exit Price': exit_price,
                    'Position Size': position_size,
                    'Risk Amount': risk_amount,
                    'PnL': pnl,
                    'Balance After Trade': balance,
                    'Exit Reason': exit_reason
                })
                
                position = None
        
        # Check for new trade signals if we're not in a position
        if not position:
            # Long signal
            if current_data['long_signal']:
                position = "Long"
                entry_price = current_data['close']
                entry_time = data.index[i]
                
                # Calculate ATR for stop loss and take profit - Maintained
                atr_value = current_data['atr']
                stop_loss = entry_price - (1.0 * atr_value)  # Maintained at 1x ATR
                take_profit = entry_price + (4.0 * atr_value)  # Maintained at 4x ATR
                
                # Calculate position size with 1.5% risk
                risk_amount = balance * risk_percent
                position_size = risk_amount / (1.0 * atr_value)
                position_size = min(position_size, balance / entry_price)  # Ensure we don't exceed balance
            
            # Short signal
            elif current_data['short_signal']:
                position = "Short"
                entry_price = current_data['close']
                entry_time = data.index[i]
                
                # Calculate ATR for stop loss and take profit - Maintained
                atr_value = current_data['atr']
                stop_loss = entry_price + (1.0 * atr_value)  # Maintained at 1x ATR
                take_profit = entry_price - (4.0 * atr_value)  # Maintained at 4x ATR
                
                # Calculate position size with 1.5% risk
                risk_amount = balance * risk_percent
                position_size = risk_amount / (1.0 * atr_value)
                position_size = min(position_size, balance / entry_price)  # Ensure we don't exceed balance
    
    # Close any open position at the end of the test period
    if position:
        exit_price = data.iloc[-1]['close']
        pnl = 0
        if position == "Long":
            pnl = (exit_price - entry_price) * position_size
        elif position == "Short":
            pnl = (entry_price - exit_price) * position_size
        
        balance += pnl
        
        trades.append({
            'Position': position,
            'Entry Time': entry_time,
            'Exit Time': data.index[-1],
            'Entry Price': entry_price,
            'Stop Loss': stop_loss,
            'Take Profit': take_profit,
            'Exit Price': exit_price,
            'Position Size': position_size,
            'Risk Amount': risk_amount,
            'PnL': pnl,
            'Balance After Trade': balance,
            'Exit Reason': "End of Test Period"
        })
    
    return trades, balance, drawdown_percent

# Function to generate summary statistics (unchanged)
def generate_summary(trades, initial_balance, final_balance, max_drawdown):
    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade['PnL'] > 0)
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    
    # Additional statistics
    total_profit = sum(trade['PnL'] for trade in trades if trade['PnL'] > 0)
    total_loss = sum(trade['PnL'] for trade in trades if trade['PnL'] < 0)
    profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
    
    # Calculate average trade duration
    if trades and isinstance(trades[0]['Entry Time'], pd.Timestamp) and isinstance(trades[0]['Exit Time'], pd.Timestamp):
        durations = [(trade['Exit Time'] - trade['Entry Time']).total_seconds()/3600 for trade in trades]
        avg_duration = sum(durations) / len(durations) if durations else 0
    else:
        avg_duration = "N/A"
    
    # Risk-reward ratio based on average win vs average loss
    avg_win = total_profit / winning_trades if winning_trades > 0 else 0
    avg_loss = abs(total_loss / (total_trades - winning_trades)) if (total_trades - winning_trades) > 0 else 0
    risk_reward = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # Analysis by exit type
    exit_types = {}
    for trade in trades:
        exit_reason = trade['Exit Reason']
        if exit_reason not in exit_types:
            exit_types[exit_reason] = {'count': 0, 'profit': 0}
        exit_types[exit_reason]['count'] += 1
        exit_types[exit_reason]['profit'] += trade['PnL']
    
    # Prepare exit type statistics
    exit_stats = {}
    for reason, stats in exit_types.items():
        exit_stats[f"{reason} Count"] = stats['count']
        exit_stats[f"{reason} P/L"] = f"${stats['profit']:.2f}"
    
    summary = {
        'Initial Balance': initial_balance,
        'Final Balance': final_balance,
        'Total Return': f"{total_return:.2f}%",
        'Max Drawdown': f"{max_drawdown:.2f}%",
        'Total Trades': total_trades,
        'Win Rate': f"{win_rate:.2f}%",
        'Profit Factor': f"{profit_factor:.2f}",
        'Risk-Reward Ratio': f"{risk_reward:.2f}",
        'Average Trade P/L': f"${sum(trade['PnL'] for trade in trades) / total_trades:.2f}" if total_trades > 0 else "$0.00",
        'Average Trade Duration (hours)': f"{avg_duration:.1f}" if isinstance(avg_duration, (int, float)) else avg_duration,
        **exit_stats
    }
    
    return summary

# Function to save results to Excel (unchanged)
def save_to_excel(trades, summary, output_file='bitcoin_strategy_results_updated.xlsx'):
    # Create a DataFrame from the trades list
    trades_df = pd.DataFrame(trades)
    
    # Calculate additional metrics for each trade
    if 'Entry Price' in trades_df.columns and 'Exit Price' in trades_df.columns:
        trades_df['Return %'] = trades_df.apply(
            lambda row: ((row['Exit Price'] - row['Entry Price']) / row['Entry Price'] * 100) if row['Position'] == 'Long' 
            else ((row['Entry Price'] - row['Exit Price']) / row['Entry Price'] * 100), axis=1
        )
    
    if 'Risk Amount' in trades_df.columns and 'PnL' in trades_df.columns:
        trades_df['R Multiple'] = trades_df.apply(
            lambda row: row['PnL'] / abs(row['Risk Amount']) if row['Risk Amount'] != 0 else 0, axis=1
        )
    
    # Calculate duration if timestamps are available
    if 'Entry Time' in trades_df.columns and 'Exit Time' in trades_df.columns:
        if pd.api.types.is_datetime64_any_dtype(trades_df['Entry Time']) and pd.api.types.is_datetime64_any_dtype(trades_df['Exit Time']):
            trades_df['Duration (hours)'] = (trades_df['Exit Time'] - trades_df['Entry Time']).dt.total_seconds() / 3600
    
    # Format the numeric columns
    numeric_cols = ['Entry Price', 'Stop Loss', 'Take Profit', 'Exit Price', 'Position Size', 
                    'Risk Amount', 'PnL', 'Balance After Trade']
    for col in numeric_cols:
        if col in trades_df.columns:
            trades_df[col] = trades_df[col].map(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else x)
    
    if 'Return %' in trades_df.columns:
        trades_df['Return %'] = trades_df['Return %'].map(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
    
    if 'R Multiple' in trades_df.columns:
        trades_df['R Multiple'] = trades_df['R Multiple'].map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    if 'Duration (hours)' in trades_df.columns:
        trades_df['Duration (hours)'] = trades_df['Duration (hours)'].map(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x)
    
    # Create a DataFrame from the summary dictionary
    summary_df = pd.DataFrame([summary])
    
    # Format the summary values
    summary_df['Initial Balance'] = summary_df['Initial Balance'].map(lambda x: f"${x:.2f}")
    summary_df['Final Balance'] = summary_df['Final Balance'].map(lambda x: f"${x:.2f}")
    
    # Create equity curve
    equity_curve = None
    if trades:
        # Start with initial balance
        initial_balance = trades[0]['Balance After Trade'] - trades[0]['PnL']
        equity_points = [{'Date': trades[0]['Entry Time'], 'Equity': initial_balance}]
        
        # Add each trade's effect on equity
        for trade in trades:
            equity_points.append({
                'Date': trade['Exit Time'],
                'Equity': trade['Balance After Trade']
            })
        
        equity_curve = pd.DataFrame(equity_points)
        equity_curve['Drawdown %'] = 100 * (1 - equity_curve['Equity'] / equity_curve['Equity'].cummax())
    
    # Create a Pandas Excel writer using openpyxl
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Write the trades DataFrame to the Excel file
        trades_df.to_excel(writer, sheet_name='Trades', index=False)
        
        # Write the summary DataFrame to the Excel file
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Write equity curve if available
        if equity_curve is not None:
            equity_curve.to_excel(writer, sheet_name='Equity Curve', index=False)
    
    print(f"Results saved to {output_file}")

# Main function - MODIFIED to handle 1-hour data
def main():
    # File path to your 1h Bitcoin data csv (modified from 4h)
    data_file = 'BTCUSDT_1h.csv'  # UPDATED to 1h timeframe
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Error: The file {data_file} does not exist.")
        return
    
    # Load the data
    try:
        # Assuming your CSV has columns: open, high, low, close, volume, and datetime
        data = pd.read_csv(data_file)
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
        
        # Make sure the column names are lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Make sure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                print(f"Error: Required column '{col}' not found in the data.")
                return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Calculate indicators
    data = calculate_indicators(data)
    
    # Identify signals
    data = identify_signals(data)
    
    # Execute backtest with updated parameters
    initial_balance = 10000.00
    risk_percent = 0.015  # Maintained at 1.5%
    trades, final_balance, max_drawdown = backtest_strategy(data, initial_balance, risk_percent)
    
    # Generate summary
    summary = generate_summary(trades, initial_balance, final_balance, max_drawdown)
    
    # Save results to Excel
    save_to_excel(trades, summary, 'bitcoin_strategy_high_frequency_v1.xlsx')
    
    # Print summary to console
    print("High-Frequency Strategy Backtest Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Print strategy performance metrics
    if trades:
        win_count = sum(1 for trade in trades if trade['PnL'] > 0)
        loss_count = sum(1 for trade in trades if trade['PnL'] < 0)
        avg_win = sum(trade['PnL'] for trade in trades if trade['PnL'] > 0) / win_count if win_count > 0 else 0
        avg_loss = sum(trade['PnL'] for trade in trades if trade['PnL'] < 0) / loss_count if loss_count > 0 else 0
        
        print("\nDetailed Performance Metrics:")
        print(f"Total Winning Trades: {win_count}")
        print(f"Total Losing Trades: {loss_count}")
        print(f"Average Winning Trade: ${avg_win:.2f}")
        print(f"Average Losing Trade: ${avg_loss:.2f}")
        
        if avg_loss != 0:
            win_loss_ratio = abs(avg_win / avg_loss)
            print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
        
        # Print updated strategy metrics
        print("\nHigh-Frequency Strategy Parameters:")
        print("- Timeframe: 1-hour (changed from 4-hour)")
        print("- Take-Profit: 4x ATR (maintained)")
        print("- Stop-Loss: 1x ATR (maintained)")
        print("- Risk per trade: 1.5% (maintained)")
        print("- ADX threshold: > 25 (maintained)")
        print("- RSI entry thresholds: > 35 for longs, < 65 for shorts (widened from 40/60)")
        print("- Volatility filter: ATR > $800 (lowered from $1000)")
        print("- RSI exit condition: Removed")
        
        # Calculate average R-multiple (return in terms of risk)
        r_multiples = [(trade['PnL'] / trade['Risk Amount']) for trade in trades if trade['Risk Amount'] != 0]
        avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0
        print(f"Average R-Multiple: {avg_r:.2f}R")

if __name__ == "__main__":
    main()
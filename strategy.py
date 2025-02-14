import pandas as pd
import numpy as np
import ta
from datetime import datetime

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate Supertrend indicator"""
    
    # Calculate ATR
    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
    
    # Calculate Basic Upper and Lower Bands
    basic_upper = (df['high'] + df['low']) / 2 + (multiplier * atr)
    basic_lower = (df['high'] + df['low']) / 2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = pd.Series(index=df.index, dtype='float64')
    direction = pd.Series(index=df.index, dtype='float64')
    
    # Set first value
    supertrend.iloc[0] = basic_upper.iloc[0]
    direction.iloc[0] = 1
    
    # Calculate Supertrend
    for i in range(1, len(df)):
        if df['close'].iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = max(basic_lower.iloc[i], supertrend.iloc[i-1])
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = min(basic_upper.iloc[i], supertrend.iloc[i-1])
            direction.iloc[i] = -1
            
    return supertrend, direction

class BTCTradeBacktester:
    def __init__(self, initial_balance=10000.0, risk_percentage=0.01):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_percentage = risk_percentage
        self.trades = []
        self.current_trade = None
        
    def load_data(self, path_1h, path_5m):
        """Load and prepare data from CSV files"""
        # Load data
        self.df_1h = pd.read_csv(path_1h)
        self.df_5m = pd.read_csv(path_5m)
        
        # Convert timestamps
        self.df_1h['timestamp'] = pd.to_datetime(self.df_1h['timestamp'])
        self.df_5m['timestamp'] = pd.to_datetime(self.df_5m['timestamp'])
        
        # Calculate indicators for 1h timeframe
        self.df_1h['ema_50'] = ta.trend.ema_indicator(self.df_1h['close'], window=50)
        self.df_1h['ema_200'] = ta.trend.ema_indicator(self.df_1h['close'], window=200)
        
        # Calculate Supertrend
        self.df_1h['supertrend'], self.df_1h['supertrend_direction'] = calculate_supertrend(
            self.df_1h, period=10, multiplier=3
        )
        
        # Calculate indicators for 5m timeframe
        self.df_5m['rsi'] = ta.momentum.rsi(self.df_5m['close'], window=14)
        self.df_5m['stoch_k'] = ta.momentum.stoch(self.df_5m['high'], 
                                                 self.df_5m['low'], 
                                                 self.df_5m['close'], 
                                                 window=14, 
                                                 smooth_window=3)
        self.df_5m['stoch_d'] = ta.momentum.stoch_signal(self.df_5m['high'], 
                                                        self.df_5m['low'], 
                                                        self.df_5m['close'], 
                                                        window=14, 
                                                        smooth_window=3)
        self.df_5m['atr'] = ta.volatility.average_true_range(self.df_5m['high'], 
                                                            self.df_5m['low'], 
                                                            self.df_5m['close'], 
                                                            window=14)
    
    def check_1h_trend(self, timestamp):
        """Check 1h trend at given timestamp"""
        hour_data = self.df_1h[self.df_1h['timestamp'] <= timestamp].iloc[-1]
        
        bullish = (hour_data['ema_50'] > hour_data['ema_200'] and 
                  hour_data['supertrend_direction'] == 1)
        bearish = (hour_data['ema_50'] < hour_data['ema_200'] and 
                  hour_data['supertrend_direction'] == -1)
        
        return 'bullish' if bullish else 'bearish' if bearish else None
    
    def check_5m_entry(self, row, trend):
        """Check 5m entry conditions"""
        if trend == 'bullish':
            return ('long' if row['rsi'] < 40 and 
                   row['stoch_k'] < 20 and 
                   row['stoch_k'] > row['stoch_d'] else None)
        elif trend == 'bearish':
            return ('short' if row['rsi'] > 60 and 
                   row['stoch_k'] > 80 and 
                   row['stoch_k'] < row['stoch_d'] else None)
        return None
    
    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on risk"""
        risk_amount = self.current_balance * self.risk_percentage
        price_risk = abs(entry_price - stop_loss)
        return risk_amount / price_risk
    
    def run_backtest(self):
        """Run the backtest"""
        for index, row in self.df_5m.iterrows():
            # Skip if not enough data for indicators
            if index < 200:
                continue
                
            current_time = row['timestamp'].time()
            # Check if within trading hours (13:00-21:00 UTC)
            if not (13 <= current_time.hour <= 21):
                continue
            
            # Check for open trade
            if self.current_trade:
                # Check if stop loss or take profit hit
                if self.current_trade['type'] == 'long':
                    if row['low'] <= self.current_trade['stop_loss']:
                        self.close_trade(row, 'stop_loss')
                    elif row['high'] >= self.current_trade['take_profit']:
                        self.close_trade(row, 'take_profit')
                else:  # short
                    if row['high'] >= self.current_trade['stop_loss']:
                        self.close_trade(row, 'stop_loss')
                    elif row['low'] <= self.current_trade['take_profit']:
                        self.close_trade(row, 'take_profit')
            else:
                # Check for new trade
                trend = self.check_1h_trend(row['timestamp'])
                entry_signal = self.check_5m_entry(row, trend)
                
                if entry_signal:
                    entry_price = row['close']
                    stop_loss = (entry_price - (2 * row['atr']) if entry_signal == 'long' 
                               else entry_price + (2 * row['atr']))
                    # Modified take profit for 1:3 risk-reward ratio
                    take_profit = (entry_price + (6 * row['atr']) if entry_signal == 'long'
                                 else entry_price - (6 * row['atr']))
                    
                    position_size = self.calculate_position_size(entry_price, stop_loss)
                    
                    self.current_trade = {
                        'type': entry_signal,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'position_size': position_size,
                        'entry_time': row['timestamp'],
                        'entry_balance': self.current_balance
                    }

    def close_trade(self, row, exit_type):
        """Close a trade and update balance"""
        exit_price = (row['low'] if exit_type == 'stop_loss' else row['high'] 
                     if self.current_trade['type'] == 'long' else 
                     row['high'] if exit_type == 'stop_loss' else row['low'])
    
        price_change = (exit_price - self.current_trade['entry_price'] 
                       if self.current_trade['type'] == 'long' 
                       else self.current_trade['entry_price'] - exit_price)
    
        pnl = price_change * self.current_trade['position_size']
        self.current_balance += pnl
    
        trade_record = {
            **self.current_trade,
            'exit_price': exit_price,
            'exit_time': row['timestamp'],
            'exit_type': exit_type,
            'pnl': pnl,
            'exit_balance': self.current_balance
        }

        self.trades.append(trade_record)
        self.current_trade = None

    def generate_report(self):
        """Generate performance report"""
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        print("\n=== BACKTEST PERFORMANCE REPORT ===")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.current_balance:,.2f}")
        print(f"Total Return: {((self.current_balance/self.initial_balance - 1) * 100):.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
    
        print("\n=== DETAILED TRADE HISTORY ===")
        for i, trade in enumerate(self.trades, 1):
            print(f"\nTrade #{i}")
            print(f"Type: {trade['type'].upper()}")
            print(f"Entry Time: {trade['entry_time']}")
            print(f"Exit Time: {trade['exit_time']}")
            print(f"Entry Price: ${trade['entry_price']:,.2f}")
            print(f"Exit Price: ${trade['exit_price']:,.2f}")
            print(f"Stop Loss: ${trade['stop_loss']:,.2f}")
            print(f"Take Profit: ${trade['take_profit']:,.2f}")
            print(f"Position Size: {trade['position_size']:.4f}")
            print(f"PnL: ${trade['pnl']:,.2f}")
            print(f"Exit Type: {trade['exit_type'].upper()}")

    def export_to_excel(self, filename='trading_results.xlsx'):
        """Export trade results to Excel"""
        if not self.trades:
            print("No trades to export")
            return
        
        # Create DataFrame from trades
        trades_data = []
        for trade in self.trades:
            trade_data = {
                'Position': trade['type'].upper(),
                'Entry Time': trade['entry_time'],
                'Exit Time': trade['exit_time'],
                'Entry Price': trade['entry_price'],
                'Stop Loss': trade['stop_loss'],
                'Take Profit': trade['take_profit'],
                'Exit Price': trade['exit_price'],
                'Position Size': trade['position_size'],
                'Outcome': trade['exit_type'].upper(),
                'PnL ($)': trade['pnl'],
                'Balance After Trade': trade['exit_balance'],
                'Risk Amount ($)': trade['entry_balance'] * self.risk_percentage,
                'Risk-Reward Ratio': '1:3',
                'Price Risk (Points)': abs(trade['entry_price'] - trade['stop_loss']),
                'Potential Reward (Points)': abs(trade['take_profit'] - trade['entry_price'])
            }
            trades_data.append(trade_data)
        
        df_trades = pd.DataFrame(trades_data)
        
        # Create Excel writer object
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            # Write trades to sheet
            df_trades.to_excel(writer, sheet_name='Trade History', index=True)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Trade History']
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D3D3D3',
                'border': 1
            })
            
            cell_format = workbook.add_format({
                'border': 1
            })
            
            money_format = workbook.add_format({
                'border': 1,
                'num_format': '$#,##0.00'
            })
            
            percent_format = workbook.add_format({
                'border': 1,
                'num_format': '0.00%'
            })
            
            # Format headers
            for col_num, value in enumerate(df_trades.columns.values):
                worksheet.write(0, col_num + 1, value, header_format)
            
            # Format columns
            worksheet.set_column('B:C', 20)  # Position column
            worksheet.set_column('D:E', 25)  # Time columns
            worksheet.set_column('F:I', 15, money_format)  # Price columns
            worksheet.set_column('J:J', 15)  # Position Size
            worksheet.set_column('K:K', 15)  # Outcome
            worksheet.set_column('L:L', 15, money_format)  # PnL
            worksheet.set_column('M:M', 20, money_format)  # Balance
            
            # Add summary statistics
            summary_start_row = len(df_trades) + 4
            worksheet.write(summary_start_row, 0, 'Summary Statistics', header_format)
            worksheet.write(summary_start_row + 1, 0, 'Initial Balance:')
            worksheet.write(summary_start_row + 1, 1, self.initial_balance, money_format)
            worksheet.write(summary_start_row + 2, 0, 'Final Balance:')
            worksheet.write(summary_start_row + 2, 1, self.current_balance, money_format)
            worksheet.write(summary_start_row + 3, 0, 'Total Return:')
            worksheet.write(summary_start_row + 3, 1, 
                          (self.current_balance/self.initial_balance - 1), percent_format)
            worksheet.write(summary_start_row + 4, 0, 'Total Trades:')
            worksheet.write(summary_start_row + 4, 1, len(self.trades))
            worksheet.write(summary_start_row + 5, 0, 'Win Rate:')
            win_rate = len([t for t in self.trades if t['pnl'] > 0]) / len(self.trades)
            worksheet.write(summary_start_row + 5, 1, win_rate, percent_format)
            
        print(f"Results exported to {filename}")

# Modified usage example
if __name__ == "__main__":
    backtest = BTCTradeBacktester(initial_balance=10000.0, risk_percentage=0.01)
    backtest.load_data("BTCUSDT_1h.csv", "BTCUSDT_5m.csv")
    backtest.run_backtest()
    backtest.generate_report()
    backtest.export_to_excel()  # This will create trading_results.xlsx
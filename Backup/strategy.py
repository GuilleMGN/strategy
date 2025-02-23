import pandas as pd
import numpy as np
import ta
from datetime import datetime
# from fetch_data import asset
asset = "BTC"

def calculate_ichimoku(df):
    """Calculate Ichimoku Cloud indicators"""
    df['tenkan_sen'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
    return df

class Backtester:
    def __init__(self, initial_balance=20000.0, risk_percentage=0.01, leverage=10):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_percentage = risk_percentage
        self.leverage = leverage
        self.trades = []
        self.current_trade = None

    def load_data(self, path_1h, path_5m):
        self.df_1h = pd.read_csv(path_1h)
        self.df_5m = pd.read_csv(path_5m)
        
        self.df_1h['timestamp'] = pd.to_datetime(self.df_1h['timestamp']).dt.tz_localize('UTC')
        self.df_5m['timestamp'] = pd.to_datetime(self.df_5m['timestamp']).dt.tz_localize('UTC')
        
        self.df_1h = calculate_ichimoku(self.df_1h)
        
        self.df_5m['ema_5'] = ta.trend.ema_indicator(self.df_5m['close'], window=5)
        self.df_5m['ema_8'] = ta.trend.ema_indicator(self.df_5m['close'], window=8)
        self.df_5m['ema_13'] = ta.trend.ema_indicator(self.df_5m['close'], window=13)
        self.df_5m['rsi'] = ta.momentum.rsi(self.df_5m['close'], window=14)
        self.df_5m['atr'] = ta.volatility.average_true_range(self.df_5m['high'], self.df_5m['low'], self.df_5m['close'], window=14)
    
    def check_1h_trend(self, timestamp):
        timestamp = timestamp.tz_localize('UTC') if timestamp.tzinfo is None else timestamp
        hour_data = self.df_1h[self.df_1h['timestamp'] <= timestamp].iloc[-1]
        bullish = hour_data['close'] > hour_data['senkou_span_a'] and hour_data['close'] > hour_data['senkou_span_b']
        bearish = hour_data['close'] < hour_data['senkou_span_a'] and hour_data['close'] < hour_data['senkou_span_b']
        return 'bullish' if bullish else 'bearish' if bearish else None
    
    def check_5m_entry(self, row, trend):
        bullish_crossover = row['ema_5'] > row['ema_8'] > row['ema_13']
        bearish_crossover = row['ema_5'] < row['ema_8'] < row['ema_13']
        rsi_filter = 40 <= row['rsi'] <= 60
        if trend == 'bullish' and bullish_crossover and rsi_filter:
            return 'long'
        elif trend == 'bearish' and bearish_crossover and rsi_filter:
            return 'short'
        return None
    
    def run_backtest(self):
        for index, row in self.df_5m.iterrows():
            if index < 50:
                continue
            row['timestamp'] = row['timestamp'].tz_localize('UTC') if row['timestamp'].tzinfo is None else row['timestamp']
            current_time = row['timestamp'].time()
            if not (13 <= current_time.hour < 21):
                continue
            
            # Manage existing trade
            if self.current_trade:
                # Check and update trailing stop for long positions
                if self.current_trade['type'] == 'long':
                    trailing_stop = row['close'] - row['atr']
                    if trailing_stop > self.current_trade['stop_loss']:
                        self.current_trade['stop_loss'] = trailing_stop
                        self.current_trade['trailing_active'] = True
                    
                    # Check if stop loss was hit (using low price for more realistic simulation)
                    if row['low'] <= self.current_trade['stop_loss']:
                        self.close_trade(row, 'trailing_stop')
                
                # Check and update trailing stop for short positions
                else:
                    trailing_stop = row['close'] + row['atr']
                    if trailing_stop < self.current_trade['stop_loss']:
                        self.current_trade['stop_loss'] = trailing_stop
                        self.current_trade['trailing_active'] = True
                    
                    # Check if stop loss was hit (using high price for more realistic simulation)
                    if row['high'] >= self.current_trade['stop_loss']:
                        self.close_trade(row, 'trailing_stop')
            
            # Check for new entry signals
            else:
                trend = self.check_1h_trend(row['timestamp'])
                entry_signal = self.check_5m_entry(row, trend)
                
                if entry_signal:
                    entry_price = row['close']
                    
                    # Calculate stop loss level based on ATR
                    stop_loss = entry_price - (2 * row['atr']) if entry_signal == 'long' else entry_price + (2 * row['atr'])
                    stop_loss_distance = abs(entry_price - stop_loss)
                    
                    # Calculate risk amount (fixed at 1% of current balance)
                    risk_amount = self.current_balance * self.risk_percentage
                    
                    # Calculate position size based on risk amount and stop loss distance
                    # This ensures that if stopped out, the loss will be exactly the risk amount
                    position_size = (risk_amount / stop_loss_distance) * self.leverage
                    
                    self.current_trade = {
                        'type': entry_signal,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'position_size': position_size,
                        'risk_amount': risk_amount,
                        'entry_time': row['timestamp'],
                        'initial_stop_distance': stop_loss_distance,
                        'trailing_active': False
                    }
    
    def close_trade(self, row, exit_type):
        # Determine the correct exit price based on exit type
        if exit_type == 'trailing_stop':
            # Use the stop loss price for accurate PnL calculation
            exit_price = self.current_trade['stop_loss']
        else:
            exit_price = row['close']
        
        # Calculate price difference based on trade direction
        price_diff = (exit_price - self.current_trade['entry_price']) if self.current_trade['type'] == 'long' else (self.current_trade['entry_price'] - exit_price)
        
        # Calculate actual PnL accounting for position size and leverage
        # Dividing by leverage here ensures proper calculation of actual P&L impact on account
        actual_pnl = price_diff * self.current_trade['position_size'] / self.leverage
        
        # Update account balance
        self.current_balance += actual_pnl

        # Record trade details
        trade_record = {
            'type': self.current_trade['type'],
            'entry_price': self.current_trade['entry_price'],
            'exit_price': exit_price,
            'exit_type': exit_type,
            'pnl': actual_pnl,
            'entry_time': self.current_trade['entry_time'],
            'exit_time': row['timestamp'],
            'stop_loss': self.current_trade['stop_loss'],
            'initial_stop_distance': self.current_trade.get('initial_stop_distance', 0),
            'position_size': self.current_trade['position_size'],
            'exit_balance': self.current_balance,
            'risk_amount': self.current_trade['risk_amount'],
            'trailing_stop_used': self.current_trade.get('trailing_active', False),
        }
        
        self.trades.append(trade_record)
        self.current_trade = None

    def generate_report(self):
        if not self.trades:
            print("No trades executed during backtest period")
            return
            
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate max drawdown
        balance_curve = [self.initial_balance]
        for trade in self.trades:
            balance_curve.append(trade['exit_balance'])
        
        peak = self.initial_balance
        drawdowns = []
        for balance in balance_curve:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns)
        
        # Calculate average risk per trade
        avg_risk_pct = sum([t['risk_amount'] / (t['exit_balance'] - t['pnl']) * 100 for t in self.trades]) / total_trades
        
        # Calculate largest loss as percentage of account
        if self.trades:
            losses = [t['pnl'] for t in self.trades if t['pnl'] < 0]
            largest_loss_pct = min(losses) / self.initial_balance * 100 if losses else 0
        else:
            largest_loss_pct = 0
        
        print("\n=== BACKTESTING PERFORMANCE REPORT ===")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.current_balance:,.2f}")
        print(f"Absolute Return: ${self.current_balance - self.initial_balance:,.2f}")
        print(f"Percent Return: {((self.current_balance/self.initial_balance - 1) * 100):.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"\nTotal Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {total_trades - winning_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"\nAverage Risk per Trade: {avg_risk_pct:.2f}%")
        print(f"Largest Single Loss: {abs(largest_loss_pct):.2f}% of initial account")
        print(f"Target Risk per Trade: {self.risk_percentage*100:.2f}%")
        
        # Check if risk management is working as expected
        if abs(largest_loss_pct) > (self.risk_percentage * 100 * 1.1):  # Allow 10% margin of error
            print("\n⚠️ WARNING: Largest loss exceeds your risk tolerance!")
            print(f"Target max loss: {self.risk_percentage*100:.2f}%, Actual max loss: {abs(largest_loss_pct):.2f}%")
        else:
            print("\n✅ Risk management is working correctly!")
            print(f"All losses are within the {self.risk_percentage*100:.2f}% risk tolerance")

    def export_to_excel(self, filename=f'{asset}_backtest_results.xlsx'):
        """Enhanced Excel export with trailing stop analysis"""
        if not self.trades:
            print("No trades to export")
            return
        
        trades_data = []
        for trade in self.trades:
            # Calculate percentage gain/loss relative to account at time of trade
            pnl_percentage = (trade['pnl'] / (trade['exit_balance'] - trade['pnl'])) * 100
            
            trade_data = {
                'Position': trade['type'].upper(),
                'Entry Time': trade['entry_time'].tz_localize(None),
                'Exit Time': trade['exit_time'].tz_localize(None),
                'Entry Price': trade['entry_price'],
                'Initial Stop Loss': trade['entry_price'] + (trade['initial_stop_distance'] if trade['type'] == 'short' else -trade['initial_stop_distance']),
                'Final Stop Loss': trade['stop_loss'],
                'Exit Price': trade['exit_price'],
                'Position Size': trade['position_size'],
                'Outcome': trade['exit_type'].upper(),
                'Trailing Stop Used': 'Yes' if trade.get('trailing_stop_used', False) else 'No',
                'PnL ($)': trade['pnl'],
                'PnL (%)': pnl_percentage,
                'Balance After Trade': trade['exit_balance'],
                'Risk Amount ($)': trade['risk_amount'],
                'Actual % Risked': (trade['pnl'] / (trade['exit_balance'] - trade['pnl'])) * 100 if trade['pnl'] < 0 else trade['risk_amount'] / (trade['exit_balance'] - trade['pnl']) * 100,
                'Risk Management': f"Trailing Stop with {self.risk_percentage*100:.1f}% Account Risk"
            }
            trades_data.append(trade_data)
        
        df_trades = pd.DataFrame(trades_data)
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            df_trades.to_excel(writer, sheet_name='Trade History', index=True)
            
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
            worksheet.set_column('A:A', 18, cell_format)  # Index column
            worksheet.set_column('B:B', 15, cell_format)  # Position column
            worksheet.set_column('C:D', 18, cell_format)  # Time columns
            worksheet.set_column('E:H', 15, money_format)  # Price columns
            worksheet.set_column('I:J', 15, cell_format)  # Position Outcome
            worksheet.set_column('K:K', 15, cell_format)  # Trailing Stop Used
            worksheet.set_column('L:L', 15, money_format)  # PnL column
            worksheet.set_column('M:M', 15, percent_format)  # PnL % column
            worksheet.set_column('N:N', 15, money_format)  # Balance
            worksheet.set_column('O:O', 15, money_format)  # Risk Amount
            worksheet.set_column('P:P', 15, percent_format)  # Actual % Risked
            worksheet.set_column('Q:Q', 30, cell_format)  # Risk Management
            
            # Add summary statistics
            summary_start_row = len(df_trades) + 4
            worksheet.write(summary_start_row, 0, 'Summary Statistics', header_format)
            worksheet.write(summary_start_row + 1, 0, 'Initial Balance:', cell_format)
            worksheet.write(summary_start_row + 1, 1, self.initial_balance, money_format)
            worksheet.write(summary_start_row + 2, 0, 'Final Balance:', cell_format)
            worksheet.write(summary_start_row + 2, 1, self.current_balance, money_format)
            worksheet.write(summary_start_row + 3, 0, 'Total Return:', cell_format)
            worksheet.write(summary_start_row + 3, 1, 
                          (self.current_balance/self.initial_balance - 1), percent_format)
            worksheet.write(summary_start_row + 4, 0, 'Total Trades:', cell_format)
            worksheet.write(summary_start_row + 4, 1, len(self.trades))
            worksheet.write(summary_start_row + 5, 0, 'Win Rate:', cell_format)
            win_rate = len([t for t in self.trades if t['pnl'] > 0]) / len(self.trades)
            worksheet.write(summary_start_row + 5, 1, win_rate, percent_format)
            
            # Add risk management statistics
            worksheet.write(summary_start_row + 7, 0, 'Risk Management Statistics', header_format)
            losses = [t['pnl'] for t in self.trades if t['pnl'] < 0]
            if losses:
                largest_loss = min(losses)
                largest_loss_pct = largest_loss / self.initial_balance
                worksheet.write(summary_start_row + 8, 0, 'Largest Loss ($):', cell_format)
                worksheet.write(summary_start_row + 8, 1, largest_loss, money_format)
                worksheet.write(summary_start_row + 9, 0, 'Largest Loss (%):', cell_format)
                worksheet.write(summary_start_row + 9, 1, largest_loss_pct, percent_format)
                worksheet.write(summary_start_row + 10, 0, 'Target Max Loss (%):', cell_format)
                worksheet.write(summary_start_row + 10, 1, self.risk_percentage, percent_format)
                risk_exceeded = "YES" if abs(largest_loss_pct) > (self.risk_percentage * 1.1) else "NO"
                worksheet.write(summary_start_row + 11, 0, 'Risk Tolerance Exceeded:', cell_format)
                worksheet.write(summary_start_row + 11, 1, risk_exceeded)
            
            # Add trailing stop specific statistics
            worksheet.write(summary_start_row + 13, 0, 'Trailing Stop Statistics', header_format)
            trailing_trades = len([t for t in self.trades if t.get('trailing_stop_used', False)])
            worksheet.write(summary_start_row + 14, 0, 'Trades Using Trailing Stop:', cell_format)
            worksheet.write(summary_start_row + 14, 1, trailing_trades)
            worksheet.write(summary_start_row + 15, 0, 'Trailing Stop %:', cell_format)
            worksheet.write(summary_start_row + 15, 1, trailing_trades/len(self.trades), percent_format)
            
        print(f"\nResults exported to {filename}")

# Run commands
if __name__ == "__main__":
    backtest = Backtester(initial_balance=20000.0, risk_percentage=0.01, leverage=10)
    backtest.load_data(f"{asset}USDT_1h.csv", f"{asset}USDT_5m.csv")
    backtest.run_backtest()
    backtest.generate_report()
    backtest.export_to_excel()  # This will create the backtest_results.xlsx
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
    def __init__(self, initial_balance=10000.0, risk_percentage=0.01, leverage=20):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_percentage = risk_percentage
        self.leverage = leverage
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
        price_distance = abs(entry_price - stop_loss)
        position_size = risk_amount / price_distance
        
        # Check leverage limits
        notional_value = position_size * entry_price
        required_margin = notional_value / self.leverage
        
        if required_margin > self.current_balance:
            max_notional = self.current_balance * self.leverage
            position_size = max_notional / entry_price
            actual_risk = position_size * price_distance
            print(f"Warning: Position size reduced. Actual risk: ${actual_risk:.2f}")
        
        return position_size
    
    def run_backtest(self):
        """Run the backtest with exact risk management"""
        for index, row in self.df_5m.iterrows():
            if index < 200:
                continue

            current_time = row['timestamp'].time()
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
        """
        Enhanced trade closing with:
        - Exact risk amount on stop losses
        - Full profit capture on winning trades
        """
        # Calculate base risk amount
        risk_amount = self.current_trade['entry_balance'] * self.risk_percentage
        
        if exit_type == 'stop_loss':
            # Force exact risk amount on stop loss
            pnl = -risk_amount
            # Use stop loss price for record keeping
            exit_price = self.current_trade['stop_loss']
        else:  # take_profit or other exit types
            # Capture full movement for winning trades
            if self.current_trade['type'] == 'long':
                exit_price = row['high']
                price_change = exit_price - self.current_trade['entry_price']
            else:  # short
                exit_price = row['low']
                price_change = self.current_trade['entry_price'] - exit_price
            
            # Calculate actual PnL for winning trades
            pnl = price_change * self.current_trade['position_size']
        
        # Calculate R-multiple achieved
        r_multiple = abs(pnl / risk_amount)
        
        # Calculate how far beyond 3R we went (if applicable)
        excess_r = max(0, r_multiple - 3) if exit_type == 'take_profit' else 0
        
        # Update balance
        self.current_balance += pnl
        
        # Enhanced trade record with detailed metrics
        trade_record = {
            **self.current_trade,
            'exit_price': exit_price,
            'exit_time': row['timestamp'],
            'exit_type': exit_type,
            'pnl': pnl,
            'exit_balance': self.current_balance,
            'risk_amount': risk_amount,
            'base_target': risk_amount * 3,  # Standard 3R target
            'r_multiple': r_multiple,
            'excess_r': excess_r,
            'price_movement': (price_change if exit_type != 'stop_loss' 
                             else exit_price - self.current_trade['entry_price']),
            'movement_percentage': (
                (price_change / self.current_trade['entry_price'] * 100) 
                if exit_type != 'stop_loss' 
                else -self.risk_percentage * 100
            )
        }
        
        self.trades.append(trade_record)
        self.current_trade = None

    def generate_report(self):
        """Enhanced performance report with detailed R-multiple analysis"""
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # R-multiple analysis
        r_multiples = [t['r_multiple'] for t in self.trades]
        excess_rs = [t['excess_r'] for t in self.trades]
        trades_beyond_3r = len([t for t in self.trades if t['excess_r'] > 0])
        
        print("\n=== BACKTEST PERFORMANCE REPORT ===")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.current_balance:,.2f}")
        print(f"Total Return: {((self.current_balance/self.initial_balance - 1) * 100):.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        
        print("\n=== R-MULTIPLE ANALYSIS ===")
        print(f"Average R-Multiple: {np.mean(r_multiples):.2f}")
        print(f"Best R-Multiple: {max(r_multiples):.2f}")
        print(f"Trades Beyond 3R: {trades_beyond_3r}")
        print(f"Average Excess R: {np.mean(excess_rs):.2f}")
        
        print("\n=== NOTABLE TRADES ===")
        best_trades = sorted(self.trades, key=lambda x: x['r_multiple'], reverse=True)[:5]
        for i, trade in enumerate(best_trades, 1):
            print(f"\nTop Trade #{i}")
            print(f"Entry Price: ${trade['entry_price']:,.2f}")
            print(f"Exit Price: ${trade['exit_price']:,.2f}")
            print(f"R-Multiple: {trade['r_multiple']:.2f}")
            print(f"PnL: ${trade['pnl']:,.2f}")
            print(f"Movement: {trade['movement_percentage']:.2f}%")

    def export_to_excel(self, filename='trading_results.xlsx'):
        """Enhanced Excel export with R-multiple analysis"""
        if not self.trades:
            print("No trades to export")
            return
        
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
                'Risk-Reward Ratio': '1:3'
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
            worksheet.set_column('K:K', 15, money_format)  # PnL column
            worksheet.set_column('L:N', 18, money_format)  # Balances
            
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
            
        print(f"\nResults exported to {filename}")

# Run commands
if __name__ == "__main__":
    backtest = BTCTradeBacktester(initial_balance=10000.0, risk_percentage=0.01)
    backtest.load_data("BTCUSDT_1h.csv", "BTCUSDT_5m.csv")
    backtest.run_backtest()
    backtest.generate_report()
    backtest.export_to_excel()  # This will create trading_results.xlsx
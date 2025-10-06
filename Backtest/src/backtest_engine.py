import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import os
import time

# Configuration Constants
CACHE_DURATION = 86400  # 1 day in seconds
CAPITAL_GAINS_TAX_RATE = 0.15  # 15% long-term capital gains tax
SHORT_TERM_CAPITAL_GAINS_TAX_RATE = 0.35  # 35% short-term capital gains tax
LONG_TERM_HOLDING_PERIOD_DAYS = 365  # 1 year for long-term vs short-term

def get_cached_data(symbol):
    cache_file = f"data/cache/{symbol}_cache.csv"
    if os.path.exists(cache_file):
        last_modified_time = os.path.getmtime(cache_file)
        current_time = time.time()
        if current_time - last_modified_time < CACHE_DURATION:
            return pd.read_csv(cache_file, parse_dates=['Date'], index_col='Date')
    data = download_data(symbol)
    data.to_csv(cache_file)
    return data

def download_data(symbol):
    """Improved data download with proper datetime handling"""
    stock = yf.Ticker(symbol)
    df = stock.history(start="2012-01-20", auto_adjust=False)
    
    # Keep dates as datetime64 type
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Remove timezone info
    
    # Select and order columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    
    # Set Date as index
    df = df.set_index('Date')
    return df

def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_data(symbol):
    return get_cached_data(symbol)

def determine_action(current_date, spy_data, bull_data, bear_data, holdings, bull_symbol, bear_symbol, algo_params=None):
    """
    Determine trading action based on current market conditions
    
    Args:
        algo_params: Dictionary with algorithm parameters:
            - spy_sma_window: SPY SMA window (default: 200)
            - etf_sma_window: ETF SMA window (default: 20)
            - etf_rsi_window: ETF RSI window (default: 10)
            - etf_rsi_threshold: ETF RSI threshold (default: 31)
    """
    # Use default parameters if none provided
    if algo_params is None:
        algo_params = {
            'spy_sma_window': 200,
            'etf_sma_window': 20,
            'etf_rsi_window': 10,
            'etf_rsi_threshold': 31
        }
    
    spy_price = spy_data.loc[current_date]['Close']
    spy_sma = spy_data.loc[current_date][f'{algo_params["spy_sma_window"]}_SMA']
    
    bull_price = bull_data.loc[current_date]['Close']
    bull_rsi = bull_data.loc[current_date][f'{algo_params["etf_rsi_window"]}_RSI']
    bull_sma = bull_data.loc[current_date][f'{algo_params["etf_sma_window"]}_SMA']
    
    bear_price = bear_data.loc[current_date]['Close']
    bear_rsi = bear_data.loc[current_date][f'{algo_params["etf_rsi_window"]}_RSI']
    bear_sma = bear_data.loc[current_date][f'{algo_params["etf_sma_window"]}_SMA']
    
    action = 'HOLD'
    reason = 'No trading conditions met'
    
    if spy_price > spy_sma:
        if holdings[bull_symbol] > 0:
            action = 'HOLD'
            reason = f'SPY above {algo_params["spy_sma_window"]} SMA ({spy_price:.2f} > {spy_sma:.2f}), already holding {bull_symbol}'
        else:
            action = f'BUY_{bull_symbol}_SELL_{bear_symbol}'
            reason = f'SPY above {algo_params["spy_sma_window"]} SMA ({spy_price:.2f} > {spy_sma:.2f}), switching to bull position'
    else:
        if bull_rsi < algo_params['etf_rsi_threshold']:
            if holdings[bull_symbol] > 0:
                action = 'HOLD'
                reason = f'SPY below {algo_params["spy_sma_window"]} SMA but {bull_symbol} oversold (RSI: {bull_rsi:.2f} < {algo_params["etf_rsi_threshold"]}), holding {bull_symbol}'
            else:
                action = f'BUY_{bull_symbol}_SELL_{bear_symbol}'
                reason = f'SPY below {algo_params["spy_sma_window"]} SMA but {bull_symbol} oversold (RSI: {bull_rsi:.2f} < {algo_params["etf_rsi_threshold"]}), buying {bull_symbol}'
        elif bull_price < bull_sma:
            if holdings[bull_symbol] > 0:
                action = f'SELL_{bull_symbol}_BUY_{bear_symbol}'
                reason = f'SPY below {algo_params["spy_sma_window"]} SMA & {bull_symbol} below {algo_params["etf_sma_window"]} SMA ({bull_price:.2f} < {bull_sma:.2f}), switching to bear'
            else:
                action = 'HOLD'
                reason = f'SPY below {algo_params["spy_sma_window"]} SMA & {bull_symbol} below {algo_params["etf_sma_window"]} SMA, already in defensive position'
        elif bear_price < bear_sma:
            if holdings[bear_symbol] > 0:
                action = f'SELL_{bear_symbol}'
                reason = f'{bear_symbol} below {algo_params["etf_sma_window"]} SMA ({bear_price:.2f} < {bear_sma:.2f}), selling bear position'
            else:
                action = 'HOLD'
                reason = f'{bear_symbol} below {algo_params["etf_sma_window"]} SMA but not holding it'
        else:
            action = 'HOLD'
            reason = f'SPY below {algo_params["spy_sma_window"]} SMA ({spy_price:.2f} < {spy_sma:.2f}), defensive mode but no specific signals'
    
    return action, reason

def sell_shares(symbol, holdings, price, lots, current_date):
    shares_to_sell = holdings[symbol]
    if shares_to_sell == 0:
        return holdings['Cash']

    proceeds = shares_to_sell * price
    
    holdings['Cash'] += proceeds
    holdings[symbol] = 0
    lots[symbol] = []
    
    return holdings['Cash']

def execute_action(action, current_date, prices, holdings, lots, bull_symbol, bear_symbol):
    
    if action == f'BUY_{bull_symbol}_SELL_{bear_symbol}':
        if holdings[bear_symbol] > 0:
            cash = sell_shares(bear_symbol, holdings, prices[bear_symbol], lots, current_date)
        if holdings['Cash'] > 0:
            shares_to_buy = int(holdings['Cash'] // prices[bull_symbol])
            if shares_to_buy > 0:
                cost = shares_to_buy * prices[bull_symbol]
                holdings[bull_symbol] += shares_to_buy
                holdings['Cash'] -= cost
                lots[bull_symbol].append({'shares': shares_to_buy, 'price': prices[bull_symbol], 'purchase_date': current_date})
    
    elif action == f'SELL_{bull_symbol}_BUY_{bear_symbol}':
        if holdings[bull_symbol] > 0:
            cash = sell_shares(bull_symbol, holdings, prices[bull_symbol], lots, current_date)
        if holdings['Cash'] > 0:
            shares_to_buy = int(holdings['Cash'] // prices[bear_symbol])
            if shares_to_buy > 0:
                cost = shares_to_buy * prices[bear_symbol]
                holdings[bear_symbol] += shares_to_buy
                holdings['Cash'] -= cost
                lots[bear_symbol].append({'shares': shares_to_buy, 'price': prices[bear_symbol], 'purchase_date': current_date})
    
    elif action == f'SELL_{bear_symbol}':
        if holdings[bear_symbol] > 0:
            cash = sell_shares(bear_symbol, holdings, prices[bear_symbol], lots, current_date)
    
    return holdings, lots

def run_backtest(start_date, end_date, initial_cash, recurring_investment, bull_symbol='TQQQ', bear_symbol='SQQQ', algo_params=None):
    """Run backtest with configurable algorithm parameters"""
    # Use default parameters if none provided
    if algo_params is None:
        algo_params = {
            'spy_sma_window': 200,
            'etf_sma_window': 20,
            'etf_rsi_window': 10,
            'etf_rsi_threshold': 31
        }
    
    # Convert datetime to pandas Timestamp for comparison
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    spy_data = get_data("SPY")
    bull_data = get_data(bull_symbol)
    bear_data = get_data(bear_symbol)
    
    # Calculate indicators with configurable parameters
    spy_data[f'{algo_params["spy_sma_window"]}_SMA'] = calculate_sma(spy_data, algo_params['spy_sma_window'])
    bull_data[f'{algo_params["etf_rsi_window"]}_RSI'] = calculate_rsi(bull_data, algo_params['etf_rsi_window'])
    bull_data[f'{algo_params["etf_sma_window"]}_SMA'] = calculate_sma(bull_data, algo_params['etf_sma_window'])
    bear_data[f'{algo_params["etf_rsi_window"]}_RSI'] = calculate_rsi(bear_data, algo_params['etf_rsi_window'])
    bear_data[f'{algo_params["etf_sma_window"]}_SMA'] = calculate_sma(bear_data, algo_params['etf_sma_window'])
    
    # Keep backward compatibility columns for display
    if algo_params['spy_sma_window'] == 200:
        spy_data['200_SMA'] = spy_data[f'{algo_params["spy_sma_window"]}_SMA']
    if algo_params['etf_rsi_window'] == 10:
        bull_data['10_RSI'] = bull_data[f'{algo_params["etf_rsi_window"]}_RSI']
        bear_data['10_RSI'] = bear_data[f'{algo_params["etf_rsi_window"]}_RSI']
    if algo_params['etf_sma_window'] == 20:
        bull_data['20_SMA'] = bull_data[f'{algo_params["etf_sma_window"]}_SMA']
        bear_data['20_SMA'] = bear_data[f'{algo_params["etf_sma_window"]}_SMA']
    
    # Drop NaN values
    spy_data.dropna(subset=[f'{algo_params["spy_sma_window"]}_SMA'], inplace=True)
    bull_data.dropna(subset=[f'{algo_params["etf_rsi_window"]}_RSI', f'{algo_params["etf_sma_window"]}_SMA'], inplace=True)
    bear_data.dropna(subset=[f'{algo_params["etf_rsi_window"]}_RSI', f'{algo_params["etf_sma_window"]}_SMA'], inplace=True)
    
    all_dates = spy_data.index.intersection(bull_data.index).intersection(bear_data.index)
    all_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
    
    holdings = {
        'Cash': initial_cash,
        bull_symbol: 0,
        bear_symbol: 0
    }
    lots = {
        bull_symbol: [],
        bear_symbol: []
    }
    portfolio_value = []

    # Track contributions (money added to the account) and cumulative invested
    cumulative_invested = 0.0
    first_date = all_dates[0] if len(all_dates) > 0 else None
    
    for current_date in all_dates:
        # Determine contribution for this date
        contribution = 0.0
        if first_date is not None and current_date == first_date:
            # Initial capital is considered a contribution on the first backtest date
            contribution += initial_cash
        if recurring_investment > 0 and current_date.day == 1:
            contribution += recurring_investment
            holdings['Cash'] += recurring_investment
        
        # Update cumulative invested after calculating today's contribution
        cumulative_invested += contribution
        
        prices = {
            'SPY': spy_data.loc[current_date]['Close'],
            bull_symbol: bull_data.loc[current_date]['Close'],
            bear_symbol: bear_data.loc[current_date]['Close']
        }
        indicators = {
            f'SPY_{algo_params["spy_sma_window"]}_SMA': spy_data.loc[current_date][f'{algo_params["spy_sma_window"]}_SMA'],
            f'{bull_symbol}_{algo_params["etf_rsi_window"]}_RSI': bull_data.loc[current_date][f'{algo_params["etf_rsi_window"]}_RSI'],
            f'{bull_symbol}_{algo_params["etf_sma_window"]}_SMA': bull_data.loc[current_date][f'{algo_params["etf_sma_window"]}_SMA'],
            f'{bear_symbol}_{algo_params["etf_rsi_window"]}_RSI': bear_data.loc[current_date][f'{algo_params["etf_rsi_window"]}_RSI'],
            f'{bear_symbol}_{algo_params["etf_sma_window"]}_SMA': bear_data.loc[current_date][f'{algo_params["etf_sma_window"]}_SMA']
        }
        # Add backward compatibility columns for display
        if algo_params['spy_sma_window'] == 200:
            indicators['SPY_200_SMA'] = indicators[f'SPY_{algo_params["spy_sma_window"]}_SMA']
        if algo_params['etf_rsi_window'] == 10:
            indicators[f'{bull_symbol}_10_RSI'] = indicators[f'{bull_symbol}_{algo_params["etf_rsi_window"]}_RSI']
            indicators[f'{bear_symbol}_10_RSI'] = indicators[f'{bear_symbol}_{algo_params["etf_rsi_window"]}_RSI']
        if algo_params['etf_sma_window'] == 20:
            indicators[f'{bull_symbol}_20_SMA'] = indicators[f'{bull_symbol}_{algo_params["etf_sma_window"]}_SMA']
            indicators[f'{bear_symbol}_20_SMA'] = indicators[f'{bear_symbol}_{algo_params["etf_sma_window"]}_SMA']
        
        action, reason = determine_action(current_date, spy_data, bull_data, bear_data, holdings, bull_symbol, bear_symbol, algo_params)
        
        holdings, lots = execute_action(action, current_date, prices, holdings, lots, bull_symbol, bear_symbol)
        
        portfolio_total = holdings['Cash'] + (holdings[bull_symbol] * prices[bull_symbol]) + (holdings[bear_symbol] * prices[bear_symbol])
        
        # Create portfolio record with dynamic column names
        portfolio_record = {
            'Date': current_date,
            'Action': action,
            'Reason': reason,
            'SPY Price': prices['SPY'],
            f'SPY {algo_params["spy_sma_window"]} SMA': indicators[f'SPY_{algo_params["spy_sma_window"]}_SMA'],
            f'{bull_symbol} Price': prices[bull_symbol],
            f'{bull_symbol} {algo_params["etf_rsi_window"]} RSI': indicators[f'{bull_symbol}_{algo_params["etf_rsi_window"]}_RSI'],
            f'{bull_symbol} {algo_params["etf_sma_window"]} SMA': indicators[f'{bull_symbol}_{algo_params["etf_sma_window"]}_SMA'],
            f'{bear_symbol} Price': prices[bear_symbol],
            f'{bear_symbol} {algo_params["etf_rsi_window"]} RSI': indicators[f'{bear_symbol}_{algo_params["etf_rsi_window"]}_RSI'],
            f'{bear_symbol} {algo_params["etf_sma_window"]} SMA': indicators[f'{bear_symbol}_{algo_params["etf_sma_window"]}_SMA'],
            f'{bull_symbol} Shares': holdings[bull_symbol],
            f'{bear_symbol} Shares': holdings[bear_symbol],
            'Cash': holdings['Cash'],
            'Portfolio Value': portfolio_total,
            'Contribution': contribution,
            'Total Invested': cumulative_invested
        }
        
        # Add backward compatibility columns for display if using default parameters
        if algo_params['spy_sma_window'] == 200:
            portfolio_record['SPY 200 SMA'] = portfolio_record[f'SPY {algo_params["spy_sma_window"]} SMA']
        if algo_params['etf_rsi_window'] == 10:
            portfolio_record[f'{bull_symbol} 10 RSI'] = portfolio_record[f'{bull_symbol} {algo_params["etf_rsi_window"]} RSI']
            portfolio_record[f'{bear_symbol} 10 RSI'] = portfolio_record[f'{bear_symbol} {algo_params["etf_rsi_window"]} RSI']
        if algo_params['etf_sma_window'] == 20:
            portfolio_record[f'{bull_symbol} 20 SMA'] = portfolio_record[f'{bull_symbol} {algo_params["etf_sma_window"]} SMA']
            portfolio_record[f'{bear_symbol} 20 SMA'] = portfolio_record[f'{bear_symbol} {algo_params["etf_sma_window"]} SMA']
        
        portfolio_value.append(portfolio_record)
    
    return pd.DataFrame(portfolio_value)

def run_buy_and_hold(symbol, start_date, end_date, initial_cash):
    """Run buy & hold strategy for comparison"""
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    data = get_data(symbol)
    
    # Filter data for the backtest period
    filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    if filtered_data.empty:
        return pd.DataFrame()
    
    # Get the first available price in the period
    first_price = filtered_data['Close'].iloc[0]
    shares = initial_cash / first_price
    
    portfolio_value = []
    for current_date in filtered_data.index:
        current_price = filtered_data.loc[current_date]['Close']
        portfolio_total = shares * current_price
        
        portfolio_value.append({
            'Date': current_date,
            'Portfolio Value': portfolio_total,
        })
    
    return pd.DataFrame(portfolio_value)

def format_results_for_display(results_df):
    """Format the results dataframe for display (same formatting as original algo.py)"""
    display_df = results_df.copy()
    
    # Extract symbols from column names
    bull_symbol = None
    bear_symbol = None
    for col in display_df.columns:
        if ' Price' in col and col != 'SPY Price':
            symbol = col.replace(' Price', '')
            if 'TQQQ' in symbol or 'QQQ' in symbol or 'TEC' in symbol:
                bull_symbol = symbol
            elif 'SQQQ' in symbol or 'PSQ' in symbol:
                bear_symbol = symbol
    
    if not bull_symbol:
        bull_symbol = 'BULL'
    if not bear_symbol:
        bear_symbol = 'BEAR'
    
    price_cols = ['SPY Price', 'SPY 200 SMA', f'{bull_symbol} Price', f'{bull_symbol} 20 SMA', f'{bear_symbol} Price', f'{bear_symbol} 20 SMA']
    rsi_cols = [f'{bull_symbol} 10 RSI', f'{bear_symbol} 10 RSI']
    
    for col in price_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    for col in rsi_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    display_df['Cash'] = display_df['Cash'].apply(lambda x: f"${x:,.2f}")
    display_df['Portfolio Value'] = display_df['Portfolio Value'].apply(lambda x: f"${x:,.2f}")
    if 'Contribution' in display_df.columns:
        display_df['Contribution'] = display_df['Contribution'].apply(lambda x: f"${x:,.2f}")
    if 'Total Invested' in display_df.columns:
        display_df['Total Invested'] = display_df['Total Invested'].apply(lambda x: f"${x:,.2f}")
    
    return display_df

def calculate_summary_stats(results_df, initial_cash, recurring_investment, bull_symbol, bear_symbol, show_spy=False, show_buy_hold=False):
    """Calculate summary statistics for the strategy"""
    # Get final values
    final_value = results_df['Portfolio Value'].iloc[-1]
    total_return = ((final_value / initial_cash) - 1) * 100

    # Compute total invested from results (initial + monthly contributions)
    if 'Total Invested' in results_df.columns:
        total_invested = results_df['Total Invested'].iloc[-1]
    else:
        total_invested = initial_cash

    net_profit = final_value - total_invested
    roi_on_invested = (final_value / total_invested - 1) * 100 if total_invested > 0 else 0.0
    
    summary = {
        'initial_investment': initial_cash,
        'total_invested': total_invested,
        'final_portfolio_value': final_value,
        'net_profit': net_profit,
        'roi_on_invested': roi_on_invested,
        'total_return': total_return,
        'monthly_investment': recurring_investment
    }
    
    # Add buy & hold comparisons if requested
    if show_spy or show_buy_hold:
        start_date = results_df['Date'].iloc[0]
        end_date = results_df['Date'].iloc[-1]
        
        summary['comparisons'] = {}
        
        if show_spy:
            spy_results = run_buy_and_hold('SPY', start_date, end_date, initial_cash)
            if not spy_results.empty:
                spy_final = spy_results['Portfolio Value'].iloc[-1]
                spy_return = ((spy_final / initial_cash) - 1) * 100
                vs_spy = ((final_value / spy_final) - 1) * 100
                summary['comparisons']['spy'] = {
                    'final_value': spy_final,
                    'return': spy_return,
                    'vs_strategy': vs_spy
                }
        
        if show_buy_hold:
            bull_results = run_buy_and_hold(bull_symbol, start_date, end_date, initial_cash)
            if not bull_results.empty:
                bull_final = bull_results['Portfolio Value'].iloc[-1]
                bull_return = ((bull_final / initial_cash) - 1) * 100
                vs_bull = ((final_value / bull_final) - 1) * 100
                summary['comparisons'][bull_symbol] = {
                    'final_value': bull_final,
                    'return': bull_return,
                    'vs_strategy': vs_bull
                }
    
    return summary
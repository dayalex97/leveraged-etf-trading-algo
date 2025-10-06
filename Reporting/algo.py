import pandas as pd
import yfinance as yf
import os
import time
import argparse

# Configuration Constants
CACHE_DURATION = 86400  # 1 day in seconds

def get_cached_data(symbol):
    """Get cached data or download fresh data if cache is expired"""
    cache_file = f"cache/{symbol}_cache.csv"
    if os.path.exists(cache_file):
        last_modified_time = os.path.getmtime(cache_file)
        current_time = time.time()
        if current_time - last_modified_time < CACHE_DURATION:
            return pd.read_csv(cache_file, parse_dates=['Date'], index_col='Date')
    
    # Download fresh data
    data = download_data(symbol)
    data.to_csv(cache_file)
    return data

def download_data(symbol):
    """Download historical data from yfinance"""
    stock = yf.Ticker(symbol)
    df = stock.history(start="2012-01-20", auto_adjust=False)
    
    # Clean up the data
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    df = df.set_index('Date')
    return df

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_current_price_live(symbol):
    """Fetch the most recent price using yfinance (live/current data)"""
    try:
        stock = yf.Ticker(symbol)
        # Get the most recent data - this should include today's data if markets are open
        hist = stock.history(period="2d", interval="1m")
        
        if hist.empty:
            return None
        
        # Get the most recent price and timestamp
        most_recent = hist.iloc[-1]
        current_price = float(most_recent['Close'])
        current_date = hist.index[-1]
        
        return {
            'price': current_price,
            'date': current_date,
            'source': 'live_yfinance'
        }
    except Exception as e:
        print(f"Warning: Error fetching live price for {symbol}: {str(e)}")
        return None

def save_current_price_to_csv(symbol, price_data, current_dir="current_prices"):
    """Save current price data to CSV file"""
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    
    csv_file = f"{current_dir}/{symbol}.csv"
    
    # Create a simple CSV with Date and Close columns
    timestamp = price_data['date'].strftime('%Y-%m-%d %H:%M:%S') if price_data['date'] else pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(csv_file, 'w') as f:
        f.write("Date,Close\n")
        f.write(f"{timestamp},{price_data['price']:.2f}\n")
    
    print(f"Saved current price for {symbol}: ${price_data['price']:.2f} -> {csv_file}")

def fetch_and_save_current_prices(symbols, current_dir="current_prices"):
    """Fetch current prices for multiple symbols and save to CSV files"""
    print(f"Fetching current prices for: {', '.join(symbols)}")
    
    for symbol in symbols:
        price_data = fetch_current_price_live(symbol)
        if price_data:
            save_current_price_to_csv(symbol, price_data, current_dir)
        else:
            print(f"Failed to fetch current price for {symbol}")
    
    print("Current price fetch complete!\n")

def get_current_price_from_csv(symbol, current_dir="current_prices"):
    """Read current price from CSV file (either manually downloaded or auto-fetched)"""
    csv_file = f"{current_dir}/{symbol}.csv"
    if not os.path.exists(csv_file):
        return None
    
    try:
        # Read the CSV file - assume it has at least Date and Close columns
        df = pd.read_csv(csv_file)
        
        # Try different possible column names for the close price
        close_col = None
        for col in ['Close', 'close', 'Close*', 'Adj Close', 'Price', 'Last']:
            if col in df.columns:
                close_col = col
                break
        
        if close_col is None:
            print(f"Warning: No close price column found in {csv_file}")
            return None
        
        # Get the most recent price (last row)
        current_price = float(df[close_col].iloc[-1])
        
        # Try to get the date if available
        current_date = None
        for date_col in ['Date', 'date', 'Datetime', 'datetime', 'Time', 'time']:
            if date_col in df.columns:
                try:
                    current_date = pd.to_datetime(df[date_col].iloc[-1])
                    break
                except:
                    continue
        
        return {
            'price': current_price,
            'date': current_date,
            'source': 'current_csv'
        }
    
    except Exception as e:
        print(f"Warning: Error reading {csv_file}: {str(e)}")
        return None

def determine_action_and_reason(spy_data, bull_data, bear_data, current_date, bull_symbol, bear_symbol):
    """Determine what action the algorithm would take and why"""
    spy_price = spy_data.loc[current_date]['Close']
    spy_200_sma = spy_data.loc[current_date]['200_SMA']
    
    bull_price = bull_data.loc[current_date]['Close']
    bull_10_rsi = bull_data.loc[current_date]['10_RSI']
    bull_20_sma = bull_data.loc[current_date]['20_SMA']
    
    bear_price = bear_data.loc[current_date]['Close']
    bear_20_sma = bear_data.loc[current_date]['20_SMA']
    
    # Apply the algorithm logic
    if spy_price > spy_200_sma:
        action = f'HOLD {bull_symbol}'
        reason = f'SPY above 200 SMA ({spy_price:.2f} > {spy_200_sma:.2f}), bull market - hold {bull_symbol}'
    else:
        if bull_10_rsi < 31:
            action = f'HOLD {bull_symbol}'
            reason = f'SPY below 200 SMA but {bull_symbol} oversold (RSI: {bull_10_rsi:.2f} < 31), hold {bull_symbol}'
        elif bull_price < bull_20_sma:
            action = f'HOLD {bear_symbol}'
            reason = f'SPY below 200 SMA & {bull_symbol} below 20 SMA ({bull_price:.2f} < {bull_20_sma:.2f}), hold {bear_symbol}'
        elif bear_price < bear_20_sma:
            action = 'HOLD CASH'
            reason = f'{bear_symbol} below 20 SMA ({bear_price:.2f} < {bear_20_sma:.2f}), hold cash'
        else:
            action = f'HOLD {bear_symbol}'
            reason = f'SPY below 200 SMA ({spy_price:.2f} < {spy_200_sma:.2f}), defensive mode - hold {bear_symbol}'
    
    return action, reason

def get_current_market_status(bull_symbol='TQQQ', bear_symbol='SQQQ', use_current_prices=False, current_dir="current_prices"):
    """Get current market status with latest prices and indicators"""
    # Get historical data for all symbols to calculate indicators
    spy_data = get_cached_data("SPY")
    bull_data = get_cached_data(bull_symbol)
    bear_data = get_cached_data(bear_symbol)
    
    # Calculate indicators based on historical data
    spy_data['200_SMA'] = calculate_sma(spy_data, 200)
    bull_data['10_RSI'] = calculate_rsi(bull_data, 10)
    bull_data['20_SMA'] = calculate_sma(bull_data, 20)
    bear_data['10_RSI'] = calculate_rsi(bear_data, 10)
    bear_data['20_SMA'] = calculate_sma(bear_data, 20)
    
    # Remove rows with missing data
    spy_data.dropna(subset=['200_SMA'], inplace=True)
    bull_data.dropna(subset=['10_RSI', '20_SMA'], inplace=True)
    bear_data.dropna(subset=['10_RSI', '20_SMA'], inplace=True)
    
    # Get the most recent date with data for all symbols
    all_dates = spy_data.index.intersection(bull_data.index).intersection(bear_data.index)
    historical_date = all_dates[-1]
    
    # Start with historical data
    current_data = {
        'date': historical_date,
        'spy_price': spy_data.loc[historical_date]['Close'],
        'spy_200_sma': spy_data.loc[historical_date]['200_SMA'],
        'bull_price': bull_data.loc[historical_date]['Close'],
        'bull_10_rsi': bull_data.loc[historical_date]['10_RSI'],
        'bull_20_sma': bull_data.loc[historical_date]['20_SMA'],
        'bear_price': bear_data.loc[historical_date]['Close'],
        'bear_10_rsi': bear_data.loc[historical_date]['10_RSI'],
        'bear_20_sma': bear_data.loc[historical_date]['20_SMA'],
        'data_sources': {
            'spy': 'historical',
            'bull': 'historical', 
            'bear': 'historical'
        }
    }
    
    # Override with current prices if available and requested
    if use_current_prices:
        # Try to get current SPY price
        spy_current = get_current_price_from_csv("SPY", current_dir)
        if spy_current:
            current_data['spy_price'] = spy_current['price']
            current_data['data_sources']['spy'] = 'current'
            if spy_current['date']:
                current_data['date'] = spy_current['date']
        
        # Try to get current bull ETF price
        bull_current = get_current_price_from_csv(bull_symbol, current_dir)
        if bull_current:
            current_data['bull_price'] = bull_current['price']
            current_data['data_sources']['bull'] = 'current'
            if bull_current['date'] and not spy_current:
                current_data['date'] = bull_current['date']
        
        # Try to get current bear ETF price
        bear_current = get_current_price_from_csv(bear_symbol, current_dir)
        if bear_current:
            current_data['bear_price'] = bear_current['price']
            current_data['data_sources']['bear'] = 'current'
            if bear_current['date'] and not spy_current and not bull_current:
                current_data['date'] = bear_current['date']
    
    # Determine current action based on the (possibly updated) prices
    # Create temporary data structures for the action determination
    temp_spy_data = spy_data.copy()
    temp_bull_data = bull_data.copy()
    temp_bear_data = bear_data.copy()
    
    # Update the most recent price in temp data if we have current prices
    if use_current_prices:
        if current_data['data_sources']['spy'] == 'current':
            temp_spy_data.loc[historical_date, 'Close'] = current_data['spy_price']
        if current_data['data_sources']['bull'] == 'current':
            temp_bull_data.loc[historical_date, 'Close'] = current_data['bull_price']
        if current_data['data_sources']['bear'] == 'current':
            temp_bear_data.loc[historical_date, 'Close'] = current_data['bear_price']
    
    action, reason = determine_action_and_reason(temp_spy_data, temp_bull_data, temp_bear_data, historical_date, bull_symbol, bear_symbol)
    
    return current_data, action, reason

def check_crossover_signals(bull_symbol='TQQQ', bear_symbol='SQQQ'):
    """Check for potential crossover signals that might trigger new actions"""
    spy_data = get_cached_data("SPY")
    bull_data = get_cached_data(bull_symbol)
    bear_data = get_cached_data(bear_symbol)
    
    # Calculate indicators
    spy_data['200_SMA'] = calculate_sma(spy_data, 200)
    bull_data['10_RSI'] = calculate_rsi(bull_data, 10)
    bull_data['20_SMA'] = calculate_sma(bull_data, 20)
    bear_data['20_SMA'] = calculate_sma(bear_data, 20)
    
    # Remove rows with missing data
    spy_data.dropna(subset=['200_SMA'], inplace=True)
    bull_data.dropna(subset=['10_RSI', '20_SMA'], inplace=True)
    bear_data.dropna(subset=['20_SMA'], inplace=True)
    
    # Get most recent date
    all_dates = spy_data.index.intersection(bull_data.index).intersection(bear_data.index)
    current_date = all_dates[-1]
    
    crossover_alerts = []
    
    # Check SPY vs 200 SMA proximity
    spy_price = spy_data.loc[current_date]['Close']
    spy_200_sma = spy_data.loc[current_date]['200_SMA']
    spy_distance_pct = ((spy_price - spy_200_sma) / spy_200_sma) * 100
    
    if abs(spy_distance_pct) < 1.0:  # Within 1% of crossover
        direction = "above" if spy_price > spy_200_sma else "below"
        crossover_alerts.append(f"SPY CROSSOVER ALERT: SPY is {abs(spy_distance_pct):.2f}% {direction} its 200-day SMA ({spy_price:.2f} vs {spy_200_sma:.2f})")
    
    # Check Bull RSI proximity to oversold (31)
    bull_rsi = bull_data.loc[current_date]['10_RSI']
    if 28 <= bull_rsi <= 34:  # Near oversold threshold
        crossover_alerts.append(f"{bull_symbol} RSI ALERT: RSI at {bull_rsi:.2f}, approaching oversold threshold (31)")
    
    # Check Bull vs 20 SMA proximity
    bull_price = bull_data.loc[current_date]['Close']
    bull_20_sma = bull_data.loc[current_date]['20_SMA']
    bull_distance_pct = ((bull_price - bull_20_sma) / bull_20_sma) * 100
    
    if abs(bull_distance_pct) < 1.0:  # Within 1% of crossover
        direction = "above" if bull_price > bull_20_sma else "below"
        crossover_alerts.append(f"{bull_symbol} SMA CROSSOVER ALERT: {bull_symbol} is {abs(bull_distance_pct):.2f}% {direction} its 20-day SMA ({bull_price:.2f} vs {bull_20_sma:.2f})")
    
    # Check Bear vs 20 SMA proximity
    bear_price = bear_data.loc[current_date]['Close']
    bear_20_sma = bear_data.loc[current_date]['20_SMA']
    bear_distance_pct = ((bear_price - bear_20_sma) / bear_20_sma) * 100
    
    if abs(bear_distance_pct) < 1.0:  # Within 1% of crossover
        direction = "above" if bear_price > bear_20_sma else "below"
        crossover_alerts.append(f"{bear_symbol} SMA CROSSOVER ALERT: {bear_symbol} is {abs(bear_distance_pct):.2f}% {direction} its 20-day SMA ({bear_price:.2f} vs {bear_20_sma:.2f})")
    
    return crossover_alerts

def display_market_report(bull_symbol='TQQQ', bear_symbol='SQQQ', use_current_prices=False, current_dir="current_prices"):
    """Display current market status report"""
    print(f"{bull_symbol}/{bear_symbol} Strategy - Current Market Status")
    print("=" * 80)
    
    # Get current market data
    current_data, action, reason = get_current_market_status(bull_symbol, bear_symbol, use_current_prices, current_dir)
    
    # Display date and most recent action
    print(f"Market Date: {current_data['date'].strftime('%Y-%m-%d (%A)')}")
    print(f"Most Recent Action: {action}")
    print(f"Reasoning: {reason}")
    print()
    
    # Display current prices and indicators
    print("CURRENT MARKET DATA:")
    print("-" * 40)
    
    # Show data source info if using current prices
    if use_current_prices:
        spy_source = "LIVE" if current_data['data_sources']['spy'] == 'current' else "CLOSE"
        print(f"SPY Price: ${current_data['spy_price']:.2f} ({spy_source})")
    else:
        print(f"SPY Price: ${current_data['spy_price']:.2f} (CLOSE)")
    
    print(f"SPY 200-day SMA: ${current_data['spy_200_sma']:.2f}")
    spy_vs_sma = "ABOVE" if current_data['spy_price'] > current_data['spy_200_sma'] else "BELOW"
    spy_distance = abs(((current_data['spy_price'] - current_data['spy_200_sma']) / current_data['spy_200_sma']) * 100)
    print(f"SPY Status: {spy_vs_sma} 200-day SMA by {spy_distance:.2f}%")
    print()
    
    print(f"{bull_symbol} (Bull ETF):")
    if use_current_prices:
        bull_source = "LIVE" if current_data['data_sources']['bull'] == 'current' else "CLOSE"
        print(f"  Price: ${current_data['bull_price']:.2f} ({bull_source})")
    else:
        print(f"  Price: ${current_data['bull_price']:.2f} (CLOSE)")
    print(f"  10-day RSI: {current_data['bull_10_rsi']:.2f}")
    print(f"  20-day SMA: ${current_data['bull_20_sma']:.2f}")
    bull_vs_sma = "ABOVE" if current_data['bull_price'] > current_data['bull_20_sma'] else "BELOW"
    bull_distance = abs(((current_data['bull_price'] - current_data['bull_20_sma']) / current_data['bull_20_sma']) * 100)
    print(f"  Status: {bull_vs_sma} 20-day SMA by {bull_distance:.2f}%")
    
    # RSI status
    if current_data['bull_10_rsi'] < 31:
        rsi_status = "OVERSOLD (<31)"
    elif current_data['bull_10_rsi'] > 69:
        rsi_status = "OVERBOUGHT (>69)"
    else:
        rsi_status = "NEUTRAL"
    print(f"  RSI Status: {rsi_status}")
    print()
    
    print(f"{bear_symbol} (Bear ETF):")
    if use_current_prices:
        bear_source = "LIVE" if current_data['data_sources']['bear'] == 'current' else "CLOSE"
        print(f"  Price: ${current_data['bear_price']:.2f} ({bear_source})")
    else:
        print(f"  Price: ${current_data['bear_price']:.2f} (CLOSE)")
    print(f"  10-day RSI: {current_data['bear_10_rsi']:.2f}")
    print(f"  20-day SMA: ${current_data['bear_20_sma']:.2f}")
    bear_vs_sma = "ABOVE" if current_data['bear_price'] > current_data['bear_20_sma'] else "BELOW"
    bear_distance = abs(((current_data['bear_price'] - current_data['bear_20_sma']) / current_data['bear_20_sma']) * 100)
    print(f"  Status: {bear_vs_sma} 20-day SMA by {bear_distance:.2f}%")
    print()
    
    # Check for crossover signals
    crossover_alerts = check_crossover_signals(bull_symbol, bear_symbol)
    
    if crossover_alerts:
        print("CROSSOVER ALERTS:")
        print("-" * 40)
        for alert in crossover_alerts:
            print(alert)
        print()
    else:
        print("No immediate crossover signals detected")
        print()
    
    # Trading implications
    print("TRADING IMPLICATIONS:")
    print("-" * 40)
    if "HOLD" in action:
        print(f"Strategy suggests: {action}")
    
    print(f"\nNote: Trades would typically be executed near market close based on end-of-day prices.")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Show current market status for trading strategy')
    parser.add_argument('--bull', default='QLD', help='Bull market ETF symbol (default: QLD)')
    parser.add_argument('--bear', default='QID', help='Bear market ETF symbol (default: QID)')
    parser.add_argument('--current', action='store_true', help='Use current/live prices from CSV files (default: use historical close prices)')
    parser.add_argument('--fetch-current', action='store_true', help='Auto-fetch current prices from yfinance and save to CSV files')
    parser.add_argument('--current-dir', default='current_prices', help='Directory containing current price CSV files (default: current_prices)')
    
    args = parser.parse_args()
    
    try:
        # Auto-fetch current prices if requested
        if args.fetch_current:
            symbols = ['SPY', args.bull, args.bear]
            fetch_and_save_current_prices(symbols, args.current_dir)
            use_current = True  # Automatically use the fetched prices
        else:
            use_current = args.current  # Only use current if --current flag is set
        
        display_market_report(args.bull, args.bear, use_current, args.current_dir)
    except Exception as e:
        print(f"Error fetching market data: {str(e)}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
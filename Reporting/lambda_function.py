import json
import boto3
import requests
import os
from datetime import datetime, timedelta
import time

def lambda_handler(event, context):
    """
    Simplified AWS Lambda function using direct Yahoo Finance API calls
    """
    
    try:
        print("Starting simplified trading report generation...")
        
        # Configuration from environment variables
        from_email = os.environ.get('FROM_EMAIL', 'alerts@yourdomain.com')
        to_email = os.environ.get('TO_EMAIL', 'your@email.com')
        bull_symbol = os.environ.get('BULL_SYMBOL', 'QLD')
        bear_symbol = os.environ.get('BEAR_SYMBOL', 'QID')
        
        # Generate the trading report
        report_html = generate_simple_trading_report(bull_symbol, bear_symbol)
        
        # Send email
        success = send_email_report(from_email, to_email, report_html, bull_symbol, bear_symbol)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Daily trading report sent successfully' if success else 'Failed to send report',
                'success': success,
                'timestamp': datetime.now().isoformat(),
                'symbols': f"{bull_symbol}/{bear_symbol}"
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }

def get_yahoo_data(symbol, period="1y"):
    """Get stock data directly from Yahoo Finance API"""
    try:
        # Yahoo Finance API endpoint
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            'period1': int((datetime.now() - timedelta(days=365)).timestamp()),
            'period2': int(datetime.now().timestamp()),
            'interval': '1d'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        result = data['chart']['result'][0]
        
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        # Convert to simple format
        prices = []
        for i, ts in enumerate(timestamps):
            if quotes['close'][i] is not None:
                prices.append({
                    'date': datetime.fromtimestamp(ts),
                    'close': quotes['close'][i],
                    'high': quotes['high'][i],
                    'low': quotes['low'][i],
                    'open': quotes['open'][i],
                    'volume': quotes['volume'][i]
                })
        
        return prices
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return []

def calculate_sma(prices, window):
    """Calculate Simple Moving Average"""
    if len(prices) < window:
        return None
    
    recent_closes = [p['close'] for p in prices[-window:]]
    return sum(recent_closes) / len(recent_closes)

def calculate_rsi(prices, window=10):
    """Calculate RSI"""
    if len(prices) < window + 1:
        return None
    
    closes = [p['close'] for p in prices[-(window+1):]]
    
    gains = []
    losses = []
    
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    if not gains or not losses:
        return None
    
    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_simple_trading_report(bull_symbol='QLD', bear_symbol='QID'):
    """Generate trading report using simple Yahoo Finance API"""
    print(f"Generating simplified report for {bull_symbol}/{bear_symbol}...")
    
    try:
        # Get data for all symbols
        print("Fetching SPY data...")
        spy_data = get_yahoo_data("SPY")
        print("Fetching bull data...")
        bull_data = get_yahoo_data(bull_symbol)
        print("Fetching bear data...")
        bear_data = get_yahoo_data(bear_symbol)
        
        if not spy_data or not bull_data or not bear_data:
            raise Exception("Failed to fetch required market data")
        
        # Calculate indicators
        current_spy = spy_data[-1]
        spy_200_sma = calculate_sma(spy_data, 200)
        
        current_bull = bull_data[-1]
        bull_20_sma = calculate_sma(bull_data, 20)
        bull_rsi = calculate_rsi(bull_data, 10)
        
        current_bear = bear_data[-1]
        bear_20_sma = calculate_sma(bear_data, 20)
        
        # Determine action
        action, reason = determine_simple_action(
            current_spy['close'], spy_200_sma,
            current_bull['close'], bull_20_sma, bull_rsi,
            current_bear['close'], bear_20_sma,
            bull_symbol, bear_symbol
        )
        
        # Generate HTML report
        html_report = generate_simple_html_report(
            current_spy, spy_200_sma,
            current_bull, bull_20_sma, bull_rsi,
            current_bear, bear_20_sma,
            action, reason, bull_symbol, bear_symbol
        )
        
        return html_report
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        raise

def determine_simple_action(spy_price, spy_200_sma, bull_price, bull_20_sma, bull_rsi, bear_price, bear_20_sma, bull_symbol, bear_symbol):
    """Simple algorithm implementation"""
    
    if spy_price > spy_200_sma:
        action = f'HOLD {bull_symbol}'
        reason = f'SPY above 200 SMA ({spy_price:.2f} > {spy_200_sma:.2f}), bull market'
    else:
        if bull_rsi and bull_rsi < 31:
            action = f'HOLD {bull_symbol}'
            reason = f'SPY below 200 SMA but {bull_symbol} oversold (RSI: {bull_rsi:.2f})'
        elif bull_price < bull_20_sma:
            action = f'HOLD {bear_symbol}'
            reason = f'{bull_symbol} below 20 SMA, defensive mode'
        elif bear_price < bear_20_sma:
            action = 'HOLD CASH'
            reason = f'{bear_symbol} below 20 SMA, hold cash'
        else:
            action = f'HOLD {bear_symbol}'
            reason = f'SPY below 200 SMA, defensive mode'
    
    return action, reason

def generate_simple_html_report(current_spy, spy_200_sma, current_bull, bull_20_sma, bull_rsi, current_bear, bear_20_sma, action, reason, bull_symbol, bear_symbol):
    """Generate simplified HTML report"""
    
    date_str = current_spy['date'].strftime('%A, %B %d, %Y')
    
    # Format RSI value properly
    rsi_display = f"{bull_rsi:.2f}" if bull_rsi is not None else "N/A"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Daily Trading Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ background: white; padding: 30px; border-radius: 8px; max-width: 800px; margin: 0 auto; }}
            .header {{ background: #007acc; color: white; padding: 20px; text-align: center; margin: -30px -30px 20px -30px; }}
            .action-box {{ background: #e7f3ff; border-left: 4px solid #007acc; padding: 20px; margin: 20px 0; }}
            .data-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
            .data-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; }}
            .positive {{ color: #28a745; }}
            .negative {{ color: #dc3545; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Daily Trading Report</h1>
                <p>{bull_symbol}/{bear_symbol} Strategy - {date_str}</p>
            </div>
            
            <div class="action-box">
                <h2>Current Recommendation: {action}</h2>
                <p><strong>Reasoning:</strong> {reason}</p>
            </div>
            
            <h3>Market Data</h3>
            <div class="data-grid">
                <div class="data-card">
                    <h3>SPY (Market Index)</h3>
                    <p><strong>Price:</strong> ${current_spy['close']:.2f}</p>
                    <p><strong>200-day SMA:</strong> ${spy_200_sma:.2f}</p>
                    <p><strong>Position:</strong> <span class="{'positive' if current_spy['close'] > spy_200_sma else 'negative'}">{'ABOVE' if current_spy['close'] > spy_200_sma else 'BELOW'}</span></p>
                </div>
                
                <div class="data-card">
                    <h3>{bull_symbol} (Bull ETF)</h3>
                    <p><strong>Price:</strong> ${current_bull['close']:.2f}</p>
                    <p><strong>20-day SMA:</strong> ${bull_20_sma:.2f}</p>
                    <p><strong>10-day RSI:</strong> {rsi_display}</p>
                    <p><strong>Position:</strong> <span class="{'positive' if current_bull['close'] > bull_20_sma else 'negative'}">{'ABOVE' if current_bull['close'] > bull_20_sma else 'BELOW'}</span></p>
                </div>
                
                <div class="data-card">
                    <h3>{bear_symbol} (Bear ETF)</h3>
                    <p><strong>Price:</strong> ${current_bear['close']:.2f}</p>
                    <p><strong>20-day SMA:</strong> ${bear_20_sma:.2f}</p>
                    <p><strong>Position:</strong> <span class="{'positive' if current_bear['close'] > bear_20_sma else 'negative'}">{'ABOVE' if current_bear['close'] > bear_20_sma else 'BELOW'}</span></p>
                </div>
            </div>
            
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 4px; margin: 20px 0;">
                <p><strong>Note:</strong> This is a simplified analysis using direct market data. 
                Always verify market conditions before making trading decisions.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def send_email_report(from_email, to_email, html_content, bull_symbol, bear_symbol):
    """Send HTML report via AWS SES"""
    
    try:
        ses_client = boto3.client('ses', region_name='us-east-1')
        
        subject = f"Your Daily Trading Update - {bull_symbol}/{bear_symbol} Strategy"
        
        response = ses_client.send_email(
            Source=from_email,
            Destination={
                'ToAddresses': [to_email]
            },
            Message={
                'Subject': {
                    'Data': subject,
                    'Charset': 'UTF-8'
                },
                'Body': {
                    'Html': {
                        'Data': html_content,
                        'Charset': 'UTF-8'
                    }
                }
            }
        )
        
        print(f"Email sent successfully. Message ID: {response['MessageId']}")
        return True
        
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import json
import pandas as pd
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from backtest_engine import (
    run_backtest, 
    format_results_for_display, 
    calculate_summary_stats,
    run_buy_and_hold
)
from parameter_manager import get_parameters, get_parameter_summary

# Ensure Flask can find templates from the correct location
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_backtest', methods=['POST'])
def run_backtest_web():
    try:
        # Get form data
        data = request.get_json()
        
        bull_symbol = data.get('bull', 'QLD')
        bear_symbol = data.get('bear', 'QID')
        start_date = data.get('start', '2020-01-10')
        end_date = data.get('end', datetime.today().strftime('%Y-%m-%d'))
        initial_cash = float(data.get('cash', 50000))
        recurring_investment = float(data.get('monthly', 0))
        show_spy = data.get('show_spy', False)
        show_buy_hold = data.get('show_buy_hold', False)
        use_ai_optimization = data.get('use_ai_optimization', False)
        
        # Get algorithm parameters (either optimized or default)
        algo_params = get_parameters(use_optimized=use_ai_optimization, 
                                   bull_symbol=bull_symbol, bear_symbol=bear_symbol)
        
        # Convert dates
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.today()
        
        # Run backtest with algorithm parameters
        results_df = run_backtest(start_date, end_date, initial_cash, recurring_investment, bull_symbol, bear_symbol, algo_params)
        
        # Format for display (same as terminal output)
        display_df = format_results_for_display(results_df.copy())
        
        # Calculate summary stats
        summary = calculate_summary_stats(results_df, initial_cash, recurring_investment, bull_symbol, bear_symbol, show_spy, show_buy_hold)
        
        # Prepare chart data
        chart_data = prepare_chart_data(results_df, initial_cash, bull_symbol, bear_symbol, show_spy, show_buy_hold)
        
        # Convert dataframe to HTML table with proper dark theme classes
        table_html = display_df.to_html(
            classes='table table-dark table-striped table-hover table-sm', 
            table_id='results-table', 
            escape=False, 
            index=False,
            border=0
        )
        
        # Get parameter summary for display
        param_summary = get_parameter_summary(use_optimized=use_ai_optimization)
        
        return jsonify({
            'success': True,
            'table_html': table_html,
            'chart_data': chart_data,
            'summary': summary,
            'bull_symbol': bull_symbol,
            'bear_symbol': bear_symbol,
            'use_ai_optimization': use_ai_optimization,
            'parameter_summary': param_summary,
            'algo_params': algo_params
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def prepare_chart_data(results_df, initial_cash, bull_symbol, bear_symbol, show_spy=False, show_buy_hold=False):
    """Prepare data for Chart.js"""
    
    # Main strategy data
    dates = results_df['Date'].dt.strftime('%Y-%m-%d').tolist()
    portfolio_values = results_df['Portfolio Value'].tolist()
    
    datasets = [{
        'label': f'{bull_symbol}/{bear_symbol} Strategy',
        'data': portfolio_values,
        'borderColor': '#00ff00',  # lime green
        'backgroundColor': 'rgba(0, 255, 0, 0.1)',
        'borderWidth': 3,
        'fill': False
    }]
    
    # Add comparison datasets if requested
    if show_spy or show_buy_hold:
        start_date = results_df['Date'].iloc[0]
        end_date = results_df['Date'].iloc[-1]
        
        if show_spy:
            spy_results = run_buy_and_hold('SPY', start_date, end_date, initial_cash)
            if not spy_results.empty:
                # Align dates with main strategy
                spy_aligned = align_comparison_data(spy_results, dates)
                datasets.append({
                    'label': 'Buy & Hold SPY',
                    'data': spy_aligned,
                    'borderColor': '#ffffff',  # white
                    'backgroundColor': 'rgba(255, 255, 255, 0.1)',
                    'borderWidth': 2,
                    'borderDash': [5, 5],
                    'fill': False
                })
        
        if show_buy_hold:
            bull_results = run_buy_and_hold(bull_symbol, start_date, end_date, initial_cash)
            if not bull_results.empty:
                # Align dates with main strategy
                bull_aligned = align_comparison_data(bull_results, dates)
                datasets.append({
                    'label': f'Buy & Hold {bull_symbol}',
                    'data': bull_aligned,
                    'borderColor': '#ff0000',  # red
                    'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                    'borderWidth': 2,
                    'borderDash': [5, 5],
                    'fill': False
                })
    
    return {
        'labels': dates,
        'datasets': datasets
    }

def align_comparison_data(comparison_df, target_dates):
    """Align comparison data with main strategy dates"""
    comparison_df = comparison_df.copy()
    comparison_df['Date'] = comparison_df['Date'].dt.strftime('%Y-%m-%d')
    
    aligned_data = []
    comparison_dict = dict(zip(comparison_df['Date'], comparison_df['Portfolio Value']))
    
    for date in target_dates:
        if date in comparison_dict:
            aligned_data.append(comparison_dict[date])
        else:
            # Use the most recent available value
            available_dates = [d for d in comparison_dict.keys() if d <= date]
            if available_dates:
                most_recent = max(available_dates)
                aligned_data.append(comparison_dict[most_recent])
            else:
                aligned_data.append(None)
    
    return aligned_data

if __name__ == '__main__':
    app.run(debug=True, port=5000)
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse
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

def plot_results(results_df, initial_cash, recurring_investment, bull_symbol='TQQQ', bear_symbol='SQQQ', 
                 show_spy=False, show_buy_hold=False):
    import pandas as pd
    plot_df = results_df.copy()
    plot_df['Date'] = pd.to_datetime(plot_df['Date'])
    plot_df['Portfolio Value'] = plot_df['Portfolio Value'].replace('[\\$,]', '', regex=True).astype(float)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Plot main strategy
    ax.plot(plot_df['Date'], plot_df['Portfolio Value'], 
            label=f'{bull_symbol}/{bear_symbol} Strategy', color='lime', linewidth=3)
    
    # Add buy & hold comparisons if requested
    if show_spy or show_buy_hold:
        start_date = plot_df['Date'].iloc[0]
        end_date = plot_df['Date'].iloc[-1]
        
        if show_spy:
            spy_results = run_buy_and_hold('SPY', start_date, end_date, initial_cash)
            if not spy_results.empty:
                ax.plot(spy_results['Date'], spy_results['Portfolio Value'], 
                        label='Buy & Hold SPY', color='white', linewidth=2, linestyle='--')
        
        if show_buy_hold:
            bull_results = run_buy_and_hold(bull_symbol, start_date, end_date, initial_cash)
            if not bull_results.empty:
                ax.plot(bull_results['Date'], bull_results['Portfolio Value'], 
                        label=f'Buy & Hold {bull_symbol}', color='red', linewidth=2, linestyle='--')
    
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:,.0f}".format(x)))
    
    # Dynamic title based on what's being shown
    title_parts = [f'{bull_symbol}/{bear_symbol} Strategy']
    if show_spy or show_buy_hold:
        title_parts.append('vs Buy & Hold')
    title = ' '.join(title_parts) + f' - Initial: ${initial_cash:,.0f}'
    if recurring_investment > 0:
        # Calculate total invested from the results dataframe
        if 'Total Invested' in plot_df.columns:
            total_invested_str = results_df['Total Invested'].iloc[-1]
            total_invested = float(total_invested_str.replace('$', '').replace(',', ''))
            title += f', Total Invested: ${total_invested:,.0f} (${recurring_investment:,.0f}/month on 1st trading day)'
    
    plt.title(title, color='white', fontsize=14, fontweight='bold')
    plt.xlabel('Date', color='white')
    plt.ylabel('Value ($)', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=11)
    plt.grid(True, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Run trading strategy with configurable bull/bear ETFs')
    parser.add_argument('--bull', default='TQQQ', help='Bull market ETF symbol (default: TQQQ)')
    parser.add_argument('--bear', default='SQQQ', help='Bear market ETF symbol (default: SQQQ)')
    parser.add_argument('--start', default='2020-01-10', help='Start date (YYYY-MM-DD, default: 2020-01-10)')
    parser.add_argument('--end', default=None, help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--cash', type=float, default=50000, help='Initial cash (default: 50000)')
    parser.add_argument('--monthly', type=float, default=0, help='Monthly recurring investment on 1st trading day of each month (default: 0)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting results')
    parser.add_argument('--show-spy', action='store_true', help='Include SPY buy & hold comparison on chart')
    parser.add_argument('--show-buy-hold', action='store_true', help='Include bull ETF buy & hold comparison on chart')
    parser.add_argument('--use-ai', action='store_true', help='Use AI-optimized parameters (requires optimized_parameters.json)')
    
    args = parser.parse_args()
    
    # Get algorithm parameters (either optimized or default)
    algo_params = get_parameters(use_optimized=args.use_ai, 
                               bull_symbol=args.bull, bear_symbol=args.bear)
    
    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else datetime.today()
    
    print(f"Running {args.bull}/{args.bear} Strategy Backtest {' (AI-Optimized)' if args.use_ai else ''}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Initial Cash: ${args.cash:,.2f}")
    print(f"Bull ETF: {args.bull}")
    print(f"Bear ETF: {args.bear}")
    if args.monthly > 0:
        print(f"Monthly Investment: ${args.monthly:,.2f} (on 1st trading day of each month)")
    
    # Show parameter information
    param_summary = get_parameter_summary(use_optimized=args.use_ai)
    print(f"\nParameters: {param_summary.strip()}")
    
    print("\n" + "="*60)
    
    # Run Backtest
    backtest_results = run_backtest(start_date, end_date, args.cash, args.monthly, args.bull, args.bear, algo_params)
    
    # Format and Display Results
    display_results = format_results_for_display(backtest_results.copy())
    print(display_results.to_string())
    
    # Plot the Results (unless --no-plot is specified)
    if not args.no_plot:
        plot_results(display_results, args.cash, args.monthly, args.bull, args.bear, 
                    args.show_spy, args.show_buy_hold)
    
    # Calculate and show summary
    summary = calculate_summary_stats(backtest_results, args.cash, args.monthly, args.bull, args.bear, args.show_spy, args.show_buy_hold)
    
    print("\n" + "="*80)
    print(f"FINAL RESULTS - {args.bull}/{args.bear} Strategy")
    print("="*80)
    print(f"Strategy Performance:")
    print(f"  Initial Investment: ${summary['initial_investment']:,.2f}")
    if summary['monthly_investment'] > 0:
        print(f"  Total Invested (Initial + Monthly): ${summary['total_invested']:,.2f}")
    else:
        print(f"  Total Invested: ${summary['total_invested']:,.2f}")
    print(f"  Final Portfolio Value: ${summary['final_portfolio_value']:,.2f}")
    print(f"  Net Profit vs Invested: ${summary['net_profit']:,.2f}")
    print(f"  ROI on Invested Capital: {summary['roi_on_invested']:+.2f}%")
    print(f"  Total Return vs Initial: {summary['total_return']:+.2f}%")
    
    # Add buy & hold comparisons if requested
    if 'comparisons' in summary:
        print("\nBuy & Hold Comparisons:")
        
        if 'spy' in summary['comparisons']:
            spy = summary['comparisons']['spy']
            print(f"  SPY Buy & Hold: ${spy['final_value']:,.2f} ({spy['return']:+.2f}%)")
            print(f"  Strategy vs SPY: {spy['vs_strategy']:+.2f}%")
        
        if args.bull in summary['comparisons']:
            bull = summary['comparisons'][args.bull]
            print(f"  {args.bull} Buy & Hold: ${bull['final_value']:,.2f} ({bull['return']:+.2f}%)")
            print(f"  Strategy vs {args.bull}: {bull['vs_strategy']:+.2f}%")
    
    print("="*80)

if __name__ == "__main__":
    main()

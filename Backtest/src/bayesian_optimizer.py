#!/usr/bin/env python3
"""
Bayesian Optimization for Trading Algorithm Parameters

This script uses Bayesian optimization to find optimal SMA and RSI thresholds
for the trading algorithm to maximize the Sharpe ratio.

Optimized Parameters:
- SPY SMA window (currently 200)
- TQQQ/SQQQ SMA window (currently 20) 
- TQQQ/SQQQ RSI window (currently 10)
- TQQQ/SQQQ RSI threshold (currently 31)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple
import argparse

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

# Check if scikit-optimize is available
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
except ImportError:
    print("Installing required dependencies...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-optimize"])
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
from backtest_engine import get_data, calculate_sma, calculate_rsi


class ParameterOptimizer:
    """
    Bayesian optimization for trading algorithm parameters
    """
    
    def __init__(self, 
                 bull_symbol: str = 'TQQQ',
                 bear_symbol: str = 'SQQQ',
                 train_years: int = 3,
                 validation_split: float = 0.2):
        """
        Initialize the optimizer
        
        Args:
            bull_symbol: Bull market ETF symbol
            bear_symbol: Bear market ETF symbol  
            train_years: Number of years to use for training data
            validation_split: Fraction of training data to use for validation
        """
        self.bull_symbol = bull_symbol
        self.bear_symbol = bear_symbol
        self.train_years = train_years
        self.validation_split = validation_split
        
        # Define parameter search space - EXPANDED RANGES for more exploration!
        self.dimensions = [
            Integer(10, 500, name='spy_sma_window'),      # SPY SMA window (default 200) - WAY MORE RANGE!
            Integer(3, 100, name='etf_sma_window'),       # ETF SMA window (default 20) - from very short to very long
            Integer(2, 50, name='etf_rsi_window'),        # ETF RSI window (default 10) - much wider range
            Real(10.0, 70.0, name='etf_rsi_threshold')    # ETF RSI threshold (default 31) - MUCH wider range!
        ]
        
        self.best_params = None
        self.optimization_history = []
        
    def load_and_prepare_data(self) -> Dict:
        """Load and prepare historical data for optimization"""
        print("Loading historical data...")
        
        # Load data
        spy_data = get_data("SPY")
        bull_data = get_data(self.bull_symbol)
        bear_data = get_data(self.bear_symbol)
        
        # Determine date range for training (last N years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.train_years * 365)
        
        print(f"Training data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Filter data to training period
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        spy_data = spy_data[(spy_data.index >= start_date) & (spy_data.index <= end_date)]
        bull_data = bull_data[(bull_data.index >= start_date) & (bull_data.index <= end_date)]
        bear_data = bear_data[(bear_data.index >= start_date) & (bear_data.index <= end_date)]
        
        return {
            'spy_data': spy_data,
            'bull_data': bull_data, 
            'bear_data': bear_data,
            'start_date': start_date,
            'end_date': end_date
        }
    
    def split_data_for_validation(self, data_dict: Dict) -> Tuple[Dict, Dict]:
        """
        Split data into training and validation sets
        
        Returns:
            Tuple of (training_data, validation_data)
        """
        spy_data = data_dict['spy_data']
        bull_data = data_dict['bull_data']
        bear_data = data_dict['bear_data']
        
        # Calculate split point (avoid look-ahead bias by using earlier data for training)
        total_days = len(spy_data)
        train_days = int(total_days * (1 - self.validation_split))
        
        train_data = {
            'spy_data': spy_data.iloc[:train_days],
            'bull_data': bull_data.iloc[:train_days],
            'bear_data': bear_data.iloc[:train_days],
            'start_date': spy_data.index[0],
            'end_date': spy_data.index[train_days-1]
        }
        
        val_data = {
            'spy_data': spy_data.iloc[train_days:],
            'bull_data': bull_data.iloc[train_days:],
            'bear_data': bear_data.iloc[train_days:], 
            'start_date': spy_data.index[train_days],
            'end_date': spy_data.index[-1]
        }
        
        print(f"Training period: {train_data['start_date'].strftime('%Y-%m-%d')} to {train_data['end_date'].strftime('%Y-%m-%d')}")
        print(f"Validation period: {val_data['start_date'].strftime('%Y-%m-%d')} to {val_data['end_date'].strftime('%Y-%m-%d')}")
        
        return train_data, val_data
    
    def calculate_indicators_with_params(self, data_dict: Dict, params: Dict) -> Dict:
        """Calculate technical indicators with given parameters"""
        spy_data = data_dict['spy_data'].copy()
        bull_data = data_dict['bull_data'].copy()
        bear_data = data_dict['bear_data'].copy()
        
        # Calculate indicators with custom parameters
        spy_data['SMA'] = calculate_sma(spy_data, params['spy_sma_window'])
        
        bull_data['RSI'] = calculate_rsi(bull_data, params['etf_rsi_window'])
        bull_data['SMA'] = calculate_sma(bull_data, params['etf_sma_window'])
        
        bear_data['RSI'] = calculate_rsi(bear_data, params['etf_rsi_window'])
        bear_data['SMA'] = calculate_sma(bear_data, params['etf_sma_window'])
        
        return {
            'spy_data': spy_data,
            'bull_data': bull_data,
            'bear_data': bear_data,
            'start_date': data_dict['start_date'],
            'end_date': data_dict['end_date']
        }
    
    def determine_action_with_params(self, current_date, spy_data, bull_data, bear_data, 
                                   holdings, params):
        """
        Determine trading action with custom parameters (modified from backtest_engine)
        """
        spy_price = spy_data.loc[current_date]['Close']
        spy_sma = spy_data.loc[current_date]['SMA']
        
        bull_price = bull_data.loc[current_date]['Close']
        bull_rsi = bull_data.loc[current_date]['RSI']
        bull_sma = bull_data.loc[current_date]['SMA']
        
        bear_price = bear_data.loc[current_date]['Close']
        bear_rsi = bear_data.loc[current_date]['RSI']
        bear_sma = bear_data.loc[current_date]['SMA']
        
        action = 'HOLD'
        
        if spy_price > spy_sma:
            if holdings[self.bull_symbol] > 0:
                action = 'HOLD'
            else:
                action = f'BUY_{self.bull_symbol}_SELL_{self.bear_symbol}'
        else:
            if bull_rsi < params['etf_rsi_threshold']:
                if holdings[self.bull_symbol] > 0:
                    action = 'HOLD'
                else:
                    action = f'BUY_{self.bull_symbol}_SELL_{self.bear_symbol}'
            elif bull_price < bull_sma:
                if holdings[self.bull_symbol] > 0:
                    action = f'SELL_{self.bull_symbol}_BUY_{self.bear_symbol}'
                else:
                    action = 'HOLD'
            elif bear_price < bear_sma:
                if holdings[self.bear_symbol] > 0:
                    action = f'SELL_{self.bear_symbol}'
                else:
                    action = 'HOLD'
            else:
                action = 'HOLD'
        
        return action
    
    def execute_action_simplified(self, action, prices, holdings):
        """Simplified action execution for optimization"""
        
        if action == f'BUY_{self.bull_symbol}_SELL_{self.bear_symbol}':
            # Sell all bear, buy all bull
            cash = holdings['Cash'] + (holdings[self.bear_symbol] * prices[self.bear_symbol])
            holdings['Cash'] = 0.0
            holdings[self.bear_symbol] = 0.0
            holdings[self.bull_symbol] = cash / prices[self.bull_symbol]
            
        elif action == f'SELL_{self.bull_symbol}_BUY_{self.bear_symbol}':
            # Sell all bull, buy all bear
            cash = holdings['Cash'] + (holdings[self.bull_symbol] * prices[self.bull_symbol])
            holdings['Cash'] = 0.0
            holdings[self.bull_symbol] = 0.0
            holdings[self.bear_symbol] = cash / prices[self.bear_symbol]
            
        elif action == f'SELL_{self.bear_symbol}':
            # Sell all bear, hold cash
            cash = holdings['Cash'] + (holdings[self.bear_symbol] * prices[self.bear_symbol])
            holdings['Cash'] = cash
            holdings[self.bear_symbol] = 0.0
        
        return holdings
    
    def run_backtest_with_params(self, data_dict: Dict, params: Dict, 
                                initial_cash: float = 50000) -> float:
        """
        Run backtest with given parameters and return Sharpe ratio
        
        Returns:
            Sharpe ratio (negative for minimization)
        """
        # Calculate indicators
        data_with_indicators = self.calculate_indicators_with_params(data_dict, params)
        
        spy_data = data_with_indicators['spy_data']
        bull_data = data_with_indicators['bull_data']
        bear_data = data_with_indicators['bear_data']
        
        # Drop NaN rows (from indicator calculations)
        max_window = max(params['spy_sma_window'], params['etf_sma_window'], params['etf_rsi_window'])
        spy_data = spy_data.iloc[max_window:]
        bull_data = bull_data.iloc[max_window:]
        bear_data = bear_data.iloc[max_window:]
        
        # Find common dates
        common_dates = spy_data.index.intersection(bull_data.index).intersection(bear_data.index)
        
        if len(common_dates) < 50:  # Need at least 50 days of data
            return -999.0  # Very bad Sharpe ratio
        
        # Initialize portfolio
        holdings = {
            'Cash': initial_cash,
            self.bull_symbol: 0.0,
            self.bear_symbol: 0.0
        }
        
        portfolio_values = []
        daily_returns = []
        
        for i, current_date in enumerate(common_dates):
            try:
                prices = {
                    'SPY': spy_data.loc[current_date]['Close'],
                    self.bull_symbol: bull_data.loc[current_date]['Close'],
                    self.bear_symbol: bear_data.loc[current_date]['Close']
                }
                
                # Determine action
                action = self.determine_action_with_params(
                    current_date, spy_data, bull_data, bear_data, holdings, params
                )
                
                # Execute action
                holdings = self.execute_action_simplified(action, prices, holdings)
                
                # Calculate portfolio value
                portfolio_value = (holdings['Cash'] + 
                                 holdings[self.bull_symbol] * prices[self.bull_symbol] +
                                 holdings[self.bear_symbol] * prices[self.bear_symbol])
                
                portfolio_values.append(portfolio_value)
                
                # Calculate daily return
                if i > 0:
                    daily_return = (portfolio_value / portfolio_values[i-1]) - 1
                    daily_returns.append(daily_return)
                    
            except Exception as e:
                # Skip problematic dates
                continue
        
        if len(daily_returns) < 10:
            return -999.0
        
        # Calculate Sharpe ratio (annualized)
        daily_returns = np.array(daily_returns)
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        if std_return == 0:
            return -999.0
        
        # Annualized Sharpe ratio (assuming 252 trading days)
        sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252))
        
        return sharpe_ratio
    
    def objective_function(self, train_data: Dict, val_data: Dict):
        """
        Create objective function for optimization
        
        Returns:
            Function that takes parameter list and returns negative Sharpe ratio
        """
        @use_named_args(self.dimensions)
        def objective(**params):
            # Run backtest on training data
            train_sharpe = self.run_backtest_with_params(train_data, params)
            
            # Run backtest on validation data
            val_sharpe = self.run_backtest_with_params(val_data, params)
            
            # Use average of train and validation Sharpe ratios
            # This helps prevent overfitting
            avg_sharpe = (train_sharpe + val_sharpe) / 2.0
            
            # Store results
            self.optimization_history.append({
                'params': params.copy(),
                'train_sharpe': train_sharpe,
                'val_sharpe': val_sharpe,
                'avg_sharpe': avg_sharpe
            })
            
            print(f"Params: {params}")
            print(f"Train Sharpe: {train_sharpe:.4f}, Val Sharpe: {val_sharpe:.4f}, Avg: {avg_sharpe:.4f}")
            print("-" * 50)
            
            # Return negative Sharpe ratio for minimization
            return -avg_sharpe
        
        return objective
    
    def optimize(self, n_calls: int = 50, random_state: int = 42) -> Dict:
        """
        Run Bayesian optimization
        
        Args:
            n_calls: Number of optimization iterations
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with best parameters and results
        """
        print("Starting Bayesian optimization...")
        print(f"Bull Symbol: {self.bull_symbol}")
        print(f"Bear Symbol: {self.bear_symbol}")
        print(f"Training Years: {self.train_years}")
        print(f"Optimization Calls: {n_calls}")
        print("=" * 60)
        
        # Load and prepare data
        data_dict = self.load_and_prepare_data()
        
        # Split into training and validation
        train_data, val_data = self.split_data_for_validation(data_dict)
        
        # Create objective function
        objective = self.objective_function(train_data, val_data)
        
        # Run optimization
        print("Running Bayesian optimization...")
        # Ensure n_calls is at least 10 for scikit-optimize
        n_calls = max(n_calls, 10)
        n_initial_points = min(10, n_calls)
        
        result = gp_minimize(
            func=objective,
            dimensions=self.dimensions,
            n_calls=n_calls,
            random_state=random_state,
            acq_func='EI',  # Expected Improvement
            n_initial_points=n_initial_points
        )
        
        # Extract best parameters
        best_params = {}
        for i, dim in enumerate(self.dimensions):
            best_params[dim.name] = result.x[i]
        
        best_sharpe = -result.fun
        
        self.best_params = best_params
        
        print("=" * 60)
        print("OPTIMIZATION COMPLETED!")
        print("=" * 60)
        print("Best Parameters:")
        print(f"  SPY SMA Window: {best_params['spy_sma_window']}")
        print(f"  ETF SMA Window: {best_params['etf_sma_window']}")
        print(f"  ETF RSI Window: {best_params['etf_rsi_window']}")
        print(f"  ETF RSI Threshold: {best_params['etf_rsi_threshold']:.2f}")
        print(f"Best Average Sharpe Ratio: {best_sharpe:.4f}")
        
        # Find best individual results
        if self.optimization_history:
            best_train = max(self.optimization_history, key=lambda x: x['train_sharpe'])
            best_val = max(self.optimization_history, key=lambda x: x['val_sharpe'])
            
            print(f"Best Training Sharpe: {best_train['train_sharpe']:.4f}")
            print(f"Best Validation Sharpe: {best_val['val_sharpe']:.4f}")
        
        return {
            'best_params': best_params,
            'best_sharpe': best_sharpe,
            'optimization_result': result,
            'history': self.optimization_history,
            'default_comparison': self.get_default_performance(train_data, val_data)
        }
    
    def get_default_performance(self, train_data: Dict, val_data: Dict) -> Dict:
        """Get performance with default parameters for comparison"""
        default_params = {
            'spy_sma_window': 200,
            'etf_sma_window': 20,
            'etf_rsi_window': 10,
            'etf_rsi_threshold': 31.0
        }
        
        train_sharpe = self.run_backtest_with_params(train_data, default_params)
        val_sharpe = self.run_backtest_with_params(val_data, default_params)
        
        return {
            'params': default_params,
            'train_sharpe': train_sharpe,
            'val_sharpe': val_sharpe,
            'avg_sharpe': (train_sharpe + val_sharpe) / 2.0
        }
    
    def save_results(self, results: Dict, filename: str = "optimized_parameters.json"):
        """Save optimization results to JSON file"""
        
        # Make results JSON serializable
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'bull_symbol': self.bull_symbol,
            'bear_symbol': self.bear_symbol,
            'train_years': self.train_years,
            'best_params': results['best_params'],
            'best_sharpe': results['best_sharpe'],
            'default_comparison': results['default_comparison'],
            'optimization_summary': {
                'total_iterations': len(self.optimization_history),
                'best_avg_sharpe': results['best_sharpe'],
                'improvement_vs_default': results['best_sharpe'] - results['default_comparison']['avg_sharpe']
            }
        }
        
        # Save to the data directory
        base_dir = os.path.dirname(os.path.dirname(__file__))
        filepath = os.path.join(base_dir, 'data', filename)
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, cls=NumpyEncoder)
        
        print(f"Results saved to: {filepath}")
        
        return filepath


def main():
    """Main function for running the optimization"""
    parser = argparse.ArgumentParser(description='Optimize trading algorithm parameters using Bayesian optimization')
    parser.add_argument('--bull', default='TQQQ', help='Bull market ETF symbol (default: TQQQ)')
    parser.add_argument('--bear', default='SQQQ', help='Bear market ETF symbol (default: SQQQ)')
    parser.add_argument('--train-years', type=int, default=3, help='Years of training data (default: 3)')
    parser.add_argument('--n-calls', type=int, default=50, help='Number of optimization iterations (default: 50)')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio (default: 0.2)')
    parser.add_argument('--output', default='optimized_parameters.json', help='Output filename (default: optimized_parameters.json)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = ParameterOptimizer(
        bull_symbol=args.bull,
        bear_symbol=args.bear,
        train_years=args.train_years,
        validation_split=args.validation_split
    )
    
    try:
        # Run optimization
        results = optimizer.optimize(n_calls=args.n_calls, random_state=args.random_seed)
        
        # Save results
        optimizer.save_results(results, args.output)
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Default Parameters Sharpe: {results['default_comparison']['avg_sharpe']:.4f}")
        print(f"Optimized Parameters Sharpe: {results['best_sharpe']:.4f}")
        improvement = results['best_sharpe'] - results['default_comparison']['avg_sharpe']
        print(f"Improvement: {improvement:+.4f} ({improvement/abs(results['default_comparison']['avg_sharpe'])*100:+.1f}%)")
        
        print(f"\nOptimized parameters saved to: {args.output}")
        print("You can now use these parameters in the main trading application!")
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        raise


if __name__ == "__main__":
    main()
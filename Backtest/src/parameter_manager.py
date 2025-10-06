#!/usr/bin/env python3
"""
Parameter Manager for Trading Algorithm

This module handles loading and managing optimized algorithm parameters.
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional


def load_optimized_parameters(filename: str = "optimized_parameters.json") -> Optional[Dict]:
    """
    Load optimized parameters from JSON file
    
    Args:
        filename: Name of the parameter file
        
    Returns:
        Dictionary with optimized parameters, or None if file doesn't exist
    """
    # Look for parameter files in the data directory
    base_dir = os.path.dirname(os.path.dirname(__file__))
    filepath = os.path.join(base_dir, 'data', filename)
    
    if not os.path.exists(filepath):
        print(f"No optimized parameters file found at: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract just the parameters we need
        if 'best_params' in data:
            params = data['best_params']
            
            print(f"Loaded optimized parameters from {filename}:")
            print(f"  SPY SMA Window: {params.get('spy_sma_window', 'N/A')}")
            print(f"  ETF SMA Window: {params.get('etf_sma_window', 'N/A')}")
            print(f"  ETF RSI Window: {params.get('etf_rsi_window', 'N/A')}")
            print(f"  ETF RSI Threshold: {params.get('etf_rsi_threshold', 'N/A')}")
            
            if 'timestamp' in data:
                print(f"  Optimization Date: {data['timestamp']}")
            if 'best_sharpe' in data:
                print(f"  Best Sharpe Ratio: {data['best_sharpe']:.4f}")
            
            return params
        else:
            print(f"Invalid parameter file format: {filepath}")
            return None
            
    except Exception as e:
        print(f"Error loading parameters from {filepath}: {str(e)}")
        return None


def get_default_parameters() -> Dict:
    """
    Get default algorithm parameters
    
    Returns:
        Dictionary with default parameters
    """
    return {
        'spy_sma_window': 200,
        'etf_sma_window': 20,
        'etf_rsi_window': 10,
        'etf_rsi_threshold': 31
    }


def get_parameters(use_optimized: bool = False, filename: str = "optimized_parameters.json", 
                   bull_symbol: str = None, bear_symbol: str = None) -> Dict:
    """
    Get algorithm parameters - either optimized or default
    
    Args:
        use_optimized: Whether to use optimized parameters
        filename: Name of the optimized parameter file
        bull_symbol: Bull ETF symbol (for auto-selecting parameter file)
        bear_symbol: Bear ETF symbol (for auto-selecting parameter file)
        
    Returns:
        Dictionary with algorithm parameters
    """
    if use_optimized:
        # Try to auto-select parameter file based on ETF symbols
        if bull_symbol and bear_symbol and filename == "optimized_parameters.json":
            etf_specific_file = f"{bull_symbol.lower()}_{bear_symbol.lower()}_params.json"
            base_dir = os.path.dirname(os.path.dirname(__file__))
            etf_specific_path = os.path.join(base_dir, 'data', etf_specific_file)
            if os.path.exists(etf_specific_path):
                filename = etf_specific_file
                print(f"Using ETF-specific parameters: {filename}")
        
        optimized_params = load_optimized_parameters(filename)
        if optimized_params is not None:
            return optimized_params
        else:
            print("Falling back to default parameters...")
    
    return get_default_parameters()


def compare_parameters(filename: str = "optimized_parameters.json") -> None:
    """
    Compare optimized parameters with defaults
    
    Args:
        filename: Name of the parameter file
    """
    optimized = load_optimized_parameters(filename)
    default = get_default_parameters()
    
    if optimized is None:
        print("No optimized parameters found for comparison.")
        return
    
    print("\n" + "="*50)
    print("PARAMETER COMPARISON")
    print("="*50)
    print(f"{'Parameter':<20} {'Default':<10} {'Optimized':<12} {'Change':<10}")
    print("-" * 50)
    
    for param in default.keys():
        default_val = default[param]
        optimized_val = optimized.get(param, 'N/A')
        
        if isinstance(default_val, (int, float)) and isinstance(optimized_val, (int, float)):
            if isinstance(default_val, int) and isinstance(optimized_val, float):
                # Handle case where default is int but optimized is float
                if optimized_val == int(optimized_val):
                    optimized_val = int(optimized_val)
            
            if default_val != optimized_val:
                change = f"{((optimized_val - default_val) / default_val * 100):+.1f}%"
            else:
                change = "No change"
        else:
            change = "N/A"
            
        print(f"{param.replace('_', ' ').title():<20} {default_val:<10} {optimized_val:<12} {change:<10}")


def save_comparison_results(optimization_results: Dict, filename: str = "parameter_comparison.json") -> str:
    """
    Save comparison results between optimized and default parameters
    
    Args:
        optimization_results: Results from the Bayesian optimization
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'default_performance': optimization_results['default_comparison'],
        'optimized_performance': {
            'params': optimization_results['best_params'],
            'sharpe_ratio': optimization_results['best_sharpe']
        },
        'improvement': {
            'sharpe_diff': optimization_results['best_sharpe'] - optimization_results['default_comparison']['avg_sharpe'],
            'percent_improvement': ((optimization_results['best_sharpe'] - optimization_results['default_comparison']['avg_sharpe']) / 
                                  abs(optimization_results['default_comparison']['avg_sharpe']) * 100)
        }
    }
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    filepath = os.path.join(base_dir, 'data', filename)
    with open(filepath, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"Parameter comparison saved to: {filepath}")
    return filepath


def get_parameter_summary(use_optimized: bool = False, filename: str = "optimized_parameters.json") -> str:
    """
    Get a summary string of the parameters being used
    
    Args:
        use_optimized: Whether optimized parameters are being used
        filename: Name of the parameter file
        
    Returns:
        Summary string describing the parameters
    """
    params = get_parameters(use_optimized, filename)
    
    param_type = "AI-Optimized" if use_optimized else "Default"
    
    summary = f"{param_type} Parameters:\n"
    summary += f"  SPY SMA: {params['spy_sma_window']} days\n"
    summary += f"  ETF SMA: {params['etf_sma_window']} days\n"
    summary += f"  ETF RSI: {params['etf_rsi_window']} days (threshold: {params['etf_rsi_threshold']})\n"
    
    return summary


if __name__ == "__main__":
    """Test the parameter manager"""
    print("Testing Parameter Manager...")
    
    # Test default parameters
    print("\nDefault Parameters:")
    default_params = get_default_parameters()
    for key, value in default_params.items():
        print(f"  {key}: {value}")
    
    # Test loading optimized parameters (if available)
    print("\nTrying to load optimized parameters...")
    optimized_params = load_optimized_parameters()
    
    if optimized_params:
        print("\nParameter comparison:")
        compare_parameters()
    else:
        print("No optimized parameters found. Run bayesian_optimizer.py first!")
    
    # Test parameter summary
    print("\nParameter Summary:")
    print(get_parameter_summary(use_optimized=False))
    
    if optimized_params:
        print("AI-Optimized Summary:")
        print(get_parameter_summary(use_optimized=True))
# Trading Strategy

A leveraged ETF trading system that attempts to time market entries and exits. Shows solid performance in backtests, though real-world results may vary.

## How It Works

The strategy is straightforward: hold leveraged bull ETFs (QLD, TQQQ, etc.) during favorable market conditions and switch to bear ETFs or cash during downturns. Uses SPY's 200-day moving average as the primary signal, combined with RSI and shorter moving averages to reduce false signals.

## Project Structure

This repo contains two main modules:

### Backtest Module
Handles strategy testing and parameter optimization. Includes a web interface for running backtests, performance comparisons against benchmarks, and visualization tools. Also features Bayesian optimization for parameter tuning, though the default settings work well for most cases.

To get started:
```bash
cd Backtest/
./start_web.sh  # Web interface at localhost:5000
```

### Reporting Module  
Provides real-time market analysis and automated reporting. Can be deployed on AWS Lambda to send daily email reports with trading recommendations (hold bull ETF, bear ETF, or cash). Useful for staying on top of market conditions without constantly checking.

To test locally:
```bash
cd Reporting/
./run_algo.sh --fetch-current
```

## Screenshots

### Backtest Web Interface
![Backtest Interface](images/Backtest%20Photo.png)

### Performance Charts
![Backtest Results](images/Backtest%20Graph.png)

### Daily Email Reports
![Email Report](images/report%20screenshot.png)

## Disclaimer

**This project is for educational and entertainment purposes only.** Past performance does not guarantee future results. Real-world trading involves significant risks not captured in backtests:

- **Slippage**: Actual execution prices may differ from backtested prices due to market volatility and liquidity
- **Timing**: The algorithm uses end-of-day closing prices, implying trades would be executed near market close in real scenarios
- **Market conditions**: Historical patterns may not repeat, and leveraged ETFs carry amplified risk during volatile periods
- **Transaction costs**: Backtests may not fully account for fees, taxes, and bid-ask spreads

**Never risk more than you can afford to lose. Consult a financial advisor before making investment decisions.**

See the README files in each folder for detailed setup instructions. The backtest module is probably more interesting to start with.

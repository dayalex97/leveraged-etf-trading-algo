# Leveraged ETF Trading Strategy

This algorithm leverages the amplified returns of leveraged ETFs to capture outsized market gains. The strategy employs technical indicators to identify bear market conditions for strategic long-term positioning, then utilizes Simple Moving Averages (SMA) and Relative Strength Index (RSI) to time transitions into inverse leveraged positions during market downturns. The core objective is optimizing market timing to maximize returns while managing downside risk.

The system operates as both a web application and command-line tool, automatically fetching current market data through yfinance. The web interface enables comprehensive backtesting across user-defined parameters: bull/bear ETF pairs, investment periods, initial capital allocation, and optional dollar-cost averaging through recurring monthly investments. Performance comparisons against buy-and-hold strategies for both the underlying leveraged ETF and SPY provide context for strategy effectiveness.

Additionally, the system includes a Bayesian optimization module that employs Gaussian Process regression to systematically optimize technical indicator parameters. This machine learning approach uses historical performance data to determine optimal SMA windows and RSI thresholds that maximize risk-adjusted returns (Sharpe ratio). The optimizer implements cross-validation techniques to mitigate overfitting, though the inherent volatility and path-dependent nature of leveraged ETFs presents significant challenges for parameter generalization across different market regimes.

## How to Run It

### Web App (Easy Way)
```bash
# Start the web interface
./start_web.sh

# Then open http://localhost:5000 in your browser
```

### Command Line
```bash
# Basic backtest
python bin/run_backtest.py

# Custom ETF pair and settings
python bin/run_backtest.py --bull QLD --bear QID --start 2020-01-01 --cash 100000

# With AI-optimized parameters (if you've trained them)
python bin/run_backtest.py --use-ai --show-spy --show-buy-hold
```

### Train the AI (Optional)
```bash
# This takes a while and probably won't help much, but it's fun
./train_ai.sh
```

## How the Algorithm Works

Pretty simple concept:
1. When SPY is above its 200-day moving average = bull market, hold the leveraged bull ETF
2. When SPY drops below 200-day average = bear market risk, but we check:
   - Is the bull ETF oversold? (RSI < 31) → Stay in bull ETF for now
   - Is bull ETF below its moving average? → Switch to bear ETF
   - Is bear ETF below its moving average? → Go to cash

Default settings use 200-day SMA for SPY, 20-day for ETF trends, and RSI threshold of 31. The AI can try to optimize these numbers but honestly the defaults work fine.

## ETF Pairs That Work
- **QLD/QID** - 2x leveraged NASDAQ (default in web app)
- **TQQQ/SQQQ** - 3x leveraged NASDAQ 
- **UPRO/SPXS** - 3x leveraged S&P 500
- Any bull/bear leveraged pair really

## Setup
You'll need Python and the requirements installed. The venv folder should have everything already set up.

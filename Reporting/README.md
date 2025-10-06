# Trading Strategy - Daily Email Reports

Automated daily trading reports sent via email using AWS Lambda and SES. Uses the same algorithm from the Backtesting folder to analyze current market conditions and email the results.

## AWS Lambda Setup

1. **Create Lambda function** with Python 3.13 runtime
2. **Add layer**: `AWSSDKPandas-Python313-Arm64`
3. **Upload** `lambda_function.py` (no zip needed)
4. **Set handler** to `lambda_function.lambda_handler`
5. **Environment variables**:
   - `FROM_EMAIL`: Your verified SES email address
   - `TO_EMAIL`: Destination email address
   - `BULL_SYMBOL`: Bull ETF symbol (default: QLD)
   - `BEAR_SYMBOL`: Bear ETF symbol (default: QID)
6. **Schedule** with EventBridge for daily execution

## Local Testing

```bash
./run_algo.sh --fetch-current
```

## What It Does

Analyzes current market data and emails a daily report showing:
- Trading recommendation (Hold QLD, Hold QID, or Hold Cash)
- Current prices and technical indicators
- Reasoning behind the decision
- Warning for potential technical indicator crossovers leading to possible trades

#!/bin/bash
# AI Parameter Optimization Script
# This script runs Bayesian optimization to find optimal trading parameters

echo "Starting AI Parameter Optimization..."
echo "=================================="

# Navigate to project root directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found!"
    echo "Please create one with: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
python3 -c "import skopt" 2>/dev/null || {
    echo "Installing scikit-optimize..."
    pip install scikit-optimize
}

# Default parameters
BULL_SYMBOL=${1:-TQQQ}
BEAR_SYMBOL=${2:-SQQQ}
N_CALLS=${3:-50}
TRAIN_YEARS=${4:-3}

echo "Training Parameters:"
echo "   Bull Symbol: $BULL_SYMBOL"
echo "   Bear Symbol: $BEAR_SYMBOL"
echo "   Optimization Calls: $N_CALLS"
echo "   Training Years: $TRAIN_YEARS"
echo ""

# Run the optimization
echo "Running Bayesian optimization..."
echo "   This may take several minutes..."
echo ""

python3 bin/run_optimizer.py \
    --bull "$BULL_SYMBOL" \
    --bear "$BEAR_SYMBOL" \
    --n-calls "$N_CALLS" \
    --train-years "$TRAIN_YEARS" \
    --random-seed 42

# Check if optimization was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Optimization completed successfully!"
    echo "Results saved to: data/optimized_parameters.json"
    echo ""
    echo "You can now use the optimized parameters by running:"
    echo "   python3 run_backtest.py --use-ai"
    echo "   or"
    echo "   python3 run_web.py (and enable AI optimization in the web interface)"
else
    echo ""
    echo "Optimization failed!"
    echo "Check the error messages above for details."
    exit 1
fi

echo ""
echo "AI training complete!"

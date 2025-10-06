#!/bin/bash
# Web Application Launcher
# This script starts the Flask web interface for the trading strategy

echo "Starting Trading Strategy Web Interface..."
echo "============================================"

# Navigate to project root directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found!"
    echo "Please create one with: python3 -m venv venv"
    echo "Then install dependencies with: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if Flask is installed
python3 -c "import flask" 2>/dev/null || {
    echo "Installing Flask and other dependencies..."
    pip install -r requirements.txt
}

# Set Flask environment variables for better development experience
export FLASK_ENV=development
export FLASK_DEBUG=1

# Check if optimized parameters exist
if [ -f "data/optimized_parameters.json" ]; then
    echo "AI-optimized parameters found!"
    echo "   You can enable them in the web interface."
else
    echo "No AI-optimized parameters found."
    echo "   Run ./train_ai.sh first to generate optimized parameters."
fi

echo ""
echo "Starting web server..."
echo "   - Local URL: http://localhost:5000"
echo "   - Network URL: http://127.0.0.1:5000"
echo ""
echo "Tips:"
echo "   - Try different ETF combinations (TQQQ/SQQQ, QLD/QID, etc.)"
echo "   - Enable AI optimization if parameters are available"
echo "   - Compare with SPY and buy-and-hold strategies"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="

# Start the Flask application
python3 bin/run_web.py

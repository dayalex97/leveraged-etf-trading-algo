#!/usr/bin/env python3
"""
Main entry point for running trading strategy backtests.

This script provides a convenient way to run backtests from the project root.
"""

import sys
import os

# Add the scripts directory to the Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(project_root, 'scripts'))

# Import and run the main algo script
from algo import main

if __name__ == "__main__":
    main()
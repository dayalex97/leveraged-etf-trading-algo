#!/usr/bin/env python3
"""
Entry point for running Bayesian parameter optimization.

This script provides a convenient way to run the optimizer from the project root.
"""

import sys
import os

# Add the src directory to the Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import and run the optimizer
from bayesian_optimizer import main

if __name__ == "__main__":
    main()
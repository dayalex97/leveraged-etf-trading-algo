#!/usr/bin/env python3
"""
Entry point for running the Flask web application.

This script provides a convenient way to start the web server from the project root.
"""

import sys
import os

# Add the web directory to the Python path but don't change working directory
project_root = os.path.dirname(os.path.dirname(__file__))
web_dir = os.path.join(project_root, 'web')
sys.path.insert(0, web_dir)

# Import and run the Flask app
from app import app

if __name__ == "__main__":
    app.run(debug=True, port=5000)
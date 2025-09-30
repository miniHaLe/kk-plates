#!/usr/bin/env python3
"""
KichiKichi Conveyor Belt Dish Counting System
Entry point script for the application
"""

import sys
import os

# Check if running in correct conda environment
def check_environment():
    """Check if running in the kichi conda environment"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    # Allow non-interactive continue when under supervisor
    non_interactive = os.environ.get('KICHI_NONINTERACTIVE', '0') == '1'
    if conda_env != 'kichi':
        print("⚠️  WARNING: You should run 'conda activate kichi' first!")
        print("   Current environment:", conda_env if conda_env else "None")
        print("   Expected environment: kichi")
        print("")
        print("🔧 To set up the environment properly:")
        print("   1. Run: conda activate kichi")
        print("   2. Or use: ./start_kichi.sh")
        print("")
        if non_interactive:
            print("[run.py] Non-interactive mode: continuing despite env mismatch.")
        else:
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("👋 Exiting. Please activate the kichi environment and try again.")
                sys.exit(1)

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main_app import main

if __name__ == "__main__":
    check_environment()
    main()

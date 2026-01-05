#!/usr/bin/env python3
"""
snn-cli.py: SNN Project Integrated CLI Wrapper
This script is a convenient wrapper for the `snn_research.cli.main` module.
It allows users to run CLI commands directly from the project root without
needing to install the package in editable mode via pip if they just want
quick access, though installation is still recommended.
"""
from snn_research.cli.main import main
import sys
import os

# Ensure the current directory is in sys.path so snn_research can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


if __name__ == "__main__":
    main()

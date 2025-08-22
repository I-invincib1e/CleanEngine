#!/usr/bin/env python3
"""
CleanEngine - Main entry point
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.dataset_cleaner.cli import main

if __name__ == "__main__":
    main()

"""
CleanEngine - The Ultimate Data Cleaning & Analysis Toolkit

A fast, modular toolkit for data cleaning, profiling, and analysis.
Supports CSV/Excel/JSON/Parquet, produces reports and visualizations,
and provides a YAML-driven rule engine for validation.
"""

__version__ = "0.1.1"
__author__ = "CleanEngine Community"
__email__ = "contact@cleanengine.com"
__url__ = "https://github.com/I-invincib1e/CleanEngine"
__license__ = "MIT"

from .analysis.analyzer import DataAnalyzer

# Main imports
from .core.cleaner import DatasetCleaner
from .utils.config_manager import ConfigManager
from .utils.file_handler import FileHandler
from .utils.rule_engine import RuleEngine

__all__ = [
    "DatasetCleaner",
    "DataAnalyzer",
    "ConfigManager",
    "FileHandler",
    "RuleEngine",
    "__version__",
    "__author__",
    "__email__",
    "__url__",
    "__license__",
]

"""
Advanced Dataset Cleaner
A comprehensive data analysis and cleaning platform.

This package provides:
- Smart data cleaning with multiple strategies
- Advanced statistical analysis and insights
- Time series analysis capabilities
- Professional visualizations and reporting
- Multiple user interfaces (CLI, GUI, API)
"""

__version__ = "2.0.0"
__author__ = "Advanced Dataset Cleaner Team"
__email__ = "dataset-cleaner@example.com"

# Import main classes for easy access
from .core.cleaner import DatasetCleaner
from .analysis.analyzer import DataAnalyzer
from .analysis.time_series import TimeSeriesAnalyzer
from .analysis.statistical_tests import StatisticalTester
from .utils.config_manager import ConfigManager
from .utils.file_handler import FileHandler

__all__ = [
    'DatasetCleaner',
    'DataAnalyzer', 
    'TimeSeriesAnalyzer',
    'StatisticalTester',
    'ConfigManager',
    'FileHandler'
]
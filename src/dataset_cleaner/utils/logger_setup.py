#!/usr/bin/env python3
"""
Logging Setup for Advanced Dataset Cleaner
Configures comprehensive logging for debugging and monitoring.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

from .config_manager import config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_size_mb: Optional[int] = None,
    backup_count: Optional[int] = None,
) -> logging.Logger:
    """
    Set up comprehensive logging for the dataset cleaner

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name
        max_size_mb: Maximum log file size in MB
        backup_count: Number of backup log files to keep

    Returns:
        Configured logger instance
    """

    # Get configuration values
    log_level = log_level or config.get("logging.level", "INFO")
    log_file = log_file or config.get("logging.file", "dataset_cleaner.log")
    max_size_mb = max_size_mb or config.get("logging.max_size_mb", 10)
    backup_count = backup_count or config.get("logging.backup_count", 3)

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / log_file

    # Configure logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("dataset_cleaner")
    logger.setLevel(numeric_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_size_mb * 1024 * 1024,  # Convert MB to bytes
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # Add a debug file handler for detailed debugging
    if numeric_level <= logging.DEBUG:
        debug_handler = logging.handlers.RotatingFileHandler(
            log_dir / "debug.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=2,
            encoding="utf-8",
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_formatter)
        logger.addHandler(debug_handler)

    # Log initial setup message
    logger.info("=" * 50)
    logger.info("Dataset Cleaner Logging Initialized")
    logger.info(f"Log Level: {log_level}")
    logger.info(f"Log File: {log_path}")
    logger.info(f"Max Size: {max_size_mb}MB")
    logger.info(f"Backup Count: {backup_count}")
    logger.info("=" * 50)

    return logger


def log_system_info(logger: logging.Logger) -> None:
    """Log system information for debugging"""
    import platform
    import sys

    import psutil

    logger.info("System Information:")
    logger.info(f"  Python Version: {sys.version}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Architecture: {platform.architecture()}")
    logger.info(f"  CPU Count: {psutil.cpu_count()}")
    logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(
        f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB"
    )


def log_dataset_info(logger: logging.Logger, df, dataset_name: str) -> None:
    """Log dataset information"""
    logger.info(f"Dataset Information - {dataset_name}:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(
        f"  Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB"
    )
    logger.info(f"  Data Types: {df.dtypes.value_counts().to_dict()}")
    logger.info(f"  Missing Values: {df.isnull().sum().sum()}")
    logger.info(f"  Duplicates: {df.duplicated().sum()}")


def log_performance(
    logger: logging.Logger, operation: str, duration: float, rows_processed: int = 0
) -> None:
    """Log performance metrics"""
    logger.info(f"Performance - {operation}:")
    logger.info(f"  Duration: {duration:.2f} seconds")
    if rows_processed > 0:
        logger.info(f"  Rows Processed: {rows_processed:,}")
        logger.info(f"  Rows/Second: {rows_processed/duration:,.0f}")


def log_analysis_results(logger: logging.Logger, analysis_results: dict) -> None:
    """Log analysis results summary"""
    logger.info("Analysis Results Summary:")

    if "data_quality" in analysis_results:
        quality = analysis_results["data_quality"]
        logger.info(
            f"  Data Quality Score: {quality.get('overall_quality_score', 'N/A')}"
        )
        logger.info(
            f"  Completeness: {quality.get('completeness_percentage', 'N/A'):.1f}%"
        )

    if "correlation_analysis" in analysis_results:
        corr = analysis_results["correlation_analysis"]
        strong_corr_count = len(corr.get("strong_correlations", []))
        logger.info(f"  Strong Correlations Found: {strong_corr_count}")

    if "clustering_analysis" in analysis_results:
        cluster = analysis_results["clustering_analysis"]
        logger.info(f"  Optimal Clusters: {cluster.get('optimal_clusters', 'N/A')}")

    if "anomaly_detection" in analysis_results:
        anomaly = analysis_results["anomaly_detection"]
        logger.info(f"  Anomalies Detected: {anomaly.get('total_anomalies', 'N/A')}")

    insights_count = len(analysis_results.get("insights", []))
    logger.info(f"  Insights Generated: {insights_count}")


class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, logger: logging.Logger, operation: str, rows: int = 0):
        self.logger = logger
        self.operation = operation
        self.rows = rows
        self.start_time = None

    def __enter__(self):
        import time

        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        duration = time.time() - self.start_time

        if exc_type is None:
            log_performance(self.logger, self.operation, duration, self.rows)
        else:
            self.logger.error(f"Error in {self.operation}: {exc_val}")
            self.logger.error(f"Duration before error: {duration:.2f} seconds")


# Initialize logger
logger = setup_logging()

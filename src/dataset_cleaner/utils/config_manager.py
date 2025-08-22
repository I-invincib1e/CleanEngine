#!/usr/bin/env python3
"""
Configuration Manager for Advanced Dataset Cleaner
Handles loading and managing configuration settings from YAML files.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigManager:
    """Manages configuration settings for the dataset cleaner"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.yaml"
        self.config = self._load_default_config()

        # Load custom config if it exists
        if os.path.exists(self.config_file):
            self.load_config(self.config_file)

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings"""
        return {
            "cleaning": {
                "missing_values": {
                    "threshold": 0.5,
                    "strategy": "auto",
                    "fill_numeric": "median",
                    "fill_categorical": "mode",
                },
                "duplicates": {"remove": True, "keep": "first"},
                "outliers": {"method": "iqr", "threshold": 1.5, "action": "remove"},
                "encoding": {"categorical_method": "label", "handle_unknown": "ignore"},
                "normalization": {"method": "minmax", "feature_range": [0, 1]},
            },
            "analysis": {
                "enable": True,
                "statistical": {"enable": True},
                "correlation": {"enable": True, "threshold": 0.7},
                "clustering": {"enable": True, "max_clusters": 10},
                "anomaly_detection": {"enable": True, "contamination": 0.1},
                "feature_importance": {"enable": True},
                "quality_assessment": {"enable": True},
            },
            "visualization": {
                "enable": True,
                "style": "seaborn-v0_8",
                "color_palette": "viridis",
                "figure_size": [12, 8],
                "dpi": 300,
                "format": "png",
            },
            "output": {
                "create_folders": True,
                "folder_prefix": "Cleans-",
                "save_original": False,
                "compress_output": False,
                "reports": {
                    "generate_markdown": True,
                    "generate_json": True,
                    "generate_html": False,
                    "include_visualizations": True,
                },
            },
            "logging": {
                "level": "INFO",
                "file": "dataset_cleaner.log",
                "max_size_mb": 10,
                "backup_count": 3,
            },
        }

    def load_config(self, config_file: str) -> None:
        """Load configuration from YAML file"""
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)

            # Merge user config with default config
            self.config = self._merge_configs(self.config, user_config)
            logging.info(f"Configuration loaded from {config_file}")

        except FileNotFoundError:
            logging.warning(
                f"Configuration file {config_file} not found. Using defaults."
            )
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
        except Exception as e:
            logging.error(f"Unexpected error loading configuration: {e}")

    def _merge_configs(
        self, default: Dict[str, Any], user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge user configuration with default configuration"""
        merged = default.copy()

        for key, value in user.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path to configuration value (e.g., 'cleaning.missing_values.threshold')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation

        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split(".")
        config = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value

    def save_config(self, output_file: Optional[str] = None) -> None:
        """Save current configuration to YAML file"""
        output_file = output_file or self.config_file

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logging.info(f"Configuration saved to {output_file}")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")

    def get_cleaning_config(self) -> Dict[str, Any]:
        """Get cleaning-specific configuration"""
        return self.config.get("cleaning", {})

    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis-specific configuration"""
        return self.config.get("analysis", {})

    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization-specific configuration"""
        return self.config.get("visualization", {})

    def get_output_config(self) -> Dict[str, Any]:
        """Get output-specific configuration"""
        return self.config.get("output", {})

    def is_analysis_enabled(self) -> bool:
        """Check if advanced analysis is enabled"""
        return self.get("analysis.enable", True)

    def is_visualization_enabled(self) -> bool:
        """Check if visualization is enabled"""
        return self.get("visualization.enable", True)

    def get_log_level(self) -> str:
        """Get logging level"""
        return self.get("logging.level", "INFO")

    def get_output_folder_prefix(self) -> str:
        """Get output folder prefix"""
        return self.get("output.folder_prefix", "Cleans-")

    def validate_config(self) -> bool:
        """Validate configuration settings"""
        try:
            # Validate cleaning settings
            cleaning = self.get_cleaning_config()
            if cleaning.get("missing_values", {}).get("threshold", 0.5) > 1.0:
                logging.warning("Missing values threshold should be <= 1.0")
                return False

            # Validate analysis settings
            analysis = self.get_analysis_config()
            if analysis.get("correlation", {}).get("threshold", 0.7) > 1.0:
                logging.warning("Correlation threshold should be <= 1.0")
                return False

            # Validate clustering settings
            max_clusters = analysis.get("clustering", {}).get("max_clusters", 10)
            if max_clusters < 2:
                logging.warning("Maximum clusters should be >= 2")
                return False

            return True

        except Exception as e:
            logging.error(f"Configuration validation error: {e}")
            return False

    def create_sample_config(self, output_file: str = "config_sample.yaml") -> None:
        """Create a sample configuration file with comments"""
        sample_config = """# Advanced Dataset Cleaner Configuration
# Customize these settings to match your data processing needs

# Data Cleaning Settings
cleaning:
  missing_values:
    threshold: 0.5              # Drop columns with >50% missing values
    strategy: "auto"            # auto, mean, median, mode, drop
    fill_numeric: "median"      # mean, median, mode
    fill_categorical: "mode"    # mode, constant
    
  duplicates:
    remove: true               # Remove duplicate rows
    keep: "first"              # first, last, false
    
  outliers:
    method: "iqr"              # iqr, zscore, isolation_forest
    threshold: 1.5             # IQR multiplier or Z-score threshold
    action: "remove"           # remove, cap, flag
    
  encoding:
    categorical_method: "label"  # label, onehot, target
    handle_unknown: "ignore"     # ignore, error
    
  normalization:
    method: "minmax"           # minmax, standard, robust
    feature_range: [0, 1]      # For MinMax scaling

# Advanced Analysis Settings  
analysis:
  enable: true                 # Enable advanced analysis
  
  statistical:
    enable: true
    include_skewness: true
    include_kurtosis: true
    normality_test: true
    
  correlation:
    enable: true
    method: "pearson"          # pearson, spearman, kendall
    threshold: 0.7             # Strong correlation threshold
    
  clustering:
    enable: true
    method: "kmeans"           # kmeans, hierarchical, dbscan
    max_clusters: 10           # Maximum clusters to consider
    
  anomaly_detection:
    enable: true
    method: "isolation_forest" # isolation_forest, local_outlier_factor
    contamination: 0.1         # Expected proportion of outliers

# Output Settings
output:
  create_folders: true        # Create organized output folders
  folder_prefix: "Cleans-"    # Prefix for output folders
  
  reports:
    generate_markdown: true   # Generate markdown reports
    generate_json: true       # Generate JSON reports
    include_visualizations: true

# Logging Settings
logging:
  level: "INFO"              # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: "dataset_cleaner.log" # Log file name
"""

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(sample_config)
            print(f"Sample configuration created: {output_file}")
        except Exception as e:
            logging.error(f"Error creating sample configuration: {e}")


# Global configuration instance
config = ConfigManager()

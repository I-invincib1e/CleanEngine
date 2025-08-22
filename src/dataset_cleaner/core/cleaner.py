#!/usr/bin/env python3
"""
Core Dataset Cleaner Module
Main cleaning pipeline and orchestration logic.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from scipy import stats
import json
from datetime import datetime

from ..analysis.analyzer import DataAnalyzer
from ..utils.config_manager import ConfigManager
from ..utils.logger_setup import logger, PerformanceTimer, log_dataset_info
from ..utils.file_handler import FileHandler


class DatasetCleaner:
    """Main dataset cleaning class with comprehensive cleaning pipeline"""
    
    def __init__(self, config_manager=None):
        self.config = config_manager or ConfigManager()
        self.logger = logger
        self.file_handler = FileHandler()
        self.report = {}
        self.scaler = None
        self.label_encoders = {}
        self.original_df = None
        self.cleaned_df = None
    
    def load_data(self, file_path):
        """Load data from various file formats using enhanced file handler"""
        with PerformanceTimer(self.logger, "Data Loading"):
            df = self.file_handler.load_data(file_path)
            log_dataset_info(self.logger, df, Path(file_path).stem)
            return df
    
    def analyze_data(self, df):
        """Analyze dataset and store initial statistics"""
        self.report['original_shape'] = df.shape
        self.report['original_columns'] = list(df.columns)
        self.report['data_types'] = df.dtypes.to_dict()
        self.report['missing_values_before'] = df.isnull().sum().to_dict()
        self.report['duplicates_before'] = df.duplicated().sum()
        
        # Memory usage
        self.report['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
        
        return self.report  
  
    def handle_missing_values(self, df, strategy='auto', threshold=None):
        """Handle missing values with various strategies"""
        threshold = threshold or self.config.get('cleaning.missing_values.threshold', 0.5)
        missing_before = df.isnull().sum()
        
        for column in df.columns:
            missing_ratio = df[column].isnull().sum() / len(df)
            
            if missing_ratio > threshold:
                # Drop columns with too many missing values
                df = df.drop(column, axis=1)
                self.report[f'dropped_column_{column}'] = f'Too many missing values ({missing_ratio:.2%})'
            elif missing_ratio > 0:
                if df[column].dtype in ['int64', 'float64']:
                    # Numeric columns: fill with median
                    fill_method = self.config.get('cleaning.missing_values.fill_numeric', 'median')
                    if fill_method == 'mean':
                        df[column] = df[column].fillna(df[column].mean())
                    else:  # median
                        df[column] = df[column].fillna(df[column].median())
                else:
                    # Categorical columns: fill with mode
                    mode_value = df[column].mode()
                    if not mode_value.empty:
                        df[column] = df[column].fillna(mode_value[0])
                    else:
                        df[column] = df[column].fillna('Unknown')
        
        self.report['missing_values_after'] = df.isnull().sum().to_dict()
        self.report['columns_after_missing_cleanup'] = list(df.columns)
        
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        duplicates_before = df.duplicated().sum()
        keep_method = self.config.get('cleaning.duplicates.keep', 'first')
        df_cleaned = df.drop_duplicates(keep=keep_method)
        duplicates_after = df_cleaned.duplicated().sum()
        
        self.report['duplicates_removed'] = duplicates_before - duplicates_after
        self.report['duplicates_after'] = duplicates_after
        
        return df_cleaned
    
    def detect_and_remove_outliers(self, df, method=None, z_threshold=3):
        """Detect and remove outliers using IQR or Z-score method"""
        method = method or self.config.get('cleaning.outliers.method', 'iqr')
        threshold = self.config.get('cleaning.outliers.threshold', 1.5)
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers_removed = {}
        
        for column in numeric_columns:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            else:  # z-score method
                non_null_mask = df[column].notna()
                z_scores = np.abs(stats.zscore(df[column].dropna()))
                outliers_mask = pd.Series(False, index=df.index)
                outliers_mask[non_null_mask] = z_scores > z_threshold
            
            outliers_count = outliers_mask.sum()
            outliers_removed[column] = outliers_count
            
            # Remove outliers
            df = df[~outliers_mask]
        
        self.report['outliers_removed'] = outliers_removed
        self.report['shape_after_outliers'] = df.shape
        
        return df    

    def encode_categorical_variables(self, df, encoding_method=None):
        """Encode categorical variables"""
        encoding_method = encoding_method or self.config.get('cleaning.encoding.categorical_method', 'label')
        categorical_columns = df.select_dtypes(include=['object']).columns
        encoded_columns = {}
        
        for column in categorical_columns:
            if encoding_method == 'label':
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].astype(str))
                self.label_encoders[column] = le
                encoded_columns[column] = 'label_encoded'
            elif encoding_method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[column], prefix=column)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(column, axis=1)
                encoded_columns[column] = f'one_hot_encoded_{len(dummies.columns)}_features'
        
        self.report['categorical_encoding'] = encoded_columns
        return df
    
    def normalize_data(self, df, method=None):
        """Normalize numeric columns"""
        method = method or self.config.get('cleaning.normalization.method', 'minmax')
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            feature_range = self.config.get('cleaning.normalization.feature_range', [0, 1])
            if not isinstance(feature_range, tuple):
                feature_range = tuple(feature_range)
            self.scaler = MinMaxScaler(feature_range=feature_range)
        else:  # standardization
            self.scaler = StandardScaler()
        
        if len(numeric_columns) > 0:
            df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
            self.report['normalization_method'] = method
            self.report['normalized_columns'] = list(numeric_columns)
        
        return df
    
    def clean_dataset(self, file_path, missing_threshold=None, outlier_method=None, 
                     encoding_method=None, normalization_method=None, 
                     perform_analysis=None):
        """Main cleaning pipeline with optional advanced analysis"""
        perform_analysis = perform_analysis if perform_analysis is not None else self.config.get('analysis.enable', True)
        
        self.logger.info(f"Starting dataset cleaning: {file_path}")
        print(f"🔄 Loading dataset: {file_path}")
        
        original_df = self.load_data(file_path)
        df = original_df.copy()
        
        print("📊 Analyzing initial data...")
        self.analyze_data(df)
        
        print("🧹 Handling missing values...")
        df = self.handle_missing_values(df, threshold=missing_threshold)
        
        print("🔍 Removing duplicates...")
        df = self.remove_duplicates(df)
        
        print("📈 Detecting and removing outliers...")
        df = self.detect_and_remove_outliers(df, method=outlier_method)
        
        print("🏷️ Encoding categorical variables...")
        df = self.encode_categorical_variables(df, encoding_method=encoding_method)
        
        print("⚖️ Normalizing data...")
        df = self.normalize_data(df, method=normalization_method)
        
        # Final statistics
        self.report['final_shape'] = df.shape
        self.report['cleaning_timestamp'] = datetime.now().isoformat()
        
        # Store original and cleaned dataframes for analysis
        self.original_df = original_df
        self.cleaned_df = df
        
        self.logger.info("Dataset cleaning completed successfully")
        print("✅ Dataset cleaning completed!")
        return df
    
    def create_output_folder(self, input_file_path):
        """Create organized output folder structure"""
        input_path = Path(input_file_path)
        dataset_name = input_path.stem
        
        # Get folder prefix from config
        folder_prefix = self.config.get('output.folder_prefix', 'Cleans-')
        
        # Create main output folder
        output_folder = Path(f"{folder_prefix}{dataset_name}")
        output_folder.mkdir(exist_ok=True)
        
        self.logger.info(f"Created output folder: {output_folder}")
        return output_folder
    
    def generate_report(self, output_folder, dataset_name):
        """Generate detailed cleaning report in organized folder"""
        # Create report files in the output folder
        report_file = output_folder / f"{dataset_name}_cleaning_report.json"
        summary_file = output_folder / f"{dataset_name}_cleaning_summary.txt"
        
        # Generate JSON report
        with open(report_file, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
        
        # Generate human-readable summary
        with open(summary_file, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("DATASET CLEANING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {self.report.get('cleaning_timestamp', 'N/A')}\n\n")
            
            # Shape changes
            original_shape = self.report.get('original_shape', (0, 0))
            final_shape = self.report.get('final_shape', (0, 0))
            f.write(f"Dataset Shape:\n")
            f.write(f"  Original: {original_shape[0]:,} rows × {original_shape[1]} columns\n")
            f.write(f"  Final: {final_shape[0]:,} rows × {final_shape[1]} columns\n")
            f.write(f"  Rows removed: {original_shape[0] - final_shape[0]:,}\n\n")
            
            # Missing values
            missing_before = sum(self.report.get('missing_values_before', {}).values())
            missing_after = sum(self.report.get('missing_values_after', {}).values())
            f.write(f"Missing Values:\n")
            f.write(f"  Before: {missing_before:,}\n")
            f.write(f"  After: {missing_after:,}\n")
            f.write(f"  Cleaned: {missing_before - missing_after:,}\n\n")
            
            # Duplicates
            f.write(f"Duplicates Removed: {self.report.get('duplicates_removed', 0):,}\n\n")
            
            # Outliers
            outliers = self.report.get('outliers_removed', {})
            total_outliers = sum(outliers.values())
            f.write(f"Outliers Removed: {total_outliers:,}\n")
            for col, count in outliers.items():
                if count > 0:
                    f.write(f"  {col}: {count:,}\n")
            f.write("\n")
            
            # Encoding
            encoding = self.report.get('categorical_encoding', {})
            if encoding:
                f.write("Categorical Encoding:\n")
                for col, method in encoding.items():
                    f.write(f"  {col}: {method}\n")
                f.write("\n")
            
            # Normalization
            norm_method = self.report.get('normalization_method', 'None')
            norm_cols = self.report.get('normalized_columns', [])
            f.write(f"Normalization: {norm_method}\n")
            f.write(f"Normalized columns: {len(norm_cols)}\n\n")
        
        self.logger.info(f"Reports generated: {report_file} and {summary_file}")
        print(f"📄 Reports generated: {report_file} and {summary_file}")
        return report_file, summary_file
    
    def perform_advanced_analysis(self, output_folder, dataset_name):
        """Perform advanced data analysis on cleaned dataset"""
        if not hasattr(self, 'cleaned_df') or not hasattr(self, 'original_df'):
            self.logger.warning("No cleaned dataset available for analysis")
            print("⚠️ No cleaned dataset available for analysis")
            return None
        
        if not self.config.get('analysis.enable', True):
            self.logger.info("Advanced analysis disabled in configuration")
            print("ℹ️ Advanced analysis disabled in configuration")
            return None
        
        print("🚀 Starting advanced data analysis...")
        self.logger.info("Starting advanced data analysis")
        
        # Initialize analyzer with both original and cleaned data
        analyzer = DataAnalyzer(self.cleaned_df, self.original_df, self.config)
        
        # Perform comprehensive analysis
        analysis_results = analyzer.generate_comprehensive_analysis()
        
        # Create visualizations
        viz_folder = analyzer.create_analysis_visualizations(output_folder)
        
        # Generate analysis report
        analysis_report = analyzer.generate_analysis_report(output_folder, dataset_name)
        
        # Save analysis results as JSON
        analysis_json = output_folder / f"{dataset_name}_analysis_results.json"
        with open(analysis_json, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                return str(obj)  # Convert everything else to string to avoid circular references
            
            try:
                json.dump(analysis_results, f, indent=2, default=convert_numpy)
            except Exception as e:
                # If JSON serialization fails, save a simplified version
                simplified_results = {
                    'insights': analysis_results.get('insights', []),
                    'data_quality': analysis_results.get('data_quality', {}),
                    'analysis_timestamp': pd.Timestamp.now().isoformat()
                }
                json.dump(simplified_results, f, indent=2, default=convert_numpy)
        
        self.logger.info("Advanced analysis completed successfully")
        print(f"🎉 Advanced analysis completed!")
        print(f"📊 Analysis report: {analysis_report}")
        print(f"📈 Visualizations: {viz_folder}")
        print(f"📋 Analysis data: {analysis_json}")
        
        return {
            'analysis_results': analysis_results,
            'report_file': analysis_report,
            'visualizations_folder': viz_folder,
            'analysis_json': analysis_json
        }
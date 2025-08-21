#!/usr/bin/env python3
"""
Automated Dataset Cleaner
A tool to automatically clean CSV/Excel datasets with comprehensive reporting.
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from scipy import stats
import json
from datetime import datetime
from data_analyzer import DataAnalyzer
from config_manager import config
from logger_setup import logger, PerformanceTimer, log_dataset_info, log_analysis_results
from file_handler import file_handler

class DatasetCleaner:
    def __init__(self):
        self.report = {}
        self.scaler = None
        self.label_encoders = {}
    
    def load_data(self, file_path):
        """Load data from various file formats using enhanced file handler"""
        with PerformanceTimer(logger, "Data Loading"):
            df = file_handler.load_data(file_path)
            log_dataset_info(logger, df, Path(file_path).stem)
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
  
    def handle_missing_values(self, df, strategy='auto', threshold=0.5):
        """Handle missing values with various strategies"""
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
        df_cleaned = df.drop_duplicates()
        duplicates_after = df_cleaned.duplicated().sum()
        
        self.report['duplicates_removed'] = duplicates_before - duplicates_after
        self.report['duplicates_after'] = duplicates_after
        
        return df_cleaned
    
    def detect_and_remove_outliers(self, df, method='iqr', z_threshold=3):
        """Detect and remove outliers using IQR or Z-score method"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers_removed = {}
        
        for column in numeric_columns:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
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

    def encode_categorical_variables(self, df, encoding_method='label'):
        """Encode categorical variables"""
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
    
    def normalize_data(self, df, method='minmax'):
        """Normalize numeric columns"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        else:  # standardization
            self.scaler = StandardScaler()
        
        if len(numeric_columns) > 0:
            df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
            self.report['normalization_method'] = method
            self.report['normalized_columns'] = list(numeric_columns)
        
        return df
    
    def clean_dataset(self, file_path, missing_threshold=0.5, outlier_method='iqr', 
                     encoding_method='label', normalization_method='minmax', 
                     perform_analysis=True):
        """Main cleaning pipeline with optional advanced analysis"""
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
        
        print("✅ Dataset cleaning completed!")
        return df
    
    def create_output_folder(self, input_file_path):
        """Create organized output folder structure"""
        input_path = Path(input_file_path)
        dataset_name = input_path.stem
        
        # Get folder prefix from config
        folder_prefix = config.get_output_folder_prefix()
        
        # Create main output folder
        output_folder = Path(f"{folder_prefix}{dataset_name}")
        output_folder.mkdir(exist_ok=True)
        
        logger.info(f"Created output folder: {output_folder}")
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
        
        print(f"📄 Reports generated: {report_file} and {summary_file}")
        return report_file, summary_file
    
    def perform_advanced_analysis(self, output_folder, dataset_name):
        """Perform advanced data analysis on cleaned dataset"""
        if not hasattr(self, 'cleaned_df') or not hasattr(self, 'original_df'):
            print("⚠️ No cleaned dataset available for analysis")
            return None
        
        print("🚀 Starting advanced data analysis...")
        
        # Initialize analyzer with both original and cleaned data
        analyzer = DataAnalyzer(self.cleaned_df, self.original_df)
        
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
            
            import json
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


def main():
    parser = argparse.ArgumentParser(description='Automated Dataset Cleaner')
    parser.add_argument('input_file', help='Path to input CSV or Excel file')
    parser.add_argument('-o', '--output', help='Output file path (default: cleaned_<input_name>)')
    parser.add_argument('--missing-threshold', type=float, default=0.5, 
                       help='Threshold for dropping columns with missing values (default: 0.5)')
    parser.add_argument('--outlier-method', choices=['iqr', 'zscore'], default='iqr',
                       help='Method for outlier detection (default: iqr)')
    parser.add_argument('--encoding', choices=['label', 'onehot'], default='label',
                       help='Categorical encoding method (default: label)')
    parser.add_argument('--normalization', choices=['minmax', 'standard'], default='minmax',
                       help='Normalization method (default: minmax)')
    parser.add_argument('--analysis', action='store_true', default=True,
                       help='Perform advanced data analysis (default: True)')
    parser.add_argument('--no-analysis', action='store_true',
                       help='Skip advanced data analysis')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output:
        input_path = Path(args.input_file)
        args.output = f"cleaned_{input_path.stem}{input_path.suffix}"
    
    try:
        # Initialize cleaner
        cleaner = DatasetCleaner()
        
        # Create organized output folder
        input_path = Path(args.input_file)
        dataset_name = input_path.stem
        output_folder = cleaner.create_output_folder(args.input_file)
        
        print(f"📁 Created output folder: {output_folder}")
        
        # Clean the dataset
        cleaned_df = cleaner.clean_dataset(
            args.input_file,
            missing_threshold=args.missing_threshold,
            outlier_method=args.outlier_method,
            encoding_method=args.encoding,
            normalization_method=args.normalization
        )
        
        # Determine output file path in the organized folder
        if args.output:
            # If user specified output, use it but place in organized folder
            output_filename = Path(args.output).name
        else:
            # Default naming
            output_filename = f"cleaned_{input_path.name}"
        
        output_file_path = output_folder / output_filename
        
        # Save cleaned dataset
        if output_file_path.suffix == '.csv':
            cleaned_df.to_csv(output_file_path, index=False)
        else:
            cleaned_df.to_excel(output_file_path, index=False)
        
        print(f"💾 Cleaned dataset saved: {output_file_path}")
        
        # Generate reports in organized folder
        cleaner.generate_report(output_folder, dataset_name)
        
        # Perform advanced analysis if requested
        perform_analysis = args.analysis and not args.no_analysis
        if perform_analysis:
            try:
                analysis_results = cleaner.perform_advanced_analysis(output_folder, dataset_name)
                if analysis_results:
                    print("🔬 Advanced analysis completed with insights and visualizations!")
            except Exception as e:
                print(f"⚠️ Advanced analysis failed: {str(e)}")
                print("📊 Basic cleaning reports are still available")
        
        print(f"\n🎉 Dataset cleaning completed successfully!")
        print(f"📂 All files saved in: {output_folder.absolute()}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
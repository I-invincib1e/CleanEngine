#!/usr/bin/env python3
"""
Enhanced File Handler for Advanced Dataset Cleaner
Supports multiple file formats: CSV, Excel, JSON, Parquet
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging
from config_manager import config

logger = logging.getLogger('dataset_cleaner')

class FileHandler:
    """Enhanced file handler supporting multiple formats"""
    
    def __init__(self):
        self.supported_formats = {
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.json': self._load_json,
            '.parquet': self._load_parquet
        }
        
        self.save_formats = {
            '.csv': self._save_csv,
            '.xlsx': self._save_excel,
            '.json': self._save_json,
            '.parquet': self._save_parquet
        }
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            file_path: Path to the data file
            
        Returns:
            pandas DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}. "
                           f"Supported formats: {list(self.supported_formats.keys())}")
        
        logger.info(f"Loading {file_extension} file: {file_path}")
        
        try:
            df = self.supported_formats[file_extension](file_path)
            logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def save_data(self, df: pd.DataFrame, file_path: Union[str, Path], 
                  **kwargs) -> None:
        """
        Save DataFrame to various file formats
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            **kwargs: Additional arguments for specific formats
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.save_formats:
            raise ValueError(f"Unsupported save format: {file_extension}. "
                           f"Supported formats: {list(self.save_formats.keys())}")
        
        logger.info(f"Saving data to {file_extension} file: {file_path}")
        
        try:
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.save_formats[file_extension](df, file_path, **kwargs)
            logger.info(f"Successfully saved data: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {str(e)}")
            raise
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file with encoding detection"""
        csv_config = config.get('file_formats.csv', {})
        encoding = csv_config.get('encoding', 'auto')
        delimiter = csv_config.get('delimiter', ',')
        
        if encoding == 'auto':
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    return pd.read_csv(file_path, encoding=enc, delimiter=delimiter)
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode CSV file with any encoding: {encodings}")
        else:
            return pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
    
    def _load_excel(self, file_path: Path) -> pd.DataFrame:
        """Load Excel file"""
        excel_config = config.get('file_formats.excel', {})
        engine = excel_config.get('engine', 'openpyxl')
        
        return pd.read_excel(file_path, engine=engine)
    
    def _load_json(self, file_path: Path) -> pd.DataFrame:
        """Load JSON file"""
        json_config = config.get('file_formats.json', {})
        orient = json_config.get('orient', 'records')
        
        try:
            # Try pandas JSON reader first
            return pd.read_json(file_path, orient=orient)
        except ValueError:
            # Fallback to manual JSON loading
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to convert dict to DataFrame
                if all(isinstance(v, list) for v in data.values()):
                    return pd.DataFrame(data)
                else:
                    # Single record
                    return pd.DataFrame([data])
            else:
                raise ValueError("JSON format not suitable for DataFrame conversion")
    
    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load Parquet file"""
        parquet_config = config.get('file_formats.parquet', {})
        engine = parquet_config.get('engine', 'pyarrow')
        
        return pd.read_parquet(file_path, engine=engine)
    
    def _save_csv(self, df: pd.DataFrame, file_path: Path, **kwargs) -> None:
        """Save to CSV format"""
        csv_config = config.get('file_formats.csv', {})
        encoding = kwargs.get('encoding', 'utf-8')
        delimiter = kwargs.get('delimiter', csv_config.get('delimiter', ','))
        
        df.to_csv(file_path, index=False, encoding=encoding, sep=delimiter)
    
    def _save_excel(self, df: pd.DataFrame, file_path: Path, **kwargs) -> None:
        """Save to Excel format"""
        excel_config = config.get('file_formats.excel', {})
        engine = kwargs.get('engine', excel_config.get('engine', 'openpyxl'))
        
        df.to_excel(file_path, index=False, engine=engine)
    
    def _save_json(self, df: pd.DataFrame, file_path: Path, **kwargs) -> None:
        """Save to JSON format"""
        json_config = config.get('file_formats.json', {})
        orient = kwargs.get('orient', json_config.get('orient', 'records'))
        
        # Handle datetime and other non-serializable types
        df_copy = df.copy()
        
        # Convert datetime columns to strings
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert numpy types to native Python types
        df_copy = df_copy.replace({np.nan: None})
        
        df_copy.to_json(file_path, orient=orient, indent=2)
    
    def _save_parquet(self, df: pd.DataFrame, file_path: Path, **kwargs) -> None:
        """Save to Parquet format"""
        parquet_config = config.get('file_formats.parquet', {})
        engine = kwargs.get('engine', parquet_config.get('engine', 'pyarrow'))
        
        df.to_parquet(file_path, index=False, engine=engine)
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        info = {
            'file_name': file_path.name,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'file_extension': file_path.suffix.lower(),
            'is_supported': file_path.suffix.lower() in self.supported_formats
        }
        
        # Try to get data info if supported format
        if info['is_supported']:
            try:
                df = self.load_data(file_path)
                info.update({
                    'rows': df.shape[0],
                    'columns': df.shape[1],
                    'column_names': list(df.columns),
                    'data_types': df.dtypes.to_dict(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                    'missing_values': df.isnull().sum().to_dict(),
                    'duplicates': df.duplicated().sum()
                })
            except Exception as e:
                info['load_error'] = str(e)
        
        return info
    
    def convert_format(self, input_file: Union[str, Path], 
                      output_file: Union[str, Path], **kwargs) -> None:
        """
        Convert file from one format to another
        
        Args:
            input_file: Source file path
            output_file: Destination file path
            **kwargs: Additional arguments for saving
        """
        logger.info(f"Converting {input_file} to {output_file}")
        
        # Load data
        df = self.load_data(input_file)
        
        # Save in new format
        self.save_data(df, output_file, **kwargs)
        
        logger.info(f"Successfully converted file format")
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate file and return validation results
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        try:
            # Get file info
            file_info = self.get_file_info(file_path)
            validation_results['file_info'] = file_info
            
            # Check if file is supported
            if not file_info['is_supported']:
                validation_results['errors'].append(
                    f"Unsupported file format: {file_info['file_extension']}"
                )
                return validation_results
            
            # Check for load errors
            if 'load_error' in file_info:
                validation_results['errors'].append(f"Load error: {file_info['load_error']}")
                return validation_results
            
            # Check file size
            if file_info['file_size_mb'] > 100:  # 100MB threshold
                validation_results['warnings'].append(
                    f"Large file size: {file_info['file_size_mb']:.1f}MB"
                )
            
            # Check data quality
            if file_info['rows'] == 0:
                validation_results['errors'].append("File contains no data rows")
                return validation_results
            
            if file_info['columns'] == 0:
                validation_results['errors'].append("File contains no columns")
                return validation_results
            
            # Check for high missing values
            total_cells = file_info['rows'] * file_info['columns']
            total_missing = sum(file_info['missing_values'].values())
            missing_percentage = (total_missing / total_cells) * 100
            
            if missing_percentage > 50:
                validation_results['warnings'].append(
                    f"High missing values: {missing_percentage:.1f}%"
                )
            
            # Check for high duplicates
            duplicate_percentage = (file_info['duplicates'] / file_info['rows']) * 100
            if duplicate_percentage > 20:
                validation_results['warnings'].append(
                    f"High duplicate rows: {duplicate_percentage:.1f}%"
                )
            
            validation_results['is_valid'] = True
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results

# Global file handler instance
file_handler = FileHandler()
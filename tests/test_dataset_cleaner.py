#!/usr/bin/env python3
"""
Unit tests for Dataset Cleaner
Tests core cleaning functionality and edge cases.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dataset_cleaner import DatasetCleaner
from config_manager import ConfigManager

class TestDatasetCleaner(unittest.TestCase):
    """Test cases for DatasetCleaner class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cleaner = DatasetCleaner()
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],  # Contains outlier
            'categorical_col': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'missing_col': [1, 2, np.nan, 4, np.nan, 6, 7, np.nan, 9, 10],
            'duplicate_col': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        })
        
        # Create data with high missing values
        self.high_missing_data = pd.DataFrame({
            'good_col': [1, 2, 3, 4, 5],
            'bad_col': [1, np.nan, np.nan, np.nan, np.nan]  # 80% missing
        })
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_data_csv(self):
        """Test loading CSV data"""
        # Create temporary CSV file
        csv_file = os.path.join(self.temp_dir, 'test.csv')
        self.sample_data.to_csv(csv_file, index=False)
        
        # Test loading
        loaded_data = self.cleaner.load_data(csv_file)
        
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(loaded_data.shape, self.sample_data.shape)
        self.assertEqual(list(loaded_data.columns), list(self.sample_data.columns))
    
    def test_load_data_excel(self):
        """Test loading Excel data"""
        # Create temporary Excel file
        excel_file = os.path.join(self.temp_dir, 'test.xlsx')
        self.sample_data.to_excel(excel_file, index=False)
        
        # Test loading
        loaded_data = self.cleaner.load_data(excel_file)
        
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(loaded_data.shape, self.sample_data.shape)
    
    def test_load_data_unsupported_format(self):
        """Test loading unsupported file format"""
        unsupported_file = os.path.join(self.temp_dir, 'test.txt')
        with open(unsupported_file, 'w') as f:
            f.write("test data")
        
        with self.assertRaises(ValueError):
            self.cleaner.load_data(unsupported_file)
    
    def test_load_data_nonexistent_file(self):
        """Test loading non-existent file"""
        with self.assertRaises(FileNotFoundError):
            self.cleaner.load_data('nonexistent_file.csv')
    
    def test_analyze_data(self):
        """Test data analysis functionality"""
        result = self.cleaner.analyze_data(self.sample_data)
        
        self.assertIn('original_shape', result)
        self.assertIn('original_columns', result)
        self.assertIn('data_types', result)
        self.assertIn('missing_values_before', result)
        self.assertIn('duplicates_before', result)
        
        self.assertEqual(result['original_shape'], self.sample_data.shape)
        self.assertEqual(len(result['original_columns']), len(self.sample_data.columns))
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        # Test with default threshold (0.5)
        result = self.cleaner.handle_missing_values(self.sample_data.copy())
        
        # Should keep all columns (missing_col has 30% missing)
        self.assertEqual(result.shape[1], self.sample_data.shape[1])
        
        # Should fill missing values
        self.assertEqual(result['missing_col'].isnull().sum(), 0)
    
    def test_handle_missing_values_high_threshold(self):
        """Test missing value handling with high missing percentage"""
        result = self.cleaner.handle_missing_values(self.high_missing_data.copy(), threshold=0.5)
        
        # Should drop bad_col (80% missing > 50% threshold)
        self.assertEqual(result.shape[1], 1)
        self.assertIn('good_col', result.columns)
        self.assertNotIn('bad_col', result.columns)
    
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        # Create data with duplicates
        duplicate_data = pd.DataFrame({
            'col1': [1, 2, 1, 3, 2],
            'col2': ['A', 'B', 'A', 'C', 'B']
        })
        
        result = self.cleaner.remove_duplicates(duplicate_data)
        
        # Should remove 2 duplicate rows
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(self.cleaner.report['duplicates_removed'], 2)
    
    def test_detect_and_remove_outliers_iqr(self):
        """Test outlier detection using IQR method"""
        result = self.cleaner.detect_and_remove_outliers(self.sample_data.copy(), method='iqr')
        
        # Should detect and remove outlier (100 in numeric_col)
        self.assertLess(result.shape[0], self.sample_data.shape[0])
        self.assertNotIn(100, result['numeric_col'].values)
    
    def test_detect_and_remove_outliers_zscore(self):
        """Test outlier detection using Z-score method"""
        result = self.cleaner.detect_and_remove_outliers(self.sample_data.copy(), method='zscore')
        
        # Should detect outliers
        self.assertIn('outliers_removed', self.cleaner.report)
    
    def test_encode_categorical_variables_label(self):
        """Test label encoding of categorical variables"""
        result = self.cleaner.encode_categorical_variables(self.sample_data.copy(), encoding_method='label')
        
        # Categorical column should be encoded to numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(result['categorical_col']))
        
        # Should have label encoder stored
        self.assertIn('categorical_col', self.cleaner.label_encoders)
    
    def test_encode_categorical_variables_onehot(self):
        """Test one-hot encoding of categorical variables"""
        result = self.cleaner.encode_categorical_variables(self.sample_data.copy(), encoding_method='onehot')
        
        # Should have more columns due to one-hot encoding
        self.assertGreater(result.shape[1], self.sample_data.shape[1])
        
        # Original categorical column should be removed
        self.assertNotIn('categorical_col', result.columns)
    
    def test_normalize_data_minmax(self):
        """Test MinMax normalization"""
        result = self.cleaner.normalize_data(self.sample_data.copy(), method='minmax')
        
        # Numeric columns should be normalized to [0, 1]
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.assertGreaterEqual(result[col].min(), 0)
            self.assertLessEqual(result[col].max(), 1)
    
    def test_normalize_data_standard(self):
        """Test standard normalization"""
        result = self.cleaner.normalize_data(self.sample_data.copy(), method='standard')
        
        # Should have scaler stored
        self.assertIsNotNone(self.cleaner.scaler)
    
    def test_create_output_folder(self):
        """Test output folder creation"""
        test_file = os.path.join(self.temp_dir, 'test_data.csv')
        self.sample_data.to_csv(test_file, index=False)
        
        # Change to temp directory for testing
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            output_folder = self.cleaner.create_output_folder(test_file)
            
            self.assertTrue(output_folder.exists())
            self.assertEqual(output_folder.name, 'Cleans-test_data')
        finally:
            os.chdir(original_cwd)
    
    def test_full_cleaning_pipeline(self):
        """Test complete cleaning pipeline"""
        # Create temporary CSV file
        csv_file = os.path.join(self.temp_dir, 'test_full.csv')
        self.sample_data.to_csv(csv_file, index=False)
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Run full cleaning pipeline
            result = self.cleaner.clean_dataset(csv_file, perform_analysis=False)
            
            # Verify result is a DataFrame
            self.assertIsInstance(result, pd.DataFrame)
            
            # Verify cleaning report was generated
            self.assertIn('final_shape', self.cleaner.report)
            self.assertIn('cleaning_timestamp', self.cleaner.report)
            
            # Verify data was processed
            self.assertGreater(len(self.cleaner.report), 5)
            
        finally:
            os.chdir(original_cwd)

class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_default_config_loading(self):
        """Test default configuration loading"""
        config = ConfigManager()
        
        self.assertIn('cleaning', config.config)
        self.assertIn('analysis', config.config)
        self.assertIn('visualization', config.config)
        self.assertIn('output', config.config)
    
    def test_get_config_value(self):
        """Test getting configuration values"""
        config = ConfigManager()
        
        # Test existing key
        threshold = config.get('cleaning.missing_values.threshold')
        self.assertEqual(threshold, 0.5)
        
        # Test non-existing key with default
        non_existing = config.get('non.existing.key', 'default_value')
        self.assertEqual(non_existing, 'default_value')
    
    def test_set_config_value(self):
        """Test setting configuration values"""
        config = ConfigManager()
        
        config.set('cleaning.missing_values.threshold', 0.8)
        threshold = config.get('cleaning.missing_values.threshold')
        self.assertEqual(threshold, 0.8)
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = ConfigManager()
        
        # Valid configuration should pass
        self.assertTrue(config.validate_config())
        
        # Invalid configuration should fail
        config.set('cleaning.missing_values.threshold', 1.5)  # Invalid: > 1.0
        self.assertFalse(config.validate_config())
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration"""
        config = ConfigManager()
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Modify configuration
        config.set('cleaning.missing_values.threshold', 0.8)
        
        # Save configuration
        config.save_config(config_file)
        self.assertTrue(os.path.exists(config_file))
        
        # Load configuration in new instance
        new_config = ConfigManager(config_file)
        threshold = new_config.get('cleaning.missing_values.threshold')
        self.assertEqual(threshold, 0.8)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDatasetCleaner))
    test_suite.addTest(unittest.makeSuite(TestConfigManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
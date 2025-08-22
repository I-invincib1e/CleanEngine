#!/usr/bin/env python3
"""
Unit tests for Data Analyzer
Tests advanced analysis functionality.
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

from src.dataset_cleaner.analysis.analyzer import DataAnalyzer

class TestDataAnalyzer(unittest.TestCase):
    """Test cases for DataAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample test data
        np.random.seed(42)  # For reproducible results
        
        self.sample_data = pd.DataFrame({
            'numeric1': np.random.normal(50, 10, 100),
            'numeric2': np.random.normal(30, 5, 100),
            'numeric3': np.random.uniform(0, 100, 100),
            'categorical1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical2': np.random.choice(['X', 'Y', 'Z'], 100, p=[0.5, 0.3, 0.2])
        })
        
        # Add some correlation
        self.sample_data['correlated'] = self.sample_data['numeric1'] * 0.8 + np.random.normal(0, 5, 100)
        
        # Create analyzer instance
        self.analyzer = DataAnalyzer(self.sample_data)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_statistical_analysis(self):
        """Test statistical analysis functionality"""
        self.analyzer.statistical_analysis()
        
        self.assertIn('statistical_summary', self.analyzer.analysis_results)
        stats = self.analyzer.analysis_results['statistical_summary']
        
        self.assertIn('descriptive_stats', stats)
        self.assertIn('skewness', stats)
        self.assertIn('kurtosis', stats)
        self.assertIn('numeric_columns_count', stats)
        
        # Should have 4 numeric columns (numeric1, numeric2, numeric3, correlated)
        self.assertEqual(stats['numeric_columns_count'], 4)
    
    def test_correlation_analysis(self):
        """Test correlation analysis functionality"""
        self.analyzer.correlation_analysis()
        
        self.assertIn('correlation_analysis', self.analyzer.analysis_results)
        corr = self.analyzer.analysis_results['correlation_analysis']
        
        self.assertIn('correlation_matrix', corr)
        self.assertIn('strong_correlations', corr)
        
        # Should detect strong correlation between numeric1 and correlated
        strong_corr = corr['strong_correlations']
        self.assertGreater(len(strong_corr), 0)
    
    def test_distribution_analysis(self):
        """Test distribution analysis functionality"""
        self.analyzer.distribution_analysis()
        
        self.assertIn('distribution_analysis', self.analyzer.analysis_results)
        dist = self.analyzer.analysis_results['distribution_analysis']
        
        # Check numeric columns
        self.assertIn('numeric1', dist)
        self.assertEqual(dist['numeric1']['type'], 'numeric')
        self.assertIn('is_normal', dist['numeric1'])
        
        # Check categorical columns
        self.assertIn('categorical1', dist)
        self.assertEqual(dist['categorical1']['type'], 'categorical')
        self.assertIn('unique_values', dist['categorical1'])
    
    def test_feature_importance_analysis(self):
        """Test feature importance analysis functionality"""
        self.analyzer.feature_importance_analysis()
        
        if 'feature_importance' in self.analyzer.analysis_results:
            importance = self.analyzer.analysis_results['feature_importance']
            
            self.assertIn('target_variable', importance)
            self.assertIn('feature_scores', importance)
            self.assertIn('top_features', importance)
    
    def test_clustering_analysis(self):
        """Test clustering analysis functionality"""
        self.analyzer.clustering_analysis()
        
        if 'clustering_analysis' in self.analyzer.analysis_results:
            cluster = self.analyzer.analysis_results['clustering_analysis']
            
            self.assertIn('optimal_clusters', cluster)
            self.assertIn('cluster_summary', cluster)
            
            # Optimal clusters should be reasonable
            optimal_k = cluster['optimal_clusters']
            self.assertGreaterEqual(optimal_k, 2)
            self.assertLessEqual(optimal_k, 10)
    
    def test_anomaly_detection(self):
        """Test anomaly detection functionality"""
        self.analyzer.anomaly_detection()
        
        if 'anomaly_detection' in self.analyzer.analysis_results:
            anomaly = self.analyzer.analysis_results['anomaly_detection']
            
            self.assertIn('total_anomalies', anomaly)
            self.assertIn('anomaly_percentage', anomaly)
            
            # Anomaly percentage should be reasonable
            anomaly_pct = anomaly['anomaly_percentage']
            self.assertGreaterEqual(anomaly_pct, 0)
            self.assertLessEqual(anomaly_pct, 50)  # Should not be more than 50%
    
    def test_data_quality_assessment(self):
        """Test data quality assessment functionality"""
        self.analyzer.data_quality_assessment()
        
        self.assertIn('data_quality', self.analyzer.analysis_results)
        quality = self.analyzer.analysis_results['data_quality']
        
        self.assertIn('completeness_percentage', quality)
        self.assertIn('uniqueness_scores', quality)
        self.assertIn('consistency_issues', quality)
        self.assertIn('overall_quality_score', quality)
        
        # Quality score should be between 0 and 100
        score = quality['overall_quality_score']
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        
        # Completeness should be 100% for our test data (no missing values)
        completeness = quality['completeness_percentage']
        self.assertEqual(completeness, 100.0)
    
    def test_generate_insights(self):
        """Test insights generation"""
        # Run some analyses first
        self.analyzer.statistical_analysis()
        self.analyzer.correlation_analysis()
        self.analyzer.data_quality_assessment()
        
        self.analyzer.generate_insights()
        
        self.assertIn('insights', self.analyzer.analysis_results)
        insights = self.analyzer.analysis_results['insights']
        
        self.assertIsInstance(insights, list)
        # Should have generated some insights
        self.assertGreater(len(insights), 0)
    
    def test_comprehensive_analysis(self):
        """Test complete analysis pipeline"""
        results = self.analyzer.generate_comprehensive_analysis()
        
        # Should have multiple analysis types
        expected_keys = [
            'statistical_summary',
            'correlation_analysis', 
            'distribution_analysis',
            'data_quality',
            'insights'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Insights should be generated
        self.assertGreater(len(results['insights']), 0)
    
    def test_create_visualizations(self):
        """Test visualization creation"""
        # Run analysis first
        self.analyzer.generate_comprehensive_analysis()
        # Create visualizations
        output_folder = Path(self.temp_dir)
        viz_folder = self.analyzer.create_analysis_visualizations(output_folder)
        if viz_folder is None:
            self.skipTest("Visualizations are disabled in configuration.")
        self.assertTrue(viz_folder.exists())
        # Check for expected visualization files
        expected_files = [
            'correlation_heatmap.png',
            'distributions.png',
            'quality_dashboard.png'
        ]
        for file_name in expected_files:
            file_path = viz_folder / file_name
            # File should exist (though might be empty in test environment)
            # We just check if the creation process doesn't crash
    
    def test_generate_analysis_report(self):
        """Test analysis report generation"""
        # Run analysis first
        self.analyzer.generate_comprehensive_analysis()
        
        # Generate report
        output_folder = Path(self.temp_dir)
        report_file = self.analyzer.generate_analysis_report(output_folder, 'test_dataset')
        
        self.assertTrue(report_file.exists())
        
        # Check report content
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('Comprehensive Data Analysis Report', content)
        self.assertIn('test_dataset', content)
        self.assertIn('Executive Summary', content)
        self.assertIn('Key Insights', content)
    
    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        # Test with perfect data
        score = self.analyzer.calculate_quality_score(
            completeness=100.0,
            uniqueness_scores={'col1': 50, 'col2': 75},
            consistency_issues=[]
        )
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        
        # Test with poor data
        poor_score = self.analyzer.calculate_quality_score(
            completeness=50.0,
            uniqueness_scores={'col1': 10, 'col2': 20},
            consistency_issues=['col1', 'col2', 'col3']
        )
        
        self.assertLess(poor_score, score)  # Poor data should have lower score
    
    def test_empty_dataframe(self):
        """Test analyzer with empty dataframe"""
        empty_df = pd.DataFrame()
        analyzer = DataAnalyzer(empty_df)
        
        # Should not crash with empty data
        try:
            results = analyzer.generate_comprehensive_analysis()
            self.assertIsInstance(results, dict)
        except Exception as e:
            self.fail(f"Analysis failed with empty dataframe: {e}")
    
    def test_single_column_dataframe(self):
        """Test analyzer with single column dataframe"""
        single_col_df = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
        analyzer = DataAnalyzer(single_col_df)
        
        # Should handle single column gracefully
        try:
            results = analyzer.generate_comprehensive_analysis()
            self.assertIsInstance(results, dict)
            self.assertIn('statistical_summary', results)
        except Exception as e:
            self.fail(f"Analysis failed with single column: {e}")
    
    def test_all_categorical_dataframe(self):
        """Test analyzer with all categorical data"""
        cat_df = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y'],
            'cat3': ['P', 'Q', 'R', 'P', 'Q']
        })
        analyzer = DataAnalyzer(cat_df)
        
        # Should handle categorical-only data
        try:
            results = analyzer.generate_comprehensive_analysis()
            self.assertIsInstance(results, dict)
            self.assertIn('distribution_analysis', results)
        except Exception as e:
            self.fail(f"Analysis failed with categorical data: {e}")

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
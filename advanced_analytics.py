#!/usr/bin/env python3
"""
Advanced Analytics Engine
Next-generation data analysis with ML-powered insights and predictive capabilities.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import chi2_contingency, f_oneway, pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    def __init__(self, df, target_column=None):
        self.df = df.copy()
        self.target_column = target_column
        self.results = {}
        self.ml_models = {}
        self.recommendations = []
        
    def run_comprehensive_analytics(self):
        """Run all advanced analytics modules"""
        print("🚀 Starting Advanced Analytics Engine...")
        
        # 1. Predictive Modeling Analysis
        self.predictive_modeling_analysis()
        
        # 2. Advanced Statistical Tests
        self.advanced_statistical_tests()
        
        # 3. Dimensionality Reduction Analysis
        self.dimensionality_reduction_analysis()
        
        # 4. Feature Engineering Suggestions
        self.feature_engineering_analysis()
        
        # 5. Data Drift Detection
        self.data_drift_analysis()
        
        # 6. Advanced Visualization Insights
        self.advanced_visualization_insights()
        
        # 7. Business Intelligence Insights
        self.business_intelligence_analysis()
        
        print("✅ Advanced Analytics completed!")
        return self.results
    
    def predictive_modeling_analysis(self):
        """Analyze predictive modeling potential"""
        print("🤖 Analyzing predictive modeling potential...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) < 2:
            self.results['predictive_modeling'] = {'status': 'insufficient_numeric_data'}
            return
        
        # Auto-detect target variable if not specified
        if self.target_column is None:
            # Use the last numeric column as target
            self.target_column = numeric_cols[-1]
        
        if self.target_column not in self.df.columns:
            self.results['predictive_modeling'] = {'status': 'target_not_found'}
            return
        
        # Prepare features
        feature_cols = [col for col in numeric_cols if col != self.target_column]
        if len(feature_cols) == 0:
            self.results['predictive_modeling'] = {'status': 'no_features'}
            return
        
        X = self.df[feature_cols].fillna(0)
        y = self.df[self.target_column].fillna(0)
        
        # Determine if regression or classification
        is_classification = self.df[self.target_column].nunique() < 10 and self.df[self.target_column].dtype == 'object'
        
        try:
            if is_classification:
                # Classification analysis
                le = LabelEncoder()
                y_encoded = le.fit_transform(y.astype(str))
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                # Feature importance
                feature_importance = dict(zip(feature_cols, model.feature_importances_))
                
                self.results['predictive_modeling'] = {
                    'type': 'classification',
                    'target_column': self.target_column,
                    'accuracy': accuracy,
                    'feature_importance': feature_importance,
                    'model_performance': 'good' if accuracy > 0.8 else 'moderate' if accuracy > 0.6 else 'poor'
                }
                
                self.ml_models['classifier'] = model
                
                if accuracy > 0.7:
                    self.recommendations.append(f"Strong predictive potential detected for {self.target_column} (Accuracy: {accuracy:.2f})")
                
            else:
                # Regression analysis
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_siz
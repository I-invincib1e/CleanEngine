#!/usr/bin/env python3
"""
Advanced Data Analyzer Module
Comprehensive data analysis and insights generation for cleaned datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from .time_series import TimeSeriesAnalyzer
from .statistical_tests import StatisticalTester
from ..utils.logger_setup import logger

import warnings
warnings.filterwarnings('ignore')


class DataAnalyzer:
    """Advanced data analysis and insights generation"""
    
    def __init__(self, df, original_df=None, config_manager=None):
        self.df = df.copy()
        self.original_df = original_df.copy() if original_df is not None else None
        self.config = config_manager
        self.logger = logger
        self.analysis_results = {}
        self.insights = []
        
    def generate_comprehensive_analysis(self):
        """Generate comprehensive data analysis"""
        self.logger.info("Starting comprehensive data analysis")
        
        # Only show user feedback if configured
        show_feedback = self.config.get('analysis.show_user_feedback', True) if self.config else True
        if show_feedback:
            print("🔍 Performing comprehensive data analysis...")
        
        # Always perform basic statistical analysis
        self.statistical_analysis()
        
        # Always perform correlation analysis
        self.correlation_analysis()
        
        # Distribution analysis
        self.distribution_analysis()
        
        # Feature importance analysis
        if self.config and self.config.get('analysis.feature_importance.enable', True):
            self.feature_importance_analysis()
        
        # Clustering analysis
        if self.config and self.config.get('analysis.clustering.enable', True):
            self.clustering_analysis()
        
        # Anomaly detection
        if self.config and self.config.get('analysis.anomaly_detection.enable', True):
            self.anomaly_detection()
        
        # Always perform data quality assessment
        self.data_quality_assessment()
        
        # Advanced time series analysis (if applicable)
        self.time_series_analysis()
        
        # Advanced statistical testing
        self.statistical_testing()
        
        # Generate insights
        self.generate_insights()
        
        self.logger.info("Comprehensive analysis completed")
        if show_feedback:
            print("✅ Comprehensive analysis completed!")
        return self.analysis_results
    
    def statistical_analysis(self):
        """Perform statistical analysis"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_summary = self.df[numeric_cols].describe()
            # Additional statistics
            skewness = self.df[numeric_cols].skew()
            kurtosis = self.df[numeric_cols].kurtosis()
            self.analysis_results['statistical_summary'] = {
                'descriptive_stats': stats_summary.to_dict(),
                'skewness': skewness.to_dict(),
                'kurtosis': kurtosis.to_dict(),
                'numeric_columns_count': len(numeric_cols)
            }
            # Identify highly skewed columns
            highly_skewed = skewness[abs(skewness) > 1].to_dict()
            if highly_skewed:
                self.insights.append(f"Highly skewed columns detected: {list(highly_skewed.keys())}")
        else:
            # Always add statistical_summary, even if empty
            self.analysis_results['statistical_summary'] = {
                'descriptive_stats': {},
                'skewness': {},
                'kurtosis': {},
                'numeric_columns_count': 0
            }
    
    def correlation_analysis(self):
        """Analyze correlations between variables"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # Correlation method from config
            corr_method = 'pearson'
            if self.config:
                corr_method = self.config.get('analysis.correlation.method', 'pearson')
                if corr_method not in ('pearson', 'spearman', 'kendall'):
                    corr_method = 'pearson'
            
            correlation_matrix = self.df[numeric_cols].corr(method=corr_method)
            
            # Get correlation threshold from config
            threshold = self.config.get('analysis.correlation.threshold', 0.7) if self.config else 0.7
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        strong_correlations.append({
                            'var1': correlation_matrix.columns[i],
                            'var2': correlation_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            self.analysis_results['correlation_analysis'] = {
                'method': corr_method,
                'correlation_matrix': correlation_matrix.to_dict(),
                'strong_correlations': strong_correlations
            }
            
            if strong_correlations:
                self.insights.append(f"Found {len(strong_correlations)} strong correlations between variables")
        else:
            # Always add correlation_analysis, even if empty
            self.analysis_results['correlation_analysis'] = {
                'method': 'pearson',
                'correlation_matrix': {},
                'strong_correlations': []
            }
    
    def distribution_analysis(self):
        """Analyze data distributions"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        distributions = {}
        
        # Numeric distributions
        for col in numeric_cols:
            # Normality test
            _, p_value = stats.normaltest(self.df[col].dropna())
            is_normal = p_value > 0.05
            
            distributions[col] = {
                'type': 'numeric',
                'is_normal': is_normal,
                'normality_p_value': p_value,
                'unique_values': self.df[col].nunique(),
                'zero_values': (self.df[col] == 0).sum()
            }
        
        # Categorical distributions
        for col in categorical_cols:
            value_counts = self.df[col].value_counts()
            distributions[col] = {
                'type': 'categorical',
                'unique_values': len(value_counts),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'distribution_balance': value_counts.std() / value_counts.mean() if len(value_counts) > 1 else 0
            }
        
        self.analysis_results['distribution_analysis'] = distributions
        
        # Generate insights
        imbalanced_categorical = [col for col, info in distributions.items() 
                                if info.get('type') == 'categorical' and info.get('distribution_balance', 0) > 2]
        if imbalanced_categorical:
            self.insights.append(f"Highly imbalanced categorical variables: {imbalanced_categorical}")
    
    def feature_importance_analysis(self):
        """Analyze feature importance and relationships"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 2:
            # Use the last numeric column as target for demonstration
            target_col = numeric_cols[-1]
            feature_cols = numeric_cols[:-1]
            
            if len(feature_cols) > 0:
                # Calculate mutual information
                mi_scores = mutual_info_regression(
                    self.df[feature_cols].fillna(0), 
                    self.df[target_col].fillna(0)
                )
                
                feature_importance = dict(zip(feature_cols, mi_scores))
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                self.analysis_results['feature_importance'] = {
                    'target_variable': target_col,
                    'feature_scores': feature_importance,
                    'top_features': sorted_features[:5]
                }
                
                if sorted_features:
                    top_feature = sorted_features[0][0]
                    self.insights.append(f"Most important feature for predicting {target_col}: {top_feature}")
            else:
                self.analysis_results['feature_importance'] = {
                    'target_variable': None,
                    'feature_scores': {},
                    'top_features': []
                }
        else:
            # Default empty structure when not applicable
            self.analysis_results['feature_importance'] = {
                'target_variable': None,
                'feature_scores': {},
                'top_features': []
            }
    
    def clustering_analysis(self):
        """Perform clustering analysis"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Default empty structure for stability
        default_result = {
            'optimal_clusters': 0,
            'cluster_summary': {},
            'silhouette_score': None
        }
        
        if len(numeric_cols) >= 2:
            # Prepare data for clustering
            cluster_data = self.df[numeric_cols].fillna(0)
            
            if len(cluster_data) > 10:  # Only cluster if we have enough data
                # Standardize data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                # Choose method
                method = 'kmeans'
                if self.config:
                    method = self.config.get('analysis.clustering.method', 'kmeans')
                
                if method == 'kmeans':
                    # Determine optimal number of clusters using elbow method
                    max_clusters = self.config.get('analysis.clustering.max_clusters', 10) if self.config else 10
                    inertias = []
                    k_range = range(2, min(max_clusters, max(3, len(cluster_data)//2)))
                    
                    for k in k_range:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(scaled_data)
                        inertias.append(kmeans.inertia_)
                    
                    # Find elbow point (simplified)
                    optimal_k = k_range.start
                    if len(inertias) > 2:
                        diffs = np.diff(inertias)
                        optimal_k = list(k_range)[int(np.argmax(diffs))] if len(diffs) > 0 else k_range.start
                    
                    # Perform clustering with optimal k
                    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(scaled_data)
                    
                    # Analyze clusters
                    cluster_summary = {}
                    for i in range(optimal_k):
                        cluster_mask = clusters == i
                        cluster_summary[f'cluster_{i}'] = {
                            'size': int(np.sum(cluster_mask)),
                            'percentage': float(np.sum(cluster_mask) / len(clusters) * 100)
                        }
                    
                    self.analysis_results['clustering_analysis'] = {
                        'method': 'kmeans',
                        'optimal_clusters': int(optimal_k),
                        'cluster_summary': cluster_summary,
                        'silhouette_score': None
                    }
                    self.insights.append(f"Data naturally groups into {optimal_k} clusters")
                    return
                
                elif method == 'dbscan':
                    from sklearn.cluster import DBSCAN
                    eps = self.config.get('analysis.clustering.dbscan.eps', 0.5) if self.config else 0.5
                    min_samples = self.config.get('analysis.clustering.dbscan.min_samples', 5) if self.config else 5
                    db = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_data)
                    labels = db.labels_
                    # Count clusters excluding noise (-1)
                    valid_labels = [l for l in labels if l != -1]
                    n_clusters = len(set(valid_labels))
                    
                    cluster_summary = {}
                    for lbl in sorted(set(labels)):
                        size = int(np.sum(labels == lbl))
                        key = 'noise' if lbl == -1 else f'cluster_{lbl}'
                        cluster_summary[key] = {
                            'size': size,
                            'percentage': float(size / len(labels) * 100)
                        }
                    
                    self.analysis_results['clustering_analysis'] = {
                        'method': 'dbscan',
                        'optimal_clusters': int(n_clusters),
                        'cluster_summary': cluster_summary,
                        'silhouette_score': None
                    }
                    if n_clusters > 0:
                        self.insights.append(f"DBSCAN discovered {n_clusters} clusters (plus noise)")
                    return
                
                elif method == 'hierarchical':
                    from sklearn.cluster import AgglomerativeClustering
                    n_clusters = self.config.get('analysis.clustering.hierarchical.n_clusters', 3) if self.config else 3
                    model = AgglomerativeClustering(n_clusters=int(n_clusters))
                    labels = model.fit_predict(scaled_data)
                    
                    cluster_summary = {}
                    for i in range(int(n_clusters)):
                        size = int(np.sum(labels == i))
                        cluster_summary[f'cluster_{i}'] = {
                            'size': size,
                            'percentage': float(size / len(labels) * 100)
                        }
                    
                    self.analysis_results['clustering_analysis'] = {
                        'method': 'hierarchical',
                        'optimal_clusters': int(n_clusters),
                        'cluster_summary': cluster_summary,
                        'silhouette_score': None
                    }
                    self.insights.append(f"Hierarchical clustering formed {n_clusters} clusters")
                    return
        
        # Default when not applicable
        self.analysis_results['clustering_analysis'] = default_result
    
    def anomaly_detection(self):
        """Detect anomalies in the dataset"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Default result for stability
        default_result = {
            'total_anomalies': 0,
            'anomaly_percentage': 0.0,
            'anomaly_threshold': 10.0
        }
        
        if len(numeric_cols) > 0:
            anomaly_data = self.df[numeric_cols].fillna(0)
            
            if len(anomaly_data) > 10:
                method = 'isolation_forest'
                if self.config:
                    method = self.config.get('analysis.anomaly_detection.method', 'isolation_forest')
                
                if method == 'isolation_forest':
                    contamination = self.config.get('analysis.anomaly_detection.contamination', 0.1) if self.config else 0.1
                    iso_forest = IsolationForest(contamination=contamination, random_state=42)
                    anomaly_labels = iso_forest.fit_predict(anomaly_data)
                    anomaly_count = int(np.sum(anomaly_labels == -1))
                    anomaly_percentage = float(anomaly_count / len(anomaly_labels) * 100)
                    
                    self.analysis_results['anomaly_detection'] = {
                        'method': 'isolation_forest',
                        'total_anomalies': anomaly_count,
                        'anomaly_percentage': anomaly_percentage,
                        'anomaly_threshold': 10.0
                    }
                    if anomaly_percentage > 5:
                        self.insights.append(f"High anomaly rate detected: {anomaly_percentage:.1f}% of data points")
                    return
                
                elif method == 'lof':
                    from sklearn.neighbors import LocalOutlierFactor
                    contamination = self.config.get('analysis.anomaly_detection.contamination', 0.1) if self.config else 0.1
                    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
                    labels = lof.fit_predict(anomaly_data)
                    anomaly_count = int(np.sum(labels == -1))
                    anomaly_percentage = float(anomaly_count / len(labels) * 100)
                    
                    self.analysis_results['anomaly_detection'] = {
                        'method': 'lof',
                        'total_anomalies': anomaly_count,
                        'anomaly_percentage': anomaly_percentage,
                        'anomaly_threshold': 10.0
                    }
                    if anomaly_percentage > 5:
                        self.insights.append(f"High anomaly rate detected (LOF): {anomaly_percentage:.1f}% of data points")
                    return
        
        # Default when not applicable
        self.analysis_results['anomaly_detection'] = default_result
    
    def data_quality_assessment(self):
        """Assess overall data quality"""
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells * 100
        
        # Uniqueness assessment
        uniqueness_scores = {}
        for col in self.df.columns:
            unique_ratio = self.df[col].nunique() / len(self.df) * 100
            uniqueness_scores[col] = unique_ratio
        
        # Consistency assessment (for categorical columns)
        consistency_issues = []
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            # Check for potential inconsistencies (case variations, etc.)
            values = self.df[col].dropna().astype(str)
            lower_values = values.str.lower()
            if len(values.unique()) != len(lower_values.unique()):
                consistency_issues.append(col)
        
        self.analysis_results['data_quality'] = {
            'completeness_percentage': completeness,
            'uniqueness_scores': uniqueness_scores,
            'consistency_issues': consistency_issues,
            'overall_quality_score': self.calculate_quality_score(completeness, uniqueness_scores, consistency_issues)
        }
        
        if completeness < 90:
            self.insights.append(f"Data completeness is {completeness:.1f}% - consider data collection improvements")
        
        if consistency_issues:
            self.insights.append(f"Consistency issues found in columns: {consistency_issues}")
    
    def calculate_quality_score(self, completeness, uniqueness_scores, consistency_issues):
        """Calculate overall data quality score"""
        # Simple scoring algorithm
        score = completeness * 0.4  # 40% weight for completeness
        
        # Add uniqueness component (average uniqueness, capped at 100)
        avg_uniqueness = np.mean(list(uniqueness_scores.values()))
        score += min(avg_uniqueness, 100) * 0.3  # 30% weight
        
        # Subtract for consistency issues
        consistency_penalty = len(consistency_issues) * 5
        score -= consistency_penalty
        
        # Add base score for having data
        score += 30  # 30% base score
        
        return max(0, min(100, score))
    
    def generate_insights(self):
        """Generate actionable insights from analysis"""
        # Add general insights based on analysis results
        
        # Data size insights
        rows, cols = self.df.shape
        if rows < 100:
            self.insights.append("Small dataset - consider collecting more data for robust analysis")
        elif rows > 100000:
            self.insights.append("Large dataset - consider sampling for faster analysis")
        
        # Column insights
        if cols > 50:
            self.insights.append("High-dimensional dataset - consider dimensionality reduction")
        
        # Store all insights
        self.analysis_results['insights'] = self.insights
    
    def time_series_analysis(self):
        """Perform time series analysis if applicable"""
        try:
            if self.original_df is not None:
                # Check if we have time series data
                time_series_analyzer = TimeSeriesAnalyzer(self.df)
                if time_series_analyzer.date_column:
                    show_feedback = self.config.get('analysis.show_user_feedback', True) if self.config else True
                    if show_feedback:
                        print("🕒 Time series data detected - performing temporal analysis...")
                    
                    # Perform time series analysis
                    time_series_results = time_series_analyzer.comprehensive_analysis()
                    self.analysis_results['time_series_analysis'] = time_series_results
                    
                    # Merge insights
                    ts_insights = time_series_results.get('insights', [])
                    self.insights.extend(ts_insights)
                    
                    if show_feedback:
                        print("✅ Time series analysis completed!")
                    
        except Exception as e:
            self.logger.error(f"Time series analysis failed: {str(e)}")
            show_feedback = self.config.get('analysis.show_user_feedback', True) if self.config else True
            if show_feedback:
                print(f"⚠️ Time series analysis failed: {str(e)}")
            self.analysis_results['time_series_analysis'] = {
                'error': str(e)
            }
    
    def statistical_testing(self):
        """Perform advanced statistical testing"""
        try:
            show_feedback = self.config.get('analysis.show_user_feedback', True) if self.config else True
            if show_feedback:
                print("📊 Performing advanced statistical tests...")
            
            # Initialize statistical tester
            stat_tester = StatisticalTester(self.df)
            
            # Run comprehensive testing
            test_results = stat_tester.comprehensive_statistical_testing()
            
            # Merge results
            self.analysis_results['statistical_tests'] = test_results
            
            # Merge insights
            stat_insights = test_results.get('statistical_insights', [])
            self.insights.extend(stat_insights)
            
            if show_feedback:
                print("✅ Statistical testing completed!")
        
        except Exception as e:
            self.logger.error(f"Statistical testing failed: {str(e)}")
            show_feedback = self.config.get('analysis.show_user_feedback', True) if self.config else True
            if show_feedback:
                print(f"⚠️ Statistical testing failed: {str(e)}")
            self.analysis_results['statistical_tests'] = {
                'error': str(e)
            }
    
    def create_analysis_visualizations(self, output_folder):
        """Create comprehensive visualizations"""
        if not self.config or not self.config.get('visualization.enable', True):
            self.logger.info("Visualizations disabled in configuration")
            return None

        show_feedback = self.config.get('analysis.show_user_feedback', True) if self.config else True
        if show_feedback:
            print("📊 Creating analysis visualizations...")
        self.logger.info("Creating analysis visualizations")

        viz_folder = output_folder / "visualizations"
        viz_folder.mkdir(exist_ok=True)
        if show_feedback:
            print(f"📁 Created visualizations folder: {viz_folder}")
        
        # Set style
        style = self.config.get('visualization.style', 'seaborn-v0_8') if self.config else 'seaborn-v0_8'
        plt.style.use(style)
        
        # 1. Correlation heatmap
        self.create_correlation_heatmap(viz_folder)
        
        # 2. Distribution plots
        self.create_distribution_plots(viz_folder)
        
        # 3. Feature importance plot
        self.create_feature_importance_plot(viz_folder)
        
        # 4. Data quality dashboard
        self.create_quality_dashboard(viz_folder)
        
        # 5. Time series visualizations (if applicable)
        self.create_time_series_visualizations(viz_folder)
        
        self.logger.info(f"Visualizations saved in: {viz_folder}")
        if show_feedback:
            print(f"📈 Visualizations saved in: {viz_folder}")
        return viz_folder
    
    def create_correlation_heatmap(self, viz_folder):
        """Create correlation heatmap"""
        show_feedback = self.config.get('analysis.show_user_feedback', True) if self.config else True
        if show_feedback:
            print(f"🔥 Creating correlation heatmap in: {viz_folder}")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if show_feedback:
            print(f"📊 Found {len(numeric_cols)} numeric columns: {list(numeric_cols)}")

        if len(numeric_cols) > 1:
            fig_size = self.config.get('visualization.figure_size', [12, 8]) if self.config else [12, 8]
            plt.figure(figsize=fig_size)
            correlation_matrix = self.df[numeric_cols].corr()

            color_palette = self.config.get('visualization.color_palette', 'viridis') if self.config else 'coolwarm'
            sns.heatmap(correlation_matrix, annot=True, cmap=color_palette, center=0,
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()

            output_file = viz_folder / 'correlation_heatmap.png'
            dpi = self.config.get('visualization.dpi', 300) if self.config else 300
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            if show_feedback:
                print(f"✅ Saved correlation heatmap: {output_file}")
        else:
            if show_feedback:
                print("⚠️ Not enough numeric columns for correlation heatmap")
    
    def create_distribution_plots(self, viz_folder):
        """Create distribution plots for numeric columns"""
        show_feedback = self.config.get('analysis.show_user_feedback', True) if self.config else True
        if show_feedback:
            print(f"🔥 Creating distribution plots in: {viz_folder}")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if show_feedback:
            print(f"📊 Found {len(numeric_cols)} numeric columns: {list(numeric_cols)}")

        if len(numeric_cols) > 0:
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

            fig_size = self.config.get('visualization.figure_size', [12, 8]) if self.config else [12, 8]
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    axes[i].hist(self.df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution: {col}', fontweight='bold')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')

            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            output_file = viz_folder / 'distributions.png'
            dpi = self.config.get('visualization.dpi', 300) if self.config else 300
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            if show_feedback:
                print(f"✅ Saved distribution plots: {output_file}")
        else:
            if show_feedback:
                print("⚠️ No numeric columns found for distribution plots")
    
    def create_feature_importance_plot(self, viz_folder):
        """Create feature importance plot"""
        if 'feature_importance' in self.analysis_results:
            importance_data = self.analysis_results['feature_importance']
            top_features = importance_data['top_features'][:10]  # Top 10 features
            
            if top_features:
                features, scores = zip(*top_features)
                
                fig_size = self.config.get('visualization.figure_size', [12, 8]) if self.config else [10, 6]
                plt.figure(figsize=fig_size)
                bars = plt.barh(range(len(features)), scores)
                plt.yticks(range(len(features)), features)
                plt.xlabel('Importance Score')
                plt.title(f'Top Feature Importance (Target: {importance_data["target_variable"]})', 
                         fontsize=14, fontweight='bold')
                
                # Color bars
                color_palette = self.config.get('visualization.color_palette', 'viridis') if self.config else 'viridis'
                for i, bar in enumerate(bars):
                    bar.set_color(plt.cm.get_cmap(color_palette)(i / len(bars)))
                
                plt.tight_layout()
                dpi = self.config.get('visualization.dpi', 300) if self.config else 300
                plt.savefig(viz_folder / 'feature_importance.png', dpi=dpi, bbox_inches='tight')
                plt.close()
    
    def create_quality_dashboard(self, viz_folder):
        """Create data quality dashboard"""
        if 'data_quality' in self.analysis_results:
            quality_data = self.analysis_results['data_quality']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Overall quality score
            score = quality_data['overall_quality_score']
            colors = ['red' if score < 50 else 'orange' if score < 75 else 'green']
            ax1.bar(['Data Quality Score'], [score], color=colors[0])
            ax1.set_ylim(0, 100)
            ax1.set_ylabel('Score')
            ax1.set_title('Overall Data Quality Score', fontweight='bold')
            ax1.text(0, score + 2, f'{score:.1f}%', ha='center', fontweight='bold')
            
            # 2. Completeness by column
            missing_by_col = self.df.isnull().sum()
            completeness_by_col = (1 - missing_by_col / len(self.df)) * 100
            
            if len(completeness_by_col) > 0:
                top_incomplete = completeness_by_col.nsmallest(10)
                ax2.barh(range(len(top_incomplete)), top_incomplete.values)
                ax2.set_yticks(range(len(top_incomplete)))
                ax2.set_yticklabels(top_incomplete.index, fontsize=8)
                ax2.set_xlabel('Completeness %')
                ax2.set_title('Column Completeness (Bottom 10)', fontweight='bold')
            
            # 3. Data types distribution
            dtype_counts = self.df.dtypes.value_counts()
            ax3.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
            ax3.set_title('Data Types Distribution', fontweight='bold')
            
            # 4. Uniqueness scores
            uniqueness = quality_data['uniqueness_scores']
            if uniqueness:
                cols = list(uniqueness.keys())[:10]  # Top 10 columns
                scores = [uniqueness[col] for col in cols]
                
                ax4.bar(range(len(cols)), scores)
                ax4.set_xticks(range(len(cols)))
                ax4.set_xticklabels(cols, rotation=45, ha='right', fontsize=8)
                ax4.set_ylabel('Uniqueness %')
                ax4.set_title('Column Uniqueness Scores', fontweight='bold')
            
            plt.tight_layout()
            dpi = self.config.get('visualization.dpi', 300) if self.config else 300
            plt.savefig(viz_folder / 'quality_dashboard.png', dpi=dpi, bbox_inches='tight')
            plt.close()
    
    def create_time_series_visualizations(self, viz_folder):
        """Create time series visualizations if applicable"""
        try:
            if 'time_series_analysis' in self.analysis_results:
                time_series_analyzer = TimeSeriesAnalyzer(self.df)
                if time_series_analyzer.date_column:
                    time_series_analyzer.create_visualizations(viz_folder)
                    show_feedback = self.config.get('analysis.show_user_feedback', True) if self.config else True
                    if show_feedback:
                        print("📈 Time series visualizations created!")
        except Exception as e:
            self.logger.error(f"Time series visualization failed: {str(e)}")
            show_feedback = self.config.get('analysis.show_user_feedback', True) if self.config else True
            if show_feedback:
                print(f"⚠️ Time series visualization failed: {str(e)}")
    
    def generate_analysis_report(self, output_folder, dataset_name):
        """Generate comprehensive analysis report"""
        report_file = output_folder / f"{dataset_name}_analysis_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Comprehensive Data Analysis Report\n\n")
            f.write(f"**Dataset:** {dataset_name}\n")
            f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            if 'data_quality' in self.analysis_results:
                quality_score = self.analysis_results['data_quality']['overall_quality_score']
                f.write(f"- **Overall Data Quality Score:** {quality_score:.1f}/100\n")
            
            f.write(f"- **Dataset Size:** {self.df.shape[0]:,} rows × {self.df.shape[1]} columns\n")
            
            if 'clustering_analysis' in self.analysis_results:
                clusters = self.analysis_results['clustering_analysis']['optimal_clusters']
                f.write(f"- **Natural Data Clusters:** {clusters}\n")
            
            f.write("\n")
            
            # Key Insights
            f.write("## Key Insights\n\n")
            for insight in self.insights:
                f.write(f"- {insight}\n")
            f.write("\n")
            
            # Statistical Summary
            if 'statistical_summary' in self.analysis_results:
                f.write("## Statistical Summary\n\n")
                stats_data = self.analysis_results['statistical_summary']
                f.write(f"- **Numeric Columns:** {stats_data['numeric_columns_count']}\n")
                
                if stats_data['skewness']:
                    highly_skewed = {k: v for k, v in stats_data['skewness'].items() if abs(v) > 1}
                    if highly_skewed:
                        f.write(f"- **Highly Skewed Columns:** {len(highly_skewed)}\n")
                f.write("\n")
            
            # Correlation Analysis
            if 'correlation_analysis' in self.analysis_results:
                f.write("## Correlation Analysis\n\n")
                strong_corr = self.analysis_results['correlation_analysis']['strong_correlations']
                f.write(f"- **Strong Correlations Found:** {len(strong_corr)}\n")
                
                for corr in strong_corr[:5]:  # Top 5
                    f.write(f"  - {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f}\n")
                f.write("\n")
            
            # Data Quality Assessment
            if 'data_quality' in self.analysis_results:
                f.write("## Data Quality Assessment\n\n")
                quality_data = self.analysis_results['data_quality']
                f.write(f"- **Completeness:** {quality_data['completeness_percentage']:.1f}%\n")
                f.write(f"- **Consistency Issues:** {len(quality_data['consistency_issues'])} columns\n")
                f.write(f"- **Overall Quality Score:** {quality_data['overall_quality_score']:.1f}/100\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            self.generate_recommendations(f)
            
            f.write("\n---\n")
            f.write("*Report generated by Advanced Data Analyzer*\n")
        
        self.logger.info(f"Analysis report saved: {report_file}")
        show_feedback = self.config.get('analysis.show_user_feedback', True) if self.config else True
        if show_feedback:
            print(f"📄 Analysis report saved: {report_file}")
        return report_file
    
    def generate_recommendations(self, file_handle):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Quality-based recommendations
        if 'data_quality' in self.analysis_results:
            quality_data = self.analysis_results['data_quality']
            
            if quality_data['completeness_percentage'] < 90:
                recommendations.append("**Data Collection:** Improve data collection processes to reduce missing values")
            
            if quality_data['consistency_issues']:
                recommendations.append("**Data Standardization:** Standardize categorical values to improve consistency")
            
            if quality_data['overall_quality_score'] < 70:
                recommendations.append("**Data Governance:** Implement data quality monitoring and validation rules")
        
        # Statistical recommendations
        if 'statistical_summary' in self.analysis_results:
            stats_data = self.analysis_results['statistical_summary']
            highly_skewed = {k: v for k, v in stats_data.get('skewness', {}).items() if abs(v) > 2}
            
            if highly_skewed:
                recommendations.append("**Data Transformation:** Consider log transformation for highly skewed variables")
        
        # Correlation recommendations
        if 'correlation_analysis' in self.analysis_results:
            strong_corr = self.analysis_results['correlation_analysis']['strong_correlations']
            if len(strong_corr) > 5:
                recommendations.append("**Feature Selection:** Consider removing highly correlated features to reduce multicollinearity")
        
        # Clustering recommendations
        if 'clustering_analysis' in self.analysis_results:
            clusters = self.analysis_results['clustering_analysis']['optimal_clusters']
            if clusters > 1:
                recommendations.append(f"**Segmentation:** Consider segmenting analysis by the {clusters} natural data clusters")
        
        # Write recommendations
        if recommendations:
            for rec in recommendations:
                file_handle.write(f"- {rec}\n")
        else:
            file_handle.write("- No specific recommendations at this time - data quality appears good\n")

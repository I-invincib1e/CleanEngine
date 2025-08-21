#!/usr/bin/env python3
"""
Create sample datasets for testing advanced features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_time_series_sample():
    """Create a sample time series dataset"""
    # Generate date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    n_points = len(dates)
    
    # Create synthetic time series data
    np.random.seed(42)
    
    # Base trend
    trend = np.linspace(100, 150, n_points)
    
    # Seasonal component (yearly)
    seasonal_yearly = 20 * np.sin(2 * np.pi * np.arange(n_points) / 365.25)
    
    # Seasonal component (weekly)
    seasonal_weekly = 5 * np.sin(2 * np.pi * np.arange(n_points) / 7)
    
    # Random noise
    noise = np.random.normal(0, 5, n_points)
    
    # Combine components
    sales = trend + seasonal_yearly + seasonal_weekly + noise
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_points, size=20, replace=False)
    sales[anomaly_indices] += np.random.normal(0, 30, 20)
    
    # Create correlated variable
    temperature = 25 + 10 * np.sin(2 * np.pi * np.arange(n_points) / 365.25) + np.random.normal(0, 3, n_points)
    
    # Create categorical variables
    regions = np.random.choice(['North', 'South', 'East', 'West'], n_points)
    products = np.random.choice(['A', 'B', 'C'], n_points, p=[0.5, 0.3, 0.2])
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'temperature': temperature,
        'region': regions,
        'product': products,
        'marketing_spend': np.random.exponential(1000, n_points),
        'customer_satisfaction': np.random.beta(8, 2, n_points) * 10
    })
    
    return df

def create_statistical_sample():
    """Create a sample dataset for statistical testing"""
    np.random.seed(42)
    n = 500
    
    # Create groups
    groups = np.random.choice(['Group_A', 'Group_B', 'Group_C'], n, p=[0.4, 0.35, 0.25])
    
    # Create dependent variable with group effects
    base_score = np.random.normal(75, 10, n)
    group_effects = {'Group_A': 0, 'Group_B': 5, 'Group_C': -3}
    scores = base_score + [group_effects[g] for g in groups]
    
    # Add some non-normal data
    skewed_data = np.random.exponential(2, n)
    
    # Create categorical variables
    categories = np.random.choice(['Cat1', 'Cat2', 'Cat3'], n)
    binary_var = np.random.choice(['Yes', 'No'], n, p=[0.6, 0.4])
    
    # Create correlated variables
    corr_var1 = scores * 0.7 + np.random.normal(0, 5, n)
    corr_var2 = scores * -0.5 + np.random.normal(0, 8, n)
    
    df = pd.DataFrame({
        'group': groups,
        'score': scores,
        'skewed_variable': skewed_data,
        'category': categories,
        'binary_variable': binary_var,
        'correlated_var1': corr_var1,
        'correlated_var2': corr_var2,
        'age': np.random.randint(18, 80, n),
        'income': np.random.lognormal(10, 0.5, n)
    })
    
    return df

def create_mixed_sample():
    """Create a mixed dataset with various data types"""
    np.random.seed(42)
    n = 300
    
    # Time component
    dates = pd.date_range('2024-01-01', periods=n, freq='H')
    
    # Numeric variables
    sensor_reading = 50 + 10 * np.sin(np.arange(n) * 2 * np.pi / 24) + np.random.normal(0, 2, n)
    pressure = np.random.normal(1013, 15, n)
    
    # Categorical variables
    status = np.random.choice(['Normal', 'Warning', 'Critical'], n, p=[0.7, 0.2, 0.1])
    location = np.random.choice(['Building_A', 'Building_B', 'Building_C'], n)
    
    # Missing values
    sensor_reading[np.random.choice(n, 20, replace=False)] = np.nan
    pressure[np.random.choice(n, 15, replace=False)] = np.nan
    
    # Duplicates
    duplicate_indices = np.random.choice(n-50, 10, replace=False)
    for idx in duplicate_indices:
        sensor_reading[idx+1] = sensor_reading[idx]
        pressure[idx+1] = pressure[idx]
        status[idx+1] = status[idx]
        location[idx+1] = location[idx]
    
    df = pd.DataFrame({
        'timestamp': dates,
        'sensor_reading': sensor_reading,
        'pressure': pressure,
        'status': status,
        'location': location,
        'maintenance_due': np.random.choice([True, False], n, p=[0.1, 0.9])
    })
    
    return df

if __name__ == "__main__":
    print("🔧 Creating sample datasets...")
    
    # Create time series sample
    ts_data = create_time_series_sample()
    ts_data.to_csv('sample_timeseries.csv', index=False)
    print(f"✅ Time series sample created: {ts_data.shape}")
    
    # Create statistical sample
    stat_data = create_statistical_sample()
    stat_data.to_csv('sample_statistical.csv', index=False)
    print(f"✅ Statistical sample created: {stat_data.shape}")
    
    # Create mixed sample
    mixed_data = create_mixed_sample()
    mixed_data.to_csv('sample_mixed.csv', index=False)
    print(f"✅ Mixed sample created: {mixed_data.shape}")
    
    print("\n📊 Sample datasets ready for testing!")
    print("- sample_timeseries.csv: Time series data with trends and seasonality")
    print("- sample_statistical.csv: Data for statistical testing")
    print("- sample_mixed.csv: Mixed data types with missing values and duplicates")
#!/usr/bin/env python3
"""
Time Series Analyzer Module
Specialized analysis for temporal data patterns, trends, and seasonality.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class TimeSeriesAnalyzer:
    """Advanced time series analysis capabilities"""

    def __init__(self, df, date_column=None, value_columns=None):
        self.df = df.copy()
        self.date_column = date_column
        self.value_columns = value_columns or []
        self.analysis_results = {}
        self.insights = []

        # Auto-detect date columns if not specified
        if not self.date_column:
            self.date_column = self._detect_date_column()

        # Auto-detect value columns if not specified
        if not self.value_columns:
            self.value_columns = self._detect_value_columns()

        # Prepare time series data
        if self.date_column:
            self._prepare_time_series()

    def _detect_date_column(self):
        """Automatically detect date/datetime columns"""
        date_candidates = []

        for col in self.df.columns:
            # Check if column is already datetime
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                date_candidates.append((col, "datetime"))
                continue

            # Try to parse as datetime
            if self.df[col].dtype == "object":
                try:
                    pd.to_datetime(self.df[col].dropna().head(100))
                    date_candidates.append((col, "parseable"))
                except (ValueError, TypeError) as e:
                    # Column cannot be parsed as datetime
                    pass

            # Check for date-like column names
            date_keywords = [
                "date",
                "time",
                "timestamp",
                "created",
                "updated",
                "year",
                "month",
            ]
            if any(keyword in col.lower() for keyword in date_keywords):
                date_candidates.append((col, "name_based"))

        # Return the best candidate
        if date_candidates:
            # Prefer datetime columns, then parseable, then name-based
            date_candidates.sort(
                key=lambda x: {"datetime": 0, "parseable": 1, "name_based": 2}[x[1]]
            )
            return date_candidates[0][0]

        return None

    def _detect_value_columns(self):
        """Automatically detect numeric value columns for time series analysis"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove date column if it's numeric
        if self.date_column in numeric_cols:
            numeric_cols.remove(self.date_column)

        return numeric_cols

    def _prepare_time_series(self):
        """Prepare data for time series analysis"""
        if not self.date_column:
            return

        # Convert date column to datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_column]):
            self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])

        # Sort by date
        self.df = self.df.sort_values(self.date_column)

        # Set date as index for easier time series operations
        self.ts_df = self.df.set_index(self.date_column)

        # Store time series info
        self.analysis_results["time_series_info"] = {
            "date_column": self.date_column,
            "value_columns": self.value_columns,
            "date_range": {
                "start": self.df[self.date_column].min(),
                "end": self.df[self.date_column].max(),
                "duration_days": (
                    self.df[self.date_column].max() - self.df[self.date_column].min()
                ).days,
            },
            "frequency": self._detect_frequency(),
            "data_points": len(self.df),
        }

    def _detect_frequency(self):
        """Detect the frequency of time series data"""
        if len(self.df) < 2:
            return "unknown"

        # Calculate time differences
        time_diffs = self.df[self.date_column].diff().dropna()

        # Get the most common time difference
        mode_diff = time_diffs.mode()
        if len(mode_diff) > 0:
            days = mode_diff.iloc[0].days
            seconds = mode_diff.iloc[0].seconds

            if days >= 365:
                return "yearly"
            elif days >= 28:
                return "monthly"
            elif days >= 7:
                return "weekly"
            elif days >= 1:
                return "daily"
            elif seconds >= 3600:
                return "hourly"
            elif seconds >= 60:
                return "minutely"
            else:
                return "irregular"

        return "irregular"

    def analyze_trends(self):
        """Analyze trends in time series data"""
        if not self.date_column or not self.value_columns:
            return

        trends = {}

        for col in self.value_columns:
            if col not in self.df.columns:
                continue

            # Remove missing values
            clean_data = self.df[[self.date_column, col]].dropna()
            if len(clean_data) < 3:
                continue

            # Convert dates to numeric for trend analysis
            x = pd.to_numeric(clean_data[self.date_column])
            y = clean_data[col]

            # Linear regression for trend
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            except Exception as e:
                self.insights.append(f"Trend analysis failed for {col}: {str(e)}")
                continue

            # Trend classification
            if p_value < 0.05:  # Significant trend
                if slope > 0:
                    trend_direction = "increasing"
                else:
                    trend_direction = "decreasing"
            else:
                trend_direction = "stable"

            trends[col] = {
                "direction": trend_direction,
                "slope": slope,
                "r_squared": r_value**2,
                "p_value": p_value,
                "strength": (
                    "strong"
                    if abs(r_value) > 0.7
                    else "moderate" if abs(r_value) > 0.3 else "weak"
                ),
            }

            # Generate insights
            if p_value < 0.05 and abs(r_value) > 0.5:
                self.insights.append(
                    f"{col} shows a {trend_direction} trend (R² = {r_value**2:.3f})"
                )

        self.analysis_results["trend_analysis"] = trends

    def analyze_seasonality(self):
        """Analyze seasonal patterns in time series data"""
        if not self.date_column or not self.value_columns:
            return

        seasonality = {}

        for col in self.value_columns:
            if col not in self.df.columns:
                continue

            clean_data = self.df[[self.date_column, col]].dropna()
            if len(clean_data) < 12:  # Need at least 12 points for seasonality
                continue

            # Extract time components
            clean_data = clean_data.copy()
            clean_data["month"] = clean_data[self.date_column].dt.month
            clean_data["day_of_week"] = clean_data[self.date_column].dt.dayofweek
            clean_data["hour"] = clean_data[self.date_column].dt.hour

            # Monthly seasonality
            try:
                monthly_stats = clean_data.groupby("month")[col].agg(
                    ["mean", "std", "count"]
                )
                monthly_cv = (monthly_stats["std"] / monthly_stats["mean"]).mean()
            except Exception as e:
                self.insights.append(f"Seasonality analysis failed for {col}: {str(e)}")
                continue

            # Day of week seasonality
            try:
                dow_stats = clean_data.groupby("day_of_week")[col].agg(
                    ["mean", "std", "count"]
                )
                dow_cv = (dow_stats["std"] / dow_stats["mean"]).mean()
            except Exception as e:
                self.insights.append(f"Seasonality analysis failed for {col}: {str(e)}")
                continue

            # ANOVA test for seasonality
            monthly_groups = [
                group[col].values
                for name, group in clean_data.groupby("month")
                if len(group) > 1
            ]
            dow_groups = [
                group[col].values
                for name, group in clean_data.groupby("day_of_week")
                if len(group) > 1
            ]

            monthly_p = None
            dow_p = None

            if len(monthly_groups) > 1:
                try:
                    _, monthly_p = stats.f_oneway(*monthly_groups)
                except Exception as e:
                    self.insights.append(
                        f"Seasonality analysis failed for {col}: {str(e)}"
                    )
                    pass

            if len(dow_groups) > 1:
                try:
                    _, dow_p = stats.f_oneway(*dow_groups)
                except Exception as e:
                    self.insights.append(
                        f"Seasonality analysis failed for {col}: {str(e)}"
                    )
                    pass

            seasonality[col] = {
                "monthly_seasonality": {
                    "detected": monthly_p is not None and monthly_p < 0.05,
                    "p_value": monthly_p,
                    "coefficient_of_variation": monthly_cv,
                },
                "weekly_seasonality": {
                    "detected": dow_p is not None and dow_p < 0.05,
                    "p_value": dow_p,
                    "coefficient_of_variation": dow_cv,
                },
            }

            # Generate insights
            if monthly_p and monthly_p < 0.05:
                self.insights.append(
                    f"{col} shows significant monthly seasonality (p = {monthly_p:.4f})"
                )
            if dow_p and dow_p < 0.05:
                self.insights.append(
                    f"{col} shows significant weekly seasonality (p = {dow_p:.4f})"
                )

        self.analysis_results["seasonality_analysis"] = seasonality

    def analyze_stationarity(self):
        """Analyze stationarity of time series data"""
        if not self.date_column or not self.value_columns:
            return

        stationarity = {}

        for col in self.value_columns:
            if col not in self.df.columns:
                continue

            clean_data = self.df[col].dropna()
            if len(clean_data) < 10:
                continue

            # Augmented Dickey-Fuller test (simplified version)
            # We'll use a simple approach since we don't have statsmodels

            # Rolling statistics
            window = min(12, len(clean_data) // 4)
            if window < 2:
                window = 2

            rolling_mean = clean_data.rolling(window=window).mean()
            rolling_std = clean_data.rolling(window=window).std()

            # Check if mean and variance are relatively stable
            mean_stability = (
                rolling_mean.std() / rolling_mean.mean()
                if rolling_mean.mean() != 0
                else float("inf")
            )
            std_stability = (
                rolling_std.std() / rolling_std.mean()
                if rolling_std.mean() != 0
                else float("inf")
            )

            # Simple stationarity assessment
            is_stationary = mean_stability < 0.1 and std_stability < 0.5

            stationarity[col] = {
                "is_stationary": is_stationary,
                "mean_stability": mean_stability,
                "std_stability": std_stability,
                "assessment": "stationary" if is_stationary else "non-stationary",
            }

            # Generate insights
            if not is_stationary:
                self.insights.append(
                    f"{col} appears to be non-stationary (may need differencing)"
                )

        self.analysis_results["stationarity_analysis"] = stationarity

    def detect_anomalies_temporal(self):
        """Detect temporal anomalies in time series data"""
        if not self.date_column or not self.value_columns:
            return

        anomalies = {}

        for col in self.value_columns:
            if col not in self.df.columns:
                continue

            clean_data = self.df[[self.date_column, col]].dropna()
            if len(clean_data) < 10:
                continue

            # Rolling z-score method
            window = min(30, len(clean_data) // 4)
            if window < 3:
                window = 3

            rolling_mean = clean_data[col].rolling(window=window, center=True).mean()
            rolling_std = clean_data[col].rolling(window=window, center=True).std()

            z_scores = np.abs((clean_data[col] - rolling_mean) / rolling_std)
            anomaly_threshold = 3.0

            anomaly_mask = z_scores > anomaly_threshold
            anomaly_count = anomaly_mask.sum()

            anomalies[col] = {
                "total_anomalies": anomaly_count,
                "anomaly_percentage": (anomaly_count / len(clean_data)) * 100,
                "anomaly_dates": clean_data.loc[
                    anomaly_mask, self.date_column
                ].tolist(),
                "anomaly_values": clean_data.loc[anomaly_mask, col].tolist(),
            }

            # Generate insights
            if anomaly_count > 0:
                self.insights.append(
                    f"{col} has {anomaly_count} temporal anomalies ({(anomaly_count/len(clean_data)*100):.1f}%)"
                )

        self.analysis_results["temporal_anomalies"] = anomalies

    def analyze_correlations_temporal(self):
        """Analyze temporal correlations between variables"""
        if not self.value_columns or len(self.value_columns) < 2:
            return

        # Create lagged correlations
        correlations = {}
        max_lag = min(10, len(self.df) // 4)

        for i, col1 in enumerate(self.value_columns):
            for col2 in self.value_columns[i + 1 :]:
                if col1 not in self.df.columns or col2 not in self.df.columns:
                    continue

                pair_key = f"{col1}_vs_{col2}"
                correlations[pair_key] = {}

                # Current correlation
                current_corr = self.df[col1].corr(self.df[col2])
                correlations[pair_key]["current"] = current_corr

                # Lagged correlations
                lagged_corrs = {}
                for lag in range(1, max_lag + 1):
                    lagged_col2 = self.df[col2].shift(lag)
                    lagged_corr = self.df[col1].corr(lagged_col2)
                    if not np.isnan(lagged_corr):
                        lagged_corrs[f"lag_{lag}"] = lagged_corr

                correlations[pair_key]["lagged"] = lagged_corrs

                # Find strongest correlation
                all_corrs = [current_corr] + list(lagged_corrs.values())
                strongest_corr = max(all_corrs, key=abs) if all_corrs else 0
                correlations[pair_key]["strongest"] = strongest_corr

                # Generate insights
                if abs(strongest_corr) > 0.7:
                    if abs(current_corr) == abs(strongest_corr):
                        self.insights.append(
                            f"Strong current correlation between {col1} and {col2} ({strongest_corr:.3f})"
                        )
                    else:
                        best_lag = (
                            max(lagged_corrs.items(), key=lambda x: abs(x[1]))[0]
                            if lagged_corrs
                            else "unknown"
                        )
                        self.insights.append(
                            f"Strong lagged correlation between {col1} and {col2} at {best_lag} ({strongest_corr:.3f})"
                        )

        self.analysis_results["temporal_correlations"] = correlations

    def generate_time_series_insights(self):
        """Generate comprehensive time series insights"""
        if not self.date_column:
            self.insights.append(
                "No date column detected - time series analysis limited"
            )
            return

        # Data coverage insights
        ts_info = self.analysis_results.get("time_series_info", {})
        if ts_info:
            duration = ts_info.get("duration_days", 0)
            frequency = ts_info.get("frequency", "unknown")
            data_points = ts_info.get("data_points", 0)

            self.insights.append(
                f"Time series spans {duration} days with {data_points} data points ({frequency} frequency)"
            )

            if duration < 30:
                self.insights.append(
                    "Short time series - consider collecting more historical data"
                )
            elif duration > 365:
                self.insights.append(
                    "Long time series available - good for seasonal analysis"
                )

        # Frequency insights
        if ts_info.get("frequency") == "irregular":
            self.insights.append(
                "Irregular time intervals detected - consider data resampling"
            )

    def comprehensive_time_series_analysis(self):
        """Run complete time series analysis pipeline"""
        if not self.date_column:
            self.insights.append("No date column found - skipping time series analysis")
            return self.analysis_results

        print("🕒 Performing time series analysis...")

        # Run all analyses
        self.analyze_trends()
        self.analyze_seasonality()
        self.analyze_stationarity()
        self.detect_anomalies_temporal()
        self.analyze_correlations_temporal()
        self.generate_time_series_insights()

        # Store insights
        self.analysis_results["time_series_insights"] = self.insights

        print("✅ Time series analysis completed!")
        return self.analysis_results

    def create_time_series_visualizations(self, output_folder):
        """Create time series specific visualizations"""
        if not self.date_column or not self.value_columns:
            return None

        viz_folder = output_folder / "time_series_visualizations"
        viz_folder.mkdir(exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")

        # 1. Time series plots
        self._create_time_series_plots(viz_folder)

        # 2. Trend analysis plots
        self._create_trend_plots(viz_folder)

        # 3. Seasonality plots
        self._create_seasonality_plots(viz_folder)

        # 4. Anomaly detection plots
        self._create_anomaly_plots(viz_folder)

        print(f"📈 Time series visualizations saved in: {viz_folder}")
        return viz_folder

    # --- Adapters for compatibility with analyzer expectations ---
    def comprehensive_analysis(self):
        """Adapter: alias for comprehensive_time_series_analysis()."""
        return self.comprehensive_time_series_analysis()

    def create_visualizations(self, output_folder):
        """Adapter: alias for create_time_series_visualizations()."""
        return self.create_time_series_visualizations(output_folder)

    def _create_time_series_plots(self, viz_folder):
        """Create basic time series plots"""
        n_cols = min(2, len(self.value_columns))
        n_rows = (len(self.value_columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        for i, col in enumerate(self.value_columns[: len(axes)]):
            if col in self.df.columns:
                clean_data = self.df[[self.date_column, col]].dropna()
                axes[i].plot(clean_data[self.date_column], clean_data[col], linewidth=1)
                axes[i].set_title(f"Time Series: {col}", fontweight="bold")
                axes[i].set_xlabel("Date")
                axes[i].set_ylabel(col)
                axes[i].tick_params(axis="x", rotation=45)

        # Hide empty subplots
        for i in range(len(self.value_columns), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(viz_folder / "time_series_plots.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_trend_plots(self, viz_folder):
        """Create trend analysis plots"""
        trends = self.analysis_results.get("trend_analysis", {})
        if not trends:
            return

        n_cols = min(2, len(trends))
        n_rows = (len(trends) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        for i, (col, trend_info) in enumerate(trends.items()):
            if i >= len(axes) or col not in self.df.columns:
                continue

            clean_data = self.df[[self.date_column, col]].dropna()
            x_numeric = pd.to_numeric(clean_data[self.date_column])

            # Plot data points
            axes[i].scatter(
                clean_data[self.date_column], clean_data[col], alpha=0.6, s=20
            )

            # Plot trend line
            slope = trend_info["slope"]
            intercept = x_numeric.iloc[0] * slope  # Adjust intercept for plotting
            trend_line = slope * x_numeric + (
                clean_data[col].iloc[0] - slope * x_numeric.iloc[0]
            )
            axes[i].plot(
                clean_data[self.date_column],
                trend_line,
                "r-",
                linewidth=2,
                label=f"Trend: {trend_info['direction']}",
            )

            axes[i].set_title(
                f'Trend Analysis: {col}\nR² = {trend_info["r_squared"]:.3f}',
                fontweight="bold",
            )
            axes[i].set_xlabel("Date")
            axes[i].set_ylabel(col)
            axes[i].legend()
            axes[i].tick_params(axis="x", rotation=45)

        # Hide empty subplots
        for i in range(len(trends), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(viz_folder / "trend_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_seasonality_plots(self, viz_folder):
        """Create seasonality analysis plots"""
        seasonality = self.analysis_results.get("seasonality_analysis", {})
        if not seasonality:
            return

        for col, season_info in seasonality.items():
            if col not in self.df.columns:
                continue

            clean_data = self.df[[self.date_column, col]].dropna()
            if len(clean_data) < 12:
                continue

            clean_data = clean_data.copy()
            clean_data["month"] = clean_data[self.date_column].dt.month
            clean_data["day_of_week"] = clean_data[self.date_column].dt.dayofweek

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Monthly seasonality
            try:
                monthly_avg = clean_data.groupby("month")[col].mean()
                ax1.bar(monthly_avg.index, monthly_avg.values)
                ax1.set_title(f"Monthly Seasonality: {col}", fontweight="bold")
                ax1.set_xlabel("Month")
                ax1.set_ylabel(f"Average {col}")
                ax1.set_xticks(range(1, 13))
            except Exception as e:
                self.insights.append(f"Seasonality plot failed for {col}: {str(e)}")
                continue

            # Weekly seasonality
            try:
                dow_avg = clean_data.groupby("day_of_week")[col].mean()
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                ax2.bar(range(7), dow_avg.values)
                ax2.set_title(f"Weekly Seasonality: {col}", fontweight="bold")
                ax2.set_xlabel("Day of Week")
                ax2.set_ylabel(f"Average {col}")
                ax2.set_xticks(range(7))
                ax2.set_xticklabels(day_names)
            except Exception as e:
                self.insights.append(f"Seasonality plot failed for {col}: {str(e)}")
                continue

            plt.tight_layout()
            plt.savefig(
                viz_folder / f"seasonality_{col}.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    def _create_anomaly_plots(self, viz_folder):
        """Create anomaly detection plots"""
        anomalies = self.analysis_results.get("temporal_anomalies", {})
        if not anomalies:
            return

        for col, anomaly_info in anomalies.items():
            if col not in self.df.columns or anomaly_info["total_anomalies"] == 0:
                continue

            clean_data = self.df[[self.date_column, col]].dropna()

            plt.figure(figsize=(15, 6))

            # Plot normal data
            plt.plot(
                clean_data[self.date_column],
                clean_data[col],
                "b-",
                alpha=0.7,
                label="Normal",
            )

            # Plot anomalies
            anomaly_dates = pd.to_datetime(anomaly_info["anomaly_dates"])
            anomaly_values = anomaly_info["anomaly_values"]
            plt.scatter(
                anomaly_dates,
                anomaly_values,
                color="red",
                s=50,
                label=f"Anomalies ({len(anomaly_values)})",
                zorder=5,
            )

            plt.title(f"Temporal Anomaly Detection: {col}", fontweight="bold")
            plt.xlabel("Date")
            plt.ylabel(col)
            plt.legend()
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(
                viz_folder / f"anomalies_{col}.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

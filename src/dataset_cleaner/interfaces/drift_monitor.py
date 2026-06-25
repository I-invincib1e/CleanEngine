#!/usr/bin/env python3
"""
Data Drift Monitor
Compares two versions of the same dataset (e.g. last week vs this week)
and produces a structured drift report.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _ks_drift(series_ref: pd.Series, series_new: pd.Series) -> dict:
    """Kolmogorov-Smirnov test for numeric drift."""
    ref_clean = series_ref.dropna()
    new_clean = series_new.dropna()
    if len(ref_clean) < 5 or len(new_clean) < 5:
        return {"test": "ks", "statistic": None, "p_value": None, "drifted": False}
    stat, p = stats.ks_2samp(ref_clean, new_clean)
    return {
        "test": "ks",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drifted": p < 0.05,
    }


def _chi2_drift(series_ref: pd.Series, series_new: pd.Series) -> dict:
    """Chi-squared test for categorical drift."""
    ref_counts = series_ref.value_counts(normalize=True)
    new_counts = series_new.value_counts(normalize=True)
    all_cats = set(ref_counts.index) | set(new_counts.index)
    ref_freq = np.array([ref_counts.get(c, 0) for c in all_cats])
    new_freq = np.array([new_counts.get(c, 0) for c in all_cats])
    if ref_freq.sum() == 0 or new_freq.sum() == 0:
        return {"test": "chi2", "statistic": None, "p_value": None, "drifted": False}
    expected = ref_freq * len(series_new)
    observed = new_freq * len(series_new)
    try:
        stat, p = stats.chisquare(observed + 1e-9, expected + 1e-9)
        return {
            "test": "chi2",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "drifted": p < 0.05,
        }
    except Exception:
        return {"test": "chi2", "statistic": None, "p_value": None, "drifted": False}


def _numeric_summary(series: pd.Series) -> dict:
    clean = series.dropna()
    if len(clean) == 0:
        return {}
    return {
        "mean": round(float(clean.mean()), 4),
        "std": round(float(clean.std()), 4),
        "min": round(float(clean.min()), 4),
        "p25": round(float(clean.quantile(0.25)), 4),
        "median": round(float(clean.median()), 4),
        "p75": round(float(clean.quantile(0.75)), 4),
        "max": round(float(clean.max()), 4),
    }


def _categorical_summary(series: pd.Series, top_n: int = 5) -> dict:
    counts = series.value_counts()
    return {
        "unique_values": int(counts.shape[0]),
        "top_values": {str(k): int(v) for k, v in counts.head(top_n).items()},
    }


def compare_datasets(df_ref: pd.DataFrame, df_new: pd.DataFrame) -> dict[str, Any]:
    """
    Compare two dataframes and return a comprehensive drift report.

    Args:
        df_ref: Reference (baseline) dataset
        df_new: New dataset to compare against the reference

    Returns:
        dict with schema_changes, missing_drift, distribution_drift, summary
    ]
    """
    report: dict[str, Any] = {
        "reference_shape": {"rows": len(df_ref), "cols": df_ref.shape[1]},
        "new_shape": {"rows": len(df_new), "cols": df_new.shape[1]},
        "schema_changes": {},
        "missing_drift": {},
        "distribution_drift": {},
        "new_categories": {},
        "summary": {},
    }

    ref_cols = set(df_ref.columns)
    new_cols = set(df_new.columns)

    # Schema changes
    report["schema_changes"] = {
        "added_columns": sorted(new_cols - ref_cols),
        "removed_columns": sorted(ref_cols - new_cols),
        "common_columns": sorted(ref_cols & new_cols),
    }

    common = report["schema_changes"]["common_columns"]

    drifted_cols = []
    missing_changes = []
    new_cat_cols = []

    for col in common:
        ref_series = df_ref[col]
        new_series = df_new[col]

        # Missing value drift
        ref_missing_pct = round(ref_series.isnull().mean() * 100, 2)
        new_missing_pct = round(new_series.isnull().mean() * 100, 2)
        missing_delta = round(new_missing_pct - ref_missing_pct, 2)
        significant_missing = abs(missing_delta) > 5

        report["missing_drift"][col] = {
            "reference_pct": ref_missing_pct,
            "new_pct": new_missing_pct,
            "delta_pct": missing_delta,
            "significant_change": significant_missing,
        }
        if significant_missing:
            missing_changes.append(col)

        # Distribution drift
        is_numeric = pd.api.types.is_numeric_dtype(ref_series) and pd.api.types.is_numeric_dtype(new_series)

        if is_numeric:
            drift_result = _ks_drift(ref_series, new_series)
            ref_stats = _numeric_summary(ref_series)
            new_stats = _numeric_summary(new_series)

            mean_delta = None
            if ref_stats and new_stats and ref_stats.get("mean") is not None:
                mean_delta = round(new_stats["mean"] - ref_stats["mean"], 4)

            report["distribution_drift"][col] = {
                "type": "numeric",
                "statistical_test": drift_result,
                "reference_stats": ref_stats,
                "new_stats": new_stats,
                "mean_delta": mean_delta,
            }
        else:
            drift_result = _chi2_drift(ref_series.astype(str), new_series.astype(str))
            ref_summary = _categorical_summary(ref_series.astype(str))
            new_summary = _categorical_summary(new_series.astype(str))

            # New categories that didn't exist before
            ref_cats = set(ref_series.dropna().astype(str).unique())
            new_cats = set(new_series.dropna().astype(str).unique())
            added_cats = sorted(new_cats - ref_cats)

            report["distribution_drift"][col] = {
                "type": "categorical",
                "statistical_test": drift_result,
                "reference_stats": ref_summary,
                "new_stats": new_summary,
                "new_categories": added_cats,
            }
            if added_cats:
                report["new_categories"][col] = added_cats
                new_cat_cols.append(col)

        if drift_result.get("drifted"):
            drifted_cols.append(col)

    # Summary
    total_common = len(common)
    drift_rate = round(len(drifted_cols) / total_common * 100, 1) if total_common > 0 else 0

    if drift_rate == 0:
        severity = "none"
    elif drift_rate < 20:
        severity = "low"
    elif drift_rate < 50:
        severity = "medium"
    else:
        severity = "high"

    report["summary"] = {
        "total_columns_compared": total_common,
        "columns_with_drift": len(drifted_cols),
        "drifted_column_names": drifted_cols,
        "drift_rate_pct": drift_rate,
        "severity": severity,
        "columns_with_missing_changes": missing_changes,
        "columns_with_new_categories": new_cat_cols,
        "row_count_change": len(df_new) - len(df_ref),
        "row_count_change_pct": round((len(df_new) - len(df_ref)) / max(len(df_ref), 1) * 100, 2),
    }

    return report

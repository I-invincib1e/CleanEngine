#!/usr/bin/env python3
"""
Anomaly Explainer
Detects anomalous rows using Isolation Forest and explains *why* each row
is anomalous by identifying which features deviate most from the norm.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def explain_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
    max_anomalies: int = 50,
) -> dict[str, Any]:
    """
    Detect and explain anomalous rows in a dataframe.

    Args:
        df: Input dataframe (original, pre-encoded values preferred for readability)
        contamination: Expected proportion of anomalies (0.01–0.5)
        max_anomalies: Maximum anomalies to return with explanations

    Returns:
        dict with anomaly_rows, column_stats, and summary
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        return {
            "error": "No numeric columns found for anomaly detection.",
            "anomaly_rows": [],
            "summary": {"total_anomalies": 0},
        }

    df_numeric = df[numeric_cols].copy()

    # Fill missing values with median for detection (don't modify original)
    for col in numeric_cols:
        df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())

    # Scale for Isolation Forest
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    # Compute column statistics on original data (for explanations)
    col_stats = {}
    for col in numeric_cols:
        clean = df[col].dropna()
        if len(clean) == 0:
            continue
        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1
        col_stats[col] = {
            "mean": float(clean.mean()),
            "std": float(clean.std()),
            "median": float(clean.median()),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": q1 - 1.5 * iqr,
            "upper_bound": q3 + 1.5 * iqr,
            "min": float(clean.min()),
            "max": float(clean.max()),
        }

    # Fit Isolation Forest
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    labels = iso.fit_predict(X_scaled)
    scores = iso.score_samples(X_scaled)

    # Anomaly mask (-1 = anomaly)
    anomaly_mask = labels == -1
    anomaly_indices = np.where(anomaly_mask)[0]

    # Sort by anomaly score (most anomalous first)
    sorted_indices = anomaly_indices[np.argsort(scores[anomaly_indices])]
    top_indices = sorted_indices[:max_anomalies]

    explained_rows = []
    for idx in top_indices:
        row = df.iloc[idx]
        row_original_index = df.index[idx]
        score = float(scores[idx])

        # Build per-feature deviation analysis
        feature_deviations = []
        for col in numeric_cols:
            if col not in col_stats:
                continue
            stats = col_stats[col]
            val = float(df_numeric[col].iloc[idx])

            # Z-score deviation
            z = (val - stats["mean"]) / max(stats["std"], 1e-9)

            # IQR-based flags
            below_lower = val < stats["lower_bound"]
            above_upper = val > stats["upper_bound"]

            if abs(z) >= 2 or below_lower or above_upper:
                percentile = float(
                    (df_numeric[col] <= val).mean() * 100
                )
                direction = "high" if val > stats["median"] else "low"
                explanation = _build_explanation(col, val, z, percentile, direction, stats)

                feature_deviations.append({
                    "column": col,
                    "value": round(val, 4),
                    "z_score": round(z, 2),
                    "percentile": round(percentile, 1),
                    "direction": direction,
                    "below_iqr_lower": bool(below_lower),
                    "above_iqr_upper": bool(above_upper),
                    "explanation": explanation,
                })

        # Sort deviations by absolute z-score (worst first)
        feature_deviations.sort(key=lambda x: abs(x["z_score"]), reverse=True)

        # Build human-readable summary
        if feature_deviations:
            top_reason = feature_deviations[0]
            summary_text = _build_row_summary(idx, feature_deviations)
        else:
            summary_text = f"Row {idx}: Unusual combination of values across multiple features."

        # Include original (non-numeric) values for context
        row_data = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                row_data[col] = None
            elif isinstance(val, (np.integer,)):
                row_data[col] = int(val)
            elif isinstance(val, (np.floating,)):
                row_data[col] = round(float(val), 4)
            else:
                row_data[col] = str(val)

        explained_rows.append({
            "row_index": int(row_original_index),
            "anomaly_score": round(score, 4),
            "feature_deviations": feature_deviations,
            "summary": summary_text,
            "row_data": row_data,
        })

    return {
        "anomaly_rows": explained_rows,
        "column_stats": col_stats,
        "summary": {
            "total_rows": len(df),
            "total_anomalies": int(anomaly_mask.sum()),
            "anomaly_rate_pct": round(float(anomaly_mask.mean()) * 100, 2),
            "columns_analyzed": numeric_cols,
            "contamination_setting": contamination,
        },
    }


def _build_explanation(
    col: str,
    val: float,
    z: float,
    percentile: float,
    direction: str,
    stats: dict,
) -> str:
    parts = []
    direction_word = "above" if direction == "high" else "below"
    parts.append(
        f"Value {val:.4g} is in the {percentile:.0f}th percentile "
        f"({direction_word} {100 - percentile:.0f}% of records)"
    )
    if abs(z) >= 3:
        parts.append(f"z-score = {z:.1f} (extreme, >3σ from mean)")
    elif abs(z) >= 2:
        parts.append(f"z-score = {z:.1f} (unusual, >2σ from mean)")

    if val < stats["lower_bound"]:
        parts.append(
            f"Below IQR lower fence ({stats['lower_bound']:.4g})"
        )
    elif val > stats["upper_bound"]:
        parts.append(
            f"Above IQR upper fence ({stats['upper_bound']:.4g})"
        )

    return "; ".join(parts) + "."


def _build_row_summary(idx: int, deviations: list[dict]) -> str:
    if not deviations:
        return f"Row {idx}: Anomaly due to unusual combination of values."

    top = deviations[:3]
    parts = []
    for d in top:
        direction = "unusually high" if d["direction"] == "high" else "unusually low"
        parts.append(f"`{d['column']}` = {d['value']:.4g} ({direction}, {d['percentile']:.0f}th pct)")

    if len(deviations) > 3:
        parts.append(f"and {len(deviations) - 3} more deviating features")

    return f"Row {idx} is anomalous because: " + "; ".join(parts) + "."

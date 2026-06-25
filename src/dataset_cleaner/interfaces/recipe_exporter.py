#!/usr/bin/env python3
"""
Cleaning Recipe Exporter
Converts cleaning steps captured in the cleaner report into executable Python scripts and SQL.
"""

from datetime import datetime


def generate_python_recipe(report: dict, filename: str = "dataset") -> str:
    """Generate a reusable Python cleaning script from a cleaning report."""
    stem = filename.replace(".", "_").replace("-", "_")
    lines = [
        "#!/usr/bin/env python3",
        '"""',
        f"CleanEngine — Auto-generated cleaning recipe for: {filename}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Run this script to reproduce the exact cleaning steps applied in CleanEngine.",
        '"""',
        "",
        "import pandas as pd",
        "import numpy as np",
        "from scipy import stats",
        "from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler",
        "",
        f"INPUT_FILE = '{filename}'",
        f"OUTPUT_FILE = 'cleaned_{filename}'",
        "",
        "",
        "def load_data(path):",
        "    ext = path.rsplit('.', 1)[-1].lower()",
        "    if ext == 'csv':",
        "        for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:",
        "            try:",
        "                return pd.read_csv(path, encoding=enc)",
        "            except UnicodeDecodeError:",
        "                continue",
        "    elif ext in ('xlsx', 'xls'):",
        "        return pd.read_excel(path)",
        "    elif ext == 'json':",
        "        return pd.read_json(path)",
        "    elif ext == 'parquet':",
        "        return pd.read_parquet(path)",
        "    raise ValueError(f'Unsupported file format: {ext}')",
        "",
        "",
        "def clean(df: pd.DataFrame) -> pd.DataFrame:",
        '    """Apply all cleaning steps in sequence."""',
        "    df = df.copy()",
        "",
    ]

    # Step 1: drop high-missing columns
    dropped_cols = [
        k.replace("dropped_column_", "")
        for k in report
        if k.startswith("dropped_column_")
    ]
    if dropped_cols:
        lines.append("    # --- Step 1: Drop high-missing columns ---")
        for col in dropped_cols:
            reason = report.get(f"dropped_column_{col}", "")
            lines.append(f"    # Reason: {reason}")
            lines.append(f"    if '{col}' in df.columns:")
            lines.append(f"        df = df.drop(columns=['{col}'])")
        lines.append("")

    # Step 2: fill missing values
    missing_before = report.get("missing_values_before", {})
    missing_after_cols = report.get("columns_after_missing_cleanup", [])
    if missing_before:
        lines.append("    # --- Step 2: Impute missing values ---")
        lines.append("    for col in df.select_dtypes(include=[np.number]).columns:")
        lines.append("        df[col] = df[col].fillna(df[col].median())")
        lines.append("    for col in df.select_dtypes(include=['object']).columns:")
        lines.append("        mode = df[col].mode()")
        lines.append("        df[col] = df[col].fillna(mode[0] if not mode.empty else 'Unknown')")
        lines.append("")

    # Step 3: remove duplicates
    dupes = report.get("duplicates_removed", 0)
    lines.append("    # --- Step 3: Remove duplicate rows ---")
    lines.append(f"    # {dupes} duplicates were removed during original cleaning")
    lines.append("    df = df.drop_duplicates(keep='first')")
    lines.append("")

    # Step 4: outlier removal
    outliers = report.get("outliers_removed", {})
    outlier_cols = [col for col, cnt in outliers.items() if cnt > 0]
    if outlier_cols:
        lines.append("    # --- Step 4: Remove outliers (IQR method) ---")
        for col in outlier_cols:
            lines.append(f"    if '{col}' in df.columns:")
            lines.append(f"        Q1 = df['{col}'].quantile(0.25)")
            lines.append(f"        Q3 = df['{col}'].quantile(0.75)")
            lines.append(f"        IQR = Q3 - Q1")
            lines.append(f"        df = df[(df['{col}'] >= Q1 - 1.5 * IQR) & (df['{col}'] <= Q3 + 1.5 * IQR)]")
        lines.append("")

    # Step 5: categorical encoding
    encoding = report.get("categorical_encoding", {})
    if encoding:
        lines.append("    # --- Step 5: Encode categorical variables ---")
        label_cols = [c for c, m in encoding.items() if "label" in m]
        onehot_cols = [c for c, m in encoding.items() if "one_hot" in m]
        if label_cols:
            lines.append("    from sklearn.preprocessing import LabelEncoder")
            for col in label_cols:
                lines.append(f"    if '{col}' in df.columns:")
                lines.append(f"        le = LabelEncoder()")
                lines.append(f"        df['{col}'] = le.fit_transform(df['{col}'].astype(str))")
        if onehot_cols:
            for col in onehot_cols:
                lines.append(f"    if '{col}' in df.columns:")
                lines.append(f"        df = pd.get_dummies(df, columns=['{col}'], prefix='{col}')")
        lines.append("")

    # Step 6: normalization
    norm_method = report.get("normalization_method")
    norm_cols = report.get("normalized_columns", [])
    if norm_method and norm_cols:
        lines.append("    # --- Step 6: Normalize numeric columns ---")
        scaler_class = "MinMaxScaler()" if norm_method == "minmax" else "StandardScaler()"
        lines.append(f"    scaler = {scaler_class}")
        cols_repr = repr(norm_cols)
        lines.append(f"    num_cols = [c for c in {cols_repr} if c in df.columns]")
        lines.append("    if num_cols:")
        lines.append("        df[num_cols] = scaler.fit_transform(df[num_cols])")
        lines.append("")

    lines += [
        "    return df",
        "",
        "",
        "if __name__ == '__main__':",
        "    df_raw = load_data(INPUT_FILE)",
        "    print(f'Loaded {len(df_raw):,} rows × {df_raw.shape[1]} columns')",
        "    df_clean = clean(df_raw)",
        "    print(f'Cleaned: {len(df_clean):,} rows × {df_clean.shape[1]} columns')",
        "    df_clean.to_csv(OUTPUT_FILE, index=False)",
        "    print(f'Saved to {OUTPUT_FILE}')",
    ]

    return "\n".join(lines)


def generate_sql_recipe(report: dict, table_name: str = "raw_data") -> str:
    """Generate SQL transformation query from a cleaning report."""
    cleaned_table = f"cleaned_{table_name}"
    dropped_cols = [
        k.replace("dropped_column_", "")
        for k in report
        if k.startswith("dropped_column_")
    ]
    original_cols = report.get("original_columns", [])
    keep_cols = [c for c in original_cols if c not in dropped_cols]
    encoding = report.get("categorical_encoding", {})
    norm_cols = set(report.get("normalized_columns", []))
    norm_method = report.get("normalization_method", "minmax")

    lines = [
        "-- ============================================================",
        f"-- CleanEngine — Auto-generated SQL recipe for: {table_name}",
        f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "-- ============================================================",
        "",
        f"-- Step 1: Remove duplicates and filter outliers",
        f"CREATE TABLE {cleaned_table} AS",
        "WITH",
        "deduped AS (",
        f"    SELECT DISTINCT * FROM {table_name}",
        "),",
    ]

    # Outlier CTE per column
    outlier_cols = [col for col, cnt in report.get("outliers_removed", {}).items() if cnt > 0]
    if outlier_cols:
        lines.append("outlier_bounds AS (")
        lines.append("    SELECT")
        for col in outlier_cols:
            lines.append(f"        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}) AS {col}_q1,")
            lines.append(f"        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}) AS {col}_q3")
        lines.append("    FROM deduped")
        lines.append("),")

    lines += [
        "filtered AS (",
        "    SELECT d.*",
        "    FROM deduped d",
    ]
    if outlier_cols:
        lines.append("    CROSS JOIN outlier_bounds ob")
        where_clauses = []
        for col in outlier_cols:
            where_clauses.append(
                f"        d.{col} BETWEEN ob.{col}_q1 - 1.5*(ob.{col}_q3-ob.{col}_q1) "
                f"AND ob.{col}_q3 + 1.5*(ob.{col}_q3-ob.{col}_q1)"
            )
        lines.append("    WHERE")
        lines.append("\n    AND ".join(where_clauses))
    lines.append(")")

    lines.append("-- Step 2: Select columns, handle missing values, encode")
    lines.append("SELECT")

    select_parts = []
    for col in keep_cols:
        enc = encoding.get(col, "")
        if "label" in enc:
            select_parts.append(f"    DENSE_RANK() OVER (ORDER BY COALESCE({col}, 'Unknown')) - 1 AS {col}")
        elif col in norm_cols:
            if norm_method == "minmax":
                select_parts.append(
                    f"    (CAST({col} AS FLOAT) - MIN(CAST({col} AS FLOAT)) OVER()) / "
                    f"NULLIF(MAX(CAST({col} AS FLOAT)) OVER() - MIN(CAST({col} AS FLOAT)) OVER(), 0) AS {col}"
                )
            else:
                select_parts.append(
                    f"    (CAST({col} AS FLOAT) - AVG(CAST({col} AS FLOAT)) OVER()) / "
                    f"NULLIF(STDDEV(CAST({col} AS FLOAT)) OVER(), 0) AS {col}"
                )
        else:
            select_parts.append(f"    {col}")

    lines.append(",\n".join(select_parts))
    lines.append("FROM filtered")
    lines.append("WHERE")
    null_checks = [f"    {col} IS NOT NULL" for col in keep_cols[:3]]
    lines.append("\n    OR ".join(null_checks) if null_checks else "    1=1")
    lines.append(";")

    return "\n".join(lines)


def generate_yaml_config(
    missing_threshold: float,
    outlier_method: str,
    encoding_method: str,
    normalization_method: str,
) -> str:
    """Generate a shareable YAML config file for reproducing cleaning settings."""
    return f"""# CleanEngine Cleaning Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Share this file to reproduce the same cleaning settings on any dataset.

cleaning:
  missing_values:
    threshold: {missing_threshold}
    fill_numeric: median
    fill_categorical: mode

  duplicates:
    keep: first

  outliers:
    method: {outlier_method}
    threshold: 1.5

  encoding:
    categorical_method: {encoding_method}

  normalization:
    method: {normalization_method}

analysis:
  enable: true

output:
  folder_prefix: "Cleans-"
"""

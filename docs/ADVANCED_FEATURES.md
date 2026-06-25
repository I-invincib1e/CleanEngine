# 🔬 CleanEngine — Advanced Features Guide

## Overview

CleanEngine v0.2.0 extends the core cleaning pipeline with four production-grade capabilities that transform it from a one-time cleaning helper into a **data quality platform**:

1. **Domain Templates** — pre-tuned cleaning configs for 6 industry verticals
2. **Cleaning Recipe Export** — auto-generate Python scripts, SQL queries, and YAML configs
3. **Data Drift Monitor** — statistical comparison of two dataset versions
4. **Anomaly Explorer** — Isolation Forest detection with plain-English explanations

---

## 1. 🗂️ Domain Templates

### What it does
Selecting a domain template in the GUI sidebar automatically populates all cleaning parameters with industry-appropriate defaults and surfaces column-level hints and validation rules for common field names in that domain.

### Available Templates

#### 🔧 General Purpose
```yaml
missing_threshold: 0.50
outlier_method: iqr
encoding_method: label
normalization_method: minmax
```
Balanced defaults suitable for any tabular dataset.

#### 🏥 Healthcare / Clinical
```yaml
missing_threshold: 0.20   # Clinical data must be nearly complete
outlier_method: zscore    # Extreme vitals can be real events — IQR too aggressive
encoding_method: label
normalization_method: standard  # Medical measurements are normally distributed
```
**Column hints**: `age` (0–130), `weight` (0.5–500 kg), `bmi` (10–80), `blood_pressure`, `diagnosis`  
**Validation rules**: age 0–130, BMI 10–80  
**Tips**: Check for PII (names, SSNs, DOBs) before sharing output. Watch for ICD-10 code formatting.

#### 💰 Finance / Transactions
```yaml
missing_threshold: 0.10   # Financial records should be complete
outlier_method: iqr       # Detects extreme transaction amounts (possible fraud)
encoding_method: label
normalization_method: standard  # Fat-tailed monetary distributions
```
**Column hints**: `amount` (non-negative), `transaction_id` (unique), `currency` (ISO 3-letter)  
**Tips**: Deduplicate transaction IDs. Verify no negative amounts where unexpected.

#### 🛒 E-commerce / Retail
```yaml
missing_threshold: 0.30
outlier_method: iqr
encoding_method: onehot   # Product categories work well with one-hot
normalization_method: minmax  # Maps prices/quantities to [0,1] for ML
```
**Column hints**: `price` (≥0), `quantity` (positive integer), `rating` (1–5), `sku` (no whitespace)  
**Tips**: Deduplicate SKUs — same item often appears with slight name variations.

#### 📋 Survey / Research
```yaml
missing_threshold: 0.40   # Respondents often skip optional questions
outlier_method: iqr
encoding_method: label    # Preserves Likert scale order
normalization_method: minmax
```
**Tips**: Watch for straight-liners (same answer every time). Flag free-text for manual review.

#### 📡 IoT / Sensor Data
```yaml
missing_threshold: 0.15   # Gaps indicate device failures
outlier_method: zscore    # Sensor readings are normally distributed
encoding_method: label
normalization_method: standard  # Preserves physical meaning of deviations
```
**Column hints**: `timestamp` (monotonically increasing), `humidity` (0–100%), `device_id`  
**Tips**: Consider rolling median imputation for short sensor gaps instead of global median.

### Using templates programmatically
```python
from dataset_cleaner.interfaces.domain_templates import get_template

tmpl = get_template('healthcare')
print(tmpl.missing_threshold)   # 0.2
print(tmpl.tips)                # list of domain-specific tips
print(tmpl.column_hints)        # dict of column name → hint
```

---

## 2. 🧾 Cleaning Recipe Export

### What it does
After any cleaning session in the GUI, CleanEngine auto-generates production-ready code representing the exact steps that were applied to your data — so you can reproduce the same transformation on new data without the GUI.

### Python Script

Generated from `recipe_exporter.generate_python_recipe(report, filename)`.

The script:
- Loads data with automatic encoding detection (CSV) or format sniffing
- Drops high-missing columns (same ones that were dropped in your session)
- Imputes missing values (median for numeric, mode for categorical)
- Removes duplicates
- Removes outliers column-by-column using the same method (IQR or Z-score)
- Encodes categorical variables (label or one-hot)
- Normalizes numeric columns (Min-Max or Standard)

**Usage**:
```bash
python clean_my_dataset.py
# Loaded 50,000 rows × 22 columns
# Cleaned: 47,312 rows × 19 columns
# Saved to cleaned_my_dataset.csv
```

To apply to a different file, just change `INPUT_FILE` at the top of the script.

### SQL Query

Generated from `recipe_exporter.generate_sql_recipe(report, table_name)`.

The query:
- Uses `CREATE TABLE cleaned_X AS` pattern
- Deduplicates via `SELECT DISTINCT`
- Computes IQR bounds in a `WITH` CTE and filters outliers
- Applies `DENSE_RANK()` for label encoding
- Applies window-function normalization (Min-Max or Z-score)

Compatible with PostgreSQL, Snowflake, BigQuery, and any ANSI SQL-compliant database.

### YAML Config

Generated from `recipe_exporter.generate_yaml_config(...)`.

Share with your team so everyone runs the same settings:
```yaml
# cleanengine_mydata.yaml
cleaning:
  missing_values:
    threshold: 0.3
  outliers:
    method: iqr
    threshold: 1.5
  encoding:
    categorical_method: label
  normalization:
    method: minmax
analysis:
  enable: true
```

Load it in a future session or pass it to the CLI:
```bash
cleanengine clean new_data.csv --config cleanengine_mydata.yaml
```

---

## 3. 📡 Data Drift Monitor

### What it does
Compares two snapshots of the same dataset and produces a column-level statistical drift report — without any labelling or ground truth required.

### How drift is detected

| Column type | Test used | Drifted if |
|-------------|-----------|------------|
| Numeric | Kolmogorov-Smirnov (2-sample) | p-value < 0.05 |
| Categorical | Chi-squared goodness-of-fit | p-value < 0.05 |
| Missing values | Delta percentage | Δ > 5 pp |
| Schema | Set comparison | Any added/removed columns |

### Drift severity scale
| Severity | Columns drifted |
|----------|----------------|
| 🟢 None | 0% |
| 🟡 Low | < 20% |
| 🟠 Medium | 20–50% |
| 🔴 High | > 50% |

### Report structure
```json
{
  "reference_shape": {"rows": 50000, "cols": 22},
  "new_shape": {"rows": 52100, "cols": 22},
  "schema_changes": {
    "added_columns": ["new_feature"],
    "removed_columns": [],
    "common_columns": [...]
  },
  "missing_drift": {
    "age": {"reference_pct": 2.1, "new_pct": 8.4, "delta_pct": 6.3, "significant_change": true}
  },
  "distribution_drift": {
    "salary": {
      "type": "numeric",
      "statistical_test": {"test": "ks", "statistic": 0.142, "p_value": 0.001, "drifted": true},
      "reference_stats": {"mean": 75200, "std": 18400, ...},
      "new_stats": {"mean": 82100, "std": 21300, ...},
      "mean_delta": 6900
    }
  },
  "summary": {
    "columns_with_drift": 4,
    "drift_rate_pct": 18.2,
    "severity": "low"
  }
}
```

### Programmatic usage
```python
import pandas as pd
from dataset_cleaner.interfaces.drift_monitor import compare_datasets

df_jan = pd.read_csv('sales_jan.csv')
df_feb = pd.read_csv('sales_feb.csv')

report = compare_datasets(df_jan, df_feb)

# Check severity
print(report['summary']['severity'])          # "low"
print(report['summary']['drifted_column_names'])  # ['price', 'region']

# Get per-column details
for col, data in report['distribution_drift'].items():
    if data['statistical_test']['drifted']:
        print(f"{col}: p={data['statistical_test']['p_value']}")
```

### MLOps integration example
```python
import json
from dataset_cleaner.interfaces.drift_monitor import compare_datasets

report = compare_datasets(df_training, df_production)

if report['summary']['severity'] in ('medium', 'high'):
    # Trigger model retraining
    send_alert(f"Data drift detected: {report['summary']['drift_rate_pct']:.1f}% of columns drifted")
    trigger_retraining_pipeline()
```

---

## 4. 🔍 Anomaly Explorer

### What it does
Detects anomalous rows using **Isolation Forest** and explains each one in plain English by identifying which features deviate most from the dataset's normal distribution.

### Detection method
Isolation Forest works by randomly partitioning the feature space. Anomalous points are easier to isolate (require fewer splits), so they get lower anomaly scores. This method:
- Requires no labelled data
- Works on high-dimensional data
- Is robust to varying scales (internally normalized)
- Has a configurable `contamination` parameter (expected anomaly fraction)

### Explanation method
For each detected anomaly, CleanEngine checks every numeric column and flags values that:
- Have absolute z-score ≥ 2 (unusual) or ≥ 3 (extreme)
- Fall outside IQR fences (Q1 − 1.5×IQR or Q3 + 1.5×IQR)

Each flagged feature gets:
- Its raw value and percentile rank
- Direction (unusually high or low)
- Z-score
- Whether it breaches the IQR fence

### Example output
```
Row 1842 is anomalous because:
  `salary`         = 2,400,000  (unusually high, 99.9th pct); z=4.7; above IQR upper fence (385,000)
  `age`            = 19         (unusually low, 2.3rd pct); z=-2.9
  `years_at_company` = 22       (unusually high, 98th pct); z=3.1; above IQR upper fence (18)
  ...and 1 more deviating feature
```

### Report structure
```json
{
  "summary": {
    "total_rows": 50000,
    "total_anomalies": 2500,
    "anomaly_rate_pct": 5.0,
    "columns_analyzed": ["age", "salary", "years_at_company", ...]
  },
  "anomaly_rows": [
    {
      "row_index": 1842,
      "anomaly_score": -0.312,
      "summary": "Row 1842 is anomalous because: `salary` = 2400000 ...",
      "feature_deviations": [
        {
          "column": "salary",
          "value": 2400000,
          "z_score": 4.7,
          "percentile": 99.9,
          "direction": "high",
          "above_iqr_upper": true,
          "explanation": "Value 2400000 is in the 99.9th percentile; z-score = 4.7 (extreme, >3σ); Above IQR upper fence (385000)."
        }
      ]
    }
  ]
}
```

### Programmatic usage
```python
import pandas as pd
from dataset_cleaner.interfaces.anomaly_explainer import explain_anomalies

df = pd.read_csv('transactions.csv')
results = explain_anomalies(df, contamination=0.05, max_anomalies=100)

print(f"Found {results['summary']['total_anomalies']} anomalies "
      f"({results['summary']['anomaly_rate_pct']:.1f}%)")

for row in results['anomaly_rows'][:10]:
    print(f"\n{row['summary']}")
    for dev in row['feature_deviations'][:3]:
        print(f"  {dev['column']}: {dev['value']} (z={dev['z_score']}, {dev['percentile']}th pct)")
```

---

## Previous Advanced Features (v0.1.0)

### Statistical Analysis
- Descriptive statistics, skewness, kurtosis
- Shapiro-Wilk normality tests
- Pearson/Spearman/Kendall correlation matrices
- Strong correlation detection (> 0.7)

### Feature Importance
- Mutual Information scoring
- Feature ranking by predictive power

### Clustering Analysis
- K-Means with Elbow method for optimal k
- DBSCAN and Hierarchical alternatives
- Cluster profiling (size and percentage)

### Data Quality Scoring
- Completeness, uniqueness, consistency dimensions
- Composite score 0–100

### Visualizations
- Correlation heatmap, distribution plots
- Feature importance chart, quality dashboard (4-panel)

---

## Integration with Data Science Workflows

### Jupyter Notebooks
```python
from pathlib import Path
from dataset_cleaner import DatasetCleaner
from dataset_cleaner.interfaces.anomaly_explainer import explain_anomalies

cleaner = DatasetCleaner()
df = cleaner.clean_dataset('data.csv')
analysis = cleaner.perform_advanced_analysis(Path('output'), 'data')

# Anomaly report inline
results = explain_anomalies(df)
for row in results['anomaly_rows'][:5]:
    print(row['summary'])
```

### MLOps / CI-CD Pipelines
```python
from dataset_cleaner.interfaces.drift_monitor import compare_datasets

# Run in your data validation step
report = compare_datasets(df_training, df_new_batch)

assert report['summary']['severity'] not in ('medium', 'high'), \
    f"Data drift too high — {report['summary']['drift_rate_pct']}% of columns drifted"
```

### dbt / SQL Pipelines
Export the SQL cleaning recipe from the GUI and paste it directly into your dbt model or SQL pipeline — it's standard ANSI SQL with no CleanEngine runtime dependency.

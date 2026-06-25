# CleanEngine Docs

CleanEngine is a modular data quality platform for cleaning, profiling, drift monitoring, and anomaly explanation — with a YAML-driven rule engine and a multi-tab Streamlit GUI.

## Feature Overview

| Module | What it does |
|--------|-------------|
| **Core Cleaning** | Missing values, duplicates, outliers, encoding, normalization |
| **Domain Templates** | Pre-tuned cleaning configs for Healthcare, Finance, E-commerce, IoT, Survey |
| **Recipe Export** | Auto-generate Python scripts, SQL queries, and YAML configs from cleaning sessions |
| **Drift Monitor** | Statistical comparison of two dataset versions (KS test, Chi-squared, schema diff) |
| **Anomaly Explorer** | Isolation Forest detection + plain-English explanation per anomalous row |
| **Advanced Analysis** | Stats, correlations, feature importance, clustering, quality scoring |
| **Rule Engine** | YAML-defined validation (pre/post clean) |
| **File I/O** | CSV, Excel, JSON, Parquet (pyarrow/fastparquet) |
| **Interfaces** | CLI (Typer + Rich), Streamlit GUI, Folder Watcher |

## Quick Start

```bash
# Install
pip install cleanengine

# Launch the web GUI
cleanengine gui

# Or run directly with Streamlit
PYTHONPATH=src streamlit run src/dataset_cleaner/interfaces/streamlit_app.py --server.port 5000
```

## Web Interface

The Streamlit GUI has three tabs:

### 🧹 Clean & Export
1. Upload CSV, Excel, JSON, or Parquet
2. Select a **Domain Template** (sidebar) for pre-tuned settings
3. Preview data quality (missing heatmap, outlier boxplots)
4. Click **Clean Dataset**
5. Export your cleaning steps as **Python script / SQL / YAML**

### 📡 Drift Monitor
1. Upload a **reference** (baseline) dataset
2. Upload a **new** dataset
3. Get a per-column drift report with severity rating

### 🔍 Anomaly Explorer
1. Upload a dataset
2. Set contamination rate (expected % of anomalies)
3. Get plain-English explanations for each anomalous row

## CLI Quick Reference

```bash
cleanengine clean data.csv                  # Full cleaning pipeline
cleanengine analyze data.csv                # Analysis only
cleanengine validate-data data.csv          # YAML rule validation
cleanengine anomalies data.csv              # Anomaly detection
cleanengine correlations data.csv           # Correlation analysis
cleanengine clusters data.csv               # Clustering
cleanengine quality data.csv                # Quality score
cleanengine samples                         # Generate sample data
cleanengine gui                             # Launch web interface
```

## Python API Quick Reference

```python
from dataset_cleaner import DatasetCleaner
from dataset_cleaner.interfaces.drift_monitor import compare_datasets
from dataset_cleaner.interfaces.anomaly_explainer import explain_anomalies
from dataset_cleaner.interfaces.recipe_exporter import generate_python_recipe
from dataset_cleaner.interfaces.domain_templates import get_template

# Clean
cleaner = DatasetCleaner()
df = cleaner.clean_dataset('data.csv', missing_threshold=0.3)

# Drift
report = compare_datasets(df_old, df_new)

# Anomalies
results = explain_anomalies(df, contamination=0.05)

# Recipe
code = generate_python_recipe(cleaner.report, 'data.csv')

# Domain template defaults
tmpl = get_template('healthcare')
```

## Documentation Files

| File | Contents |
|------|----------|
| `README.md` (this file) | Overview, quick start, API reference |
| `ADVANCED_FEATURES.md` | Deep-dive into domain templates, recipe export, drift monitor, anomaly explorer |
| `CONTRIBUTING.md` | How to contribute |

For full usage, configuration examples, and CLI reference, see the [top-level README](../README.md).

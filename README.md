# 🧹 CleanEngine

[![GitHub stars](https://img.shields.io/github/stars/I-invincib1e/CleanEngine?style=social)](https://github.com/I-invincib1e/CleanEngine)
[![GitHub forks](https://img.shields.io/github/forks/I-invincib1e/CleanEngine?style=social)](https://github.com/I-invincib1e/CleanEngine)
[![GitHub issues](https://img.shields.io/github/issues/I-invincib1e/CleanEngine)](https://github.com/I-invincib1e/CleanEngine/issues)
[![PyPI version](https://badge.fury.io/py/cleanengine.svg)](https://badge.fury.io/py/cleanengine)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#)
[![PyPI](https://img.shields.io/pypi/v/cleanengine)](https://pypi.org/project/cleanengine/)
[![Python versions](https://img.shields.io/pypi/pyversions/cleanengine)](https://pypi.org/project/cleanengine/)


> **🚀 The Ultimate Data Cleaning & Analysis Toolkit**  
> Transform messy datasets into production-ready data with intelligent cleaning, drift detection, anomaly explanation, and auto-generated cleaning code.

CleanEngine is a powerful toolkit that handles the full data quality lifecycle: clean raw data, monitor it for drift over time, detect and explain anomalies, and export your cleaning logic as reusable Python scripts, SQL queries, or YAML configs — all from a single web interface or CLI.

---

## ✨ What's New in v0.2.0

| Feature | Description |
|---------|-------------|
| 🗂️ **Domain Templates** | Pre-tuned cleaning configs for Healthcare, Finance, E-commerce, IoT, and Survey data |
| 🧾 **Cleaning Recipe Export** | Auto-generate Python scripts, SQL queries, and YAML configs from your cleaning session |
| 📡 **Data Drift Monitor** | Upload two dataset versions and detect schema changes, distribution drift, and missing-value shifts per column |
| 🔍 **Anomaly Explorer** | Isolation Forest detection + plain-English explanation of *why* each row is anomalous |

---

## 📊 Comparison with Other Tools

| Feature | **CleanEngine** 🧹 | pandas-profiling | Sweetviz | Great Expectations |
|---------|-------------------|------------------|-----------|-------------------|
| **Data Cleaning Pipeline** | ✅ Complete | ❌ No | ❌ No | ⚠️ Limited |
| **Domain Templates** | ✅ 5 domains | ❌ No | ❌ No | ❌ No |
| **Recipe Export (Python/SQL/YAML)** | ✅ **Unique** | ❌ No | ❌ No | ❌ No |
| **Data Drift Monitoring** | ✅ Per-column | ❌ No | ❌ No | ⚠️ Basic |
| **Anomaly Explanation** | ✅ Plain-English | ❌ No | ❌ No | ❌ No |
| **Profiling & Stats** | ✅ Advanced | ✅ Yes | ✅ Yes | ⚠️ Minimal |
| **Correlation Analysis** | ✅ Multi-Method | ✅ Yes | ✅ Yes | ❌ No |
| **Feature Importance** | ✅ ML-Powered | ❌ No | ❌ No | ❌ No |
| **Clustering** | ✅ 3 Algorithms | ❌ No | ❌ No | ❌ No |
| **Rule Engine (YAML)** | ✅ Yes | ❌ No | ❌ No | ✅ Yes |
| **Web Interface** | ✅ Streamlit | CLI/Notebook | Notebook | CLI/Notebook |
| **Folder Watcher Automation** | ✅ Yes | ❌ No | ❌ No | ✅ Yes |

---

## 🚀 Installation

### Using pip (Recommended)
```bash
pip install cleanengine
```

### From source
```bash
git clone https://github.com/I-invincib1e/CleanEngine.git
cd CleanEngine
pip install -e .
```

### Launch the web interface
```bash
PYTHONPATH=src streamlit run src/dataset_cleaner/interfaces/streamlit_app.py --server.port 5000
```

---

## 🎯 Quick Start

### Launch the web GUI
```bash
cleanengine gui
```

### Clean a CSV file from the CLI
```bash
cleanengine clean data.csv
```

### Analyze data without cleaning
```bash
cleanengine analyze data.xlsx
```

### Generate sample data to test with
```bash
cleanengine samples
```

---

## 🖥️ Web Interface — 3 Tabs

### 🧹 Tab 1: Clean & Export

Upload any CSV, Excel, JSON, or Parquet file and:

1. **Pick a Domain Template** from the sidebar — the cleaning settings auto-populate for your data type
2. **Preview data quality** — missing values heatmap and outlier boxplots
3. **Click Clean Dataset** — runs the full pipeline
4. **Get Before/After metrics** and advanced analysis insights
5. **Export your cleaning recipe** as:
   - **Python script** — standalone, production-ready, runs anywhere
   - **SQL query** — `CREATE TABLE AS` with deduplication, outlier CTEs, encoding/normalization
   - **YAML config** — share with your team to reproduce identical settings

### 📡 Tab 2: Drift Monitor

Upload two versions of the same dataset (e.g. last month vs this month):

- **Schema diff** — columns added or removed
- **Per-column statistical tests** — KS test (numeric), Chi-squared (categorical), p < 0.05 = drifted
- **Missing value shifts** — flagged when delta exceeds 5 percentage points
- **New categories** — values that appeared for the first time in the new dataset
- **Severity badge** — None / Low / Medium / High overall drift rating
- **Download** the full drift report as JSON

### 🔍 Tab 3: Anomaly Explorer

Upload any dataset to find and understand anomalous rows:

- **Isolation Forest** detection with configurable contamination rate
- **Per-row explanation** — which features deviate, by how much, and in which direction
- Plain-English summary: *"Row 47 is anomalous because: `salary` = 2,400,000 (unusually high, 99th pct); `age` = 22 (unusually low, 4th pct)"*
- **Column contribution chart** — which columns drive the most anomalies
- **Download** anomalous rows as CSV or the full report as JSON

---

## 🗂️ Domain Templates

| Template | Missing Tolerance | Outlier Method | Encoding | Normalization |
|----------|------------------|----------------|----------|---------------|
| 🔧 General Purpose | 50% | IQR | Label | Min-Max |
| 🏥 Healthcare | 20% | Z-Score | Label | Standard |
| 💰 Finance | 10% | IQR | Label | Standard |
| 🛒 E-commerce | 30% | IQR | One-Hot | Min-Max |
| 📋 Survey | 40% | IQR | Label | Min-Max |
| 📡 IoT / Sensor | 15% | Z-Score | Label | Standard |

Each template includes:
- **Pre-tuned defaults** for cleaning parameters
- **Column hints** — guidance for common field names in that domain
- **Validation rules** — domain-appropriate range checks
- **Tips** — best practices shown in the sidebar

---

## 🧾 Cleaning Recipe Export

After cleaning any dataset, CleanEngine generates production-ready code from your exact cleaning steps:

### Python Script
```python
# Auto-generated by CleanEngine
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop high-missing columns
    if 'notes' in df.columns:
        df = df.drop(columns=['notes'])

    # Impute missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Remove duplicates
    df = df.drop_duplicates(keep='first')

    # Remove outliers (IQR)
    Q1 = df['salary'].quantile(0.25)
    Q3 = df['salary'].quantile(0.75)
    df = df[(df['salary'] >= Q1 - 1.5*(Q3-Q1)) & (df['salary'] <= Q3 + 1.5*(Q3-Q1))]

    # Encode categorical variables
    le = LabelEncoder()
    df['department'] = le.fit_transform(df['department'].astype(str))

    # Normalize
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df
```

### SQL Query
Auto-generates a `CREATE TABLE AS` statement with deduplication CTEs, IQR outlier bounds, encoding, and normalization — ready for PostgreSQL, Snowflake, BigQuery, or any ANSI SQL database.

### YAML Config
```yaml
# Share with your team to reproduce identical cleaning settings
cleaning:
  missing_values:
    threshold: 0.3
  outliers:
    method: iqr
  encoding:
    categorical_method: label
  normalization:
    method: minmax
```

---

## 📋 CLI Commands

### Core Commands
| Command | Description | Example |
|---------|-------------|---------|
| `clean` | Clean a dataset with full pipeline | `cleanengine clean data.csv --output ./cleaned/` |
| `analyze` | Analyze data without cleaning | `cleanengine analyze data.csv` |
| `validate-data` | Validate data with YAML rules | `cleanengine validate-data data.csv` |
| `profile` | Generate data profile report | `cleanengine profile data.csv` |
| `clean-only` | Clean without running analysis | `cleanengine clean-only data.csv` |
| `samples` | Create sample datasets | `cleanengine samples --count 5` |
| `gui` | Launch Streamlit web interface | `cleanengine gui --port 5000` |
| `info` | Show CleanEngine version info | `cleanengine info` |

### Advanced Analysis Commands
| Command | Description | Example |
|---------|-------------|---------|
| `correlations` | Analyze variable correlations | `cleanengine correlations data.csv --method pearson` |
| `features` | Analyze feature importance | `cleanengine features data.csv` |
| `clusters` | Discover data clusters | `cleanengine clusters data.csv --method kmeans` |
| `anomalies` | Detect anomalies/outliers | `cleanengine anomalies data.csv --contamination 0.1` |
| `quality` | Assess data quality score | `cleanengine quality data.csv` |
| `statistics` | Perform statistical analysis | `cleanengine statistics data.csv` |

---

## 📁 Supported File Formats

- **CSV** — with automatic encoding detection (UTF-8, Latin-1, ISO-8859-1, CP1252)
- **Excel** — `.xlsx` and `.xls`
- **JSON** — standard and records format
- **Parquet** — via PyArrow and FastParquet engines

---

## 📊 Output Structure

After processing, CleanEngine creates a `Cleans-<dataset_name>/` folder:

```
Cleans-data/
├── cleaned_data.csv                   # Cleaned dataset
├── data_cleaning_report.json          # Detailed cleaning summary
├── data_cleaning_summary.txt          # Human-readable summary
├── data_analysis_report.md            # Comprehensive analysis report
├── data_analysis_results.json         # Structured analysis data
└── visualizations/                    # Generated charts
    ├── correlation_heatmap.png
    ├── distributions.png
    ├── feature_importance.png
    └── quality_dashboard.png
```

---

## ⚙️ Configuration

### YAML Configuration File

Create a `config.yaml` in your working directory or export one from the GUI:

```yaml
cleaning:
  missing_values:
    threshold: 0.5          # Drop columns with > 50% missing
    fill_numeric: median    # median | mean
    fill_categorical: mode  # mode | constant

  duplicates:
    keep: first             # first | last | false

  outliers:
    method: iqr             # iqr | zscore
    threshold: 1.5          # IQR multiplier

  encoding:
    categorical_method: label   # label | onehot

  normalization:
    method: minmax          # minmax | standard

analysis:
  enable: true

output:
  folder_prefix: "Cleans-"
```

---

## 🐍 Python API

```python
from dataset_cleaner import DatasetCleaner, DataAnalyzer

# Clean a dataset
cleaner = DatasetCleaner()
cleaned_df = cleaner.clean_dataset(
    'data.csv',
    missing_threshold=0.3,
    outlier_method='iqr',
    encoding_method='label',
    normalization_method='minmax',
)

# Run advanced analysis
output_folder = cleaner.create_output_folder('data.csv')
analysis = cleaner.perform_advanced_analysis(output_folder, 'data')

# Access insights
insights = analysis['analysis_results']['insights']
quality_score = analysis['analysis_results']['data_quality']['overall_quality_score']
```

### Drift Monitoring (programmatic)
```python
import pandas as pd
from dataset_cleaner.interfaces.drift_monitor import compare_datasets

df_old = pd.read_csv('data_jan.csv')
df_new = pd.read_csv('data_feb.csv')

report = compare_datasets(df_old, df_new)
print(f"Drift severity: {report['summary']['severity']}")
print(f"Drifted columns: {report['summary']['drifted_column_names']}")
```

### Anomaly Explanation (programmatic)
```python
import pandas as pd
from dataset_cleaner.interfaces.anomaly_explainer import explain_anomalies

df = pd.read_csv('data.csv')
results = explain_anomalies(df, contamination=0.05)

for row in results['anomaly_rows'][:5]:
    print(row['summary'])
    # → "Row 47 is anomalous because: `salary` = 2400000 (unusually high, 99th pct); ..."
```

---

## 📈 Performance

| Dataset Size | Cleaning | Analysis | Drift Check |
|-------------|---------|---------|-------------|
| < 1 MB | < 1 sec | < 5 sec | < 2 sec |
| 1 – 100 MB | 1–30 sec | 30–120 sec | 5–30 sec |
| 100 MB – 1 GB | 30 sec – 5 min | 2–10 min | 1–3 min |
| > 1 GB | Configurable chunking | Configurable | Configurable |

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Setting up a development environment
- Code style and standards  
- Testing and quality assurance
- Pull request process

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **pandas** — data manipulation backbone
- **scikit-learn** — ML algorithms (Isolation Forest, clustering, encoding)
- **scipy** — statistical tests (KS, Chi-squared)
- **Streamlit** — web interface
- **Typer & Rich** — beautiful CLI
- **Plotly** — interactive visualizations

---

<div align="center">

**Made with ❤️ for data scientists, data engineers, and analysts**

[GitHub](https://github.com/I-invincib1e/CleanEngine) •
[PyPI](https://pypi.org/project/cleanengine/) •
[Documentation](https://github.com/I-invincib1e/CleanEngine/tree/main/docs)

</div>

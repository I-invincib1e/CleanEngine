# 🧹 CleanEngine

[![GitHub stars](https://img.shields.io/github/stars/I-invincib1e/CleanEngine?style=social)](https://github.com/I-invincib1e/CleanEngine)
[![GitHub forks](https://img.shields.io/github/forks/I-invincib1e/CleanEngine?style=social)](https://github.com/I-invincib1e/CleanEngine)
[![GitHub issues](https://img.shields.io/github/issues/I-invincib1e/CleanEngine)](https://github.com/I-invincib1e/CleanEngine/issues)
[![PyPI version](https://badge.fury.io/py/cleanengine.svg)](https://badge.fury.io/py/cleanengine)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#)
[![Downloads](https://img.shields.io/pypi/dm/cleanengine)](https://pypi.org/project/cleanengine/)

> **🚀 The Ultimate Data Cleaning & Analysis CLI Tool**  
> Transform messy datasets into clean, insights-rich data with intelligent cleaning and advanced ML analysis.

CleanEngine is a powerful command-line toolkit that handles missing values, removes duplicates, detects outliers, and provides comprehensive statistical analysis using machine learning techniques.

![CleanEngine Demo](https://img.shields.io/badge/demo-available-blue)

### 📊 **Comparison with Other Tools**

| Feature | **CleanEngine** 🧹 | pandas-profiling | Sweetviz | Great Expectations |
|---------|-------------------|------------------|-----------|-------------------|
| **Data Cleaning** | ✅ **Complete Pipeline** | ❌ No | ❌ No | ⚠️ Limited |
| **Profiling & Stats** | ✅ **Advanced Analytics** | ✅ Yes | ✅ Yes | ⚠️ Minimal |
| **Correlation Analysis** | ✅ **Multi-Method** | ✅ Yes | ✅ Yes | ❌ No |
| **Feature Importance** | ✅ **ML-Powered** | ❌ No | ❌ No | ❌ No |
| **Clustering & Patterns** | ✅ **3 Algorithms** | ❌ No | ❌ No | ❌ No |
| **Anomaly Detection** | ✅ **2 Methods** | ❌ No | ❌ No | ❌ No |
| **Rule Engine** | ✅ **YAML-Driven** | ❌ No | ❌ No | ✅ Yes |
| **Interfaces** | ✅ **CLI + GUI + Watcher** | CLI/Notebook | Notebook | CLI/Notebook |
| **Automation** | ✅ **Folder Watcher** | ❌ No | ❌ No | ✅ Yes |

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

### Verify Installation
```bash
cleanengine --help
```

---

## 🎯 Quick Start

### Clean a CSV file
```bash
cleanengine clean data.csv
```

### Analyze data without cleaning
```bash
cleanengine analyze data.xlsx
```

### Generate sample data to test
```bash
cleanengine samples
```

### Launch web interface
```bash
cleanengine gui
```

---

## 📋 CLI Commands

### Core Commands
| Command | Flags | Description | Example |
|---------|-------|-------------|---------|
| `clean` | `--output, -o`, `--verbose, -v`, `--force` | Clean a dataset with full pipeline | `cleanengine clean data.csv --output ./cleaned/ --verbose` |
| `analyze` | `--output, -o`, `--verbose, -v` | Analyze data without cleaning | `cleanengine analyze data.csv --output ./analysis/ --verbose` |
| `validate-data` | `--verbose, -v` | Validate data with rules | `cleanengine validate-data data.csv --verbose` |
| `profile` | `--output, -o`, `--verbose, -v` | Generate data profile report | `cleanengine profile data.csv --output ./profile/ --verbose` |
| `clean-only` | `--output, -o`, `--verbose, -v` | Clean without analysis | `cleanengine clean-only data.csv --output ./cleaned/ --verbose` |
| `samples` | `--output, -o`, `--count, -n`, `--verbose, -v` | Create sample datasets | `cleanengine samples --output ./samples/ --count 5 --verbose` |
| `test` | `--verbose, -v`, `--coverage` | Run test suite | `cleanengine test --verbose --coverage` |
| `gui` | `--port, -p`, `--host, -h` | Launch Streamlit web interface | `cleanengine gui --port 8501 --host localhost` |
| `info` | None | Show CleanEngine information | `cleanengine info` |

### Advanced Analysis Commands
| Command | Flags | Description | Example |
|---------|-------|-------------|---------|
| `correlations` | `--method, -m`, `--threshold, -t`, `--output, -o`, `--verbose, -v` | Analyze variable correlations | `cleanengine correlations data.csv --method pearson --threshold 0.7 --verbose` |
| `features` | `--output, -o`, `--verbose, -v` | Analyze feature importance | `cleanengine features data.csv --output ./features/ --verbose` |
| `clusters` | `--method, -m`, `--output, -o`, `--verbose, -v` | Discover data clusters | `cleanengine clusters data.csv --method kmeans --output ./clusters/ --verbose` |
| `anomalies` | `--method, -m`, `--contamination, -c`, `--output, -o`, `--verbose, -v` | Detect anomalies/outliers | `cleanengine anomalies data.csv --method isolation_forest --contamination 0.1 --verbose` |
| `quality` | `--output, -o`, `--verbose, -v` | Assess data quality | `cleanengine quality data.csv --output ./quality/ --verbose` |
| `statistics` | `--output, -o`, `--verbose, -v` | Perform statistical analysis | `cleanengine statistics data.csv --output ./stats/ --verbose` |

---

## 📁 Supported File Formats

- **CSV**: Comma-separated values
- **Excel**: .xlsx and .xls files
- **JSON**: JavaScript Object Notation
- **Parquet**: Columnar storage format

---

## 📊 Output Structure

After processing, CleanEngine creates a `Cleans-<dataset_name>/` folder with:

```
Cleans-data/
├── cleaned_data.csv          # Your cleaned dataset
├── cleaning_report.json      # Detailed cleaning summary
├── analysis_report.json      # Comprehensive analysis results
├── visualizations/           # Generated charts and plots
└── logs/                     # Processing logs
```

---

## ⚙️ Configuration

### Custom Configuration File
Create a `config.yaml` file in your working directory:

```yaml
cleaning:
  missing_values:
    strategy: "auto"  # auto, mean, median, mode, drop
  outliers:
    method: "iqr"     # iqr, zscore, custom
  encoding:
    categorical: true
    normalize: true

analysis:
  correlation:
    method: "pearson"  # pearson, spearman, kendall
  clustering:
    method: "kmeans"   # kmeans, dbscan, hierarchical
```

---

## 🎨 CLI Features

- **Rich Terminal Output**: Beautiful tables, progress bars, and colors
- **Interactive Help**: `cleanengine --help` and `cleanengine <command> --help`
- **Auto-completion**: Tab completion for commands and file paths
- **Progress Tracking**: Real-time progress bars for long operations
- **Error Handling**: Clear error messages with suggestions

---

## 📈 Performance

- **Small Datasets** (< 1MB): < 1 second
- **Medium Datasets** (1-100MB): 1-30 seconds
- **Large Datasets** (100MB-1GB): 30 seconds - 5 minutes
- **Very Large Datasets** (> 1GB): Configurable chunking

---

## 🔧 Advanced Usage

### Batch Processing Multiple Files
```bash
# Process all CSV files in current directory
for file in *.csv; do cleanengine clean "$file"; done
```

### Custom Output Directory
```bash
cleanengine clean data.csv --output-dir ./my-clean-data/
```

### Configuration File
```bash
cleanengine clean data.csv --config ./my-config.yaml
```

### Verbose Output
```bash
cleanengine clean data.csv --verbose
```

---

## 🐍 Python API

For programmatic use:

```python
from cleanengine import DatasetCleaner

# Initialize cleaner
cleaner = DatasetCleaner()

# Clean dataset
cleaned_df = cleaner.clean_dataset('data.csv')

# Get analysis results
analysis_results = cleaner.analyze_dataset('data.csv')
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up a development environment
- Code style and standards
- Testing and quality assurance
- Pull request process

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **pandas** for data manipulation
- **scikit-learn** for machine learning algorithms
- **Typer & Rich** for beautiful CLI interfaces
- **Streamlit** for web interface

---

<div align="center">

**Made with ❤️ for data scientists and analysts**

[GitHub](https://github.com/I-invincib1e/CleanEngine) •
[PyPI](https://pypi.org/project/cleanengine/) •
[Documentation](https://github.com/I-invincib1e/CleanEngine#readme)

</div>

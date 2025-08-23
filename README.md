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
| Command | Description | Example |
|---------|-------------|---------|
| `clean` | Clean a dataset with full pipeline | `cleanengine clean data.csv` |
| `analyze` | Analyze data without cleaning | `cleanengine analyze data.csv` |
| `validate-data` | Validate data with rules | `cleanengine validate-data data.csv` |
| `profile` | Generate data profile report | `cleanengine profile data.csv` |
| `clean-only` | Clean without analysis | `cleanengine clean-only data.csv` |
| `samples` | Create sample datasets | `cleanengine samples` |
| `test` | Run test suite | `cleanengine test` |
| `gui` | Launch Streamlit web interface | `cleanengine gui` |
| `info` | Show CleanEngine information | `cleanengine info` |

### Advanced Analysis Commands
| Command | Description | Example |
|---------|-------------|---------|
| `correlations` | Analyze variable correlations | `cleanengine correlations data.csv --method pearson --threshold 0.7` |
| `features` | Analyze feature importance | `cleanengine features data.csv` |
| `clusters` | Discover data clusters | `cleanengine clusters data.csv --method kmeans` |
| `anomalies` | Detect anomalies/outliers | `cleanengine anomalies data.csv --method isolation_forest` |
| `quality` | Assess data quality | `cleanengine quality data.csv` |
| `statistics` | Perform statistical analysis | `cleanengine statistics data.csv` |

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

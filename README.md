# 🧹 CleanEngine

[![Tests](https://img.shields.io/badge/tests-36%20passing-brightgreen)](./scripts/run_tests.py)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](./LICENSE)
[![CLI](https://img.shields.io/badge/CLI-Rich%20%2B%20Typer-purple)](#)
[![Status](https://img.shields.io/badge/status-production%20ready-success)](#)
[![Version](https://img.shields.io/badge/version-0.1.0-orange)](#)

> **🚀 The Ultimate Data Cleaning & Analysis Toolkit**  
> Transform messy datasets into clean, insights-rich data with CleanEngine's powerful cleaning pipeline, intelligent analysis, and YAML-driven rule engine.

---

## 🌟 What Makes CleanEngine Special?

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

## ✨ Core Features

### 🧼 **Smart Data Cleaning**
- **Missing Values**: Intelligent imputation strategies
- **Duplicate Detection**: Advanced duplicate removal with configurable thresholds
- **Outlier Handling**: IQR, Z-score, and custom outlier detection
- **Encoding**: Automatic categorical encoding and normalization
- **Data Types**: Smart type inference and conversion

### 📊 **Advanced Analytics**
- **Statistical Analysis**: Comprehensive descriptive statistics
- **Correlation Methods**: Pearson, Spearman, and Kendall correlations
- **Feature Importance**: Mutual information and target analysis
- **Distribution Analysis**: Skewness, kurtosis, and normality tests
- **Data Quality Scoring**: Completeness, uniqueness, and consistency metrics

### 🔍 **Machine Learning Insights**
- **Clustering**: K-Means, DBSCAN, and Hierarchical clustering
- **Anomaly Detection**: Isolation Forest and Local Outlier Factor (LOF)
- **Pattern Recognition**: Automatic pattern discovery in data
- **Dimensionality Insights**: Feature correlation and redundancy analysis

### ⚙️ **Rule Engine & Validation**
- **YAML Configuration**: Declarative rule definitions
- **Pre/Post Validation**: Data quality checks before and after cleaning
- **Custom Rules**: Extensible rule system for domain-specific validation
- **Compliance**: Built-in support for data governance requirements

### 🎨 **Multiple Interfaces**
- **Rich CLI**: Beautiful terminal interface with Typer and Rich
- **Streamlit GUI**: Interactive web-based interface
- **Folder Watcher**: Automated processing of new datasets
- **API Ready**: Modular design for integration

---

## 🚀 Quick Start

### 📋 **Prerequisites**
- Python 3.9 or higher
- Git (for cloning)
- pip or conda package manager

### 🔧 **Installation**
```bash
# 1. Clone the repository
git clone https://github.com/I-invincib1e/CleanEngine.git
cd CleanEngine

# 2. Install dependencies
python setup.py

# 3. Verify installation
python main.py -t
```

### 🎯 **First Run**
```bash
# Interactive menu (recommended for beginners)
python main.py

# Quick commands
python main.py -s                  # 🎲 Create sample datasets
python main.py -c sample_mixed.csv # 🧹 Clean a dataset
python main.py -t                  # 🧪 Run tests
python main.py -g                  # 🌐 Launch Streamlit GUI
```

---

## 🚀 Minimal Example

```python
import dataset_cleaner
print(dataset_cleaner.__version__)

from dataset_cleaner.core.cleaner import DatasetCleaner
cleaner = DatasetCleaner()
cleaned_df = cleaner.clean_dataset('data.csv')
cleaner.create_output_folder('data.csv')
cleaner.save_results(cleaned_df, cleaner.report, 'Cleans-data')
```

---

## 🎮 CLI Commands (Typer + Rich)

| Command | Short | Description | Example |
|---------|-------|-------------|---------|
| `clean` | `c` | Clean a dataset | `python main.py c data.csv` |
| `samples` | `s` | Generate sample data | `python main.py s` |
| `tests` | `t` | Run test suite | `python main.py t` |
| `gui` | `g` | Launch Streamlit | `python main.py g` |

### 🎨 **CLI Features**
- **Rich Output**: Colorful tables, progress bars, and status indicators
- **Interactive Menus**: User-friendly selection interfaces
- **Short Flags**: Quick access with single-letter commands
- **Help System**: Comprehensive command documentation
- **Auto-completion**: Intelligent command suggestions

---

## ⚙️ Configuration & Rule Engine

### 📁 **Configuration Structure**
```yaml
# config/default_config.yaml
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
  anomaly_detection:
    method: "isolation_forest"  # isolation_forest, lof

validation:
  enable: true
  rules:
    - type: "column_exists"
      params: { column: "age" }
    - type: "dtype_in"
      params: { column: "age", dtypes: ["int64", "float64"] }
    - type: "max_missing_pct"
      params: { column: "income", threshold: 20 }
```

### 🔍 **Available Validation Rules**
- **`column_exists`**: Verify required columns are present
- **`dtype_in`**: Check column data types
- **`max_missing_pct`**: Enforce maximum missing data percentage
- **`value_range`**: Validate numeric value ranges
- **`allowed_values`**: Restrict categorical values
- **`max_duplicates_pct`**: Control duplicate percentage
- **`custom_rule`**: User-defined validation logic

---

## 📊 **Analysis Capabilities**

### 📈 **Statistical Analysis**
- **Descriptive Statistics**: Mean, median, std, min, max, quartiles
- **Distribution Analysis**: Histograms, box plots, normality tests
- **Correlation Matrix**: Heatmaps with multiple correlation methods
- **Data Quality Metrics**: Completeness, uniqueness, consistency scores

### 🎯 **Feature Analysis**
- **Importance Ranking**: Mutual information and correlation-based ranking
- **Target Analysis**: Feature relationships with target variables
- **Pattern Discovery**: Automatic pattern identification
- **Dimensionality Insights**: Feature correlation and redundancy

### 🔬 **Advanced Analytics**
- **Clustering Analysis**: 
  - **K-Means**: Centroid-based clustering with elbow method
  - **DBSCAN**: Density-based clustering for irregular shapes
  - **Hierarchical**: Tree-based clustering with dendrograms
- **Anomaly Detection**:
  - **Isolation Forest**: Fast anomaly detection for large datasets
  - **Local Outlier Factor**: Density-based outlier detection

### 📅 **Time Series Analysis**
- **Trend Analysis**: Moving averages and trend detection
- **Seasonality**: Seasonal decomposition and pattern recognition
- **Forecasting**: Basic time series forecasting capabilities
- **Visualization**: Time series plots and analysis charts

---

## 🎨 **Outputs & Reports**

### 📁 **File Structure**
```
Cleans-<dataset_name>/
├── cleaned_data.csv          # Cleaned dataset
├── cleaning_report.json      # Detailed cleaning summary
├── analysis_report.json      # Comprehensive analysis results
├── visualizations/           # Generated charts and plots
│   ├── correlation_heatmap.png
│   ├── distribution_plots.png
│   ├── clustering_results.png
│   └── anomaly_detection.png
└── logs/                     # Processing logs
```

### 📊 **Report Contents**
- **Cleaning Summary**: Statistics on data transformations
- **Quality Metrics**: Before/after data quality scores
- **Validation Results**: Rule engine outcomes
- **Analysis Insights**: Statistical and ML analysis results
- **Recommendations**: Suggested next steps and improvements

---

## 🏗️ **Project Architecture**

```
src/dataset_cleaner/
├── core/
│   └── cleaner.py           # 🧹 Main cleaning pipeline
├── analysis/
│   ├── analyzer.py          # 📊 Analysis orchestrator
│   ├── statistical_tests.py # 📈 Statistical testing
│   └── time_series.py       # 📅 Time series analysis
├── interfaces/
│   ├── streamlit_app.py     # 🌐 Web GUI
│   └── folder_watcher.py    # 📁 Automated processing
├── utils/
│   ├── config_manager.py    # ⚙️ Configuration management
│   ├── file_handler.py      # 📁 File I/O operations
│   ├── logger_setup.py      # 📝 Logging configuration
│   └── rule_engine.py       # 🔍 Validation rule engine
└── __init__.py

scripts/                      # 🚀 Entry points
├── run_cleaner.py           # CLI for cleaning
├── run_tests.py             # Test runner
└── create_sample_data.py    # Sample data generator

config/                       # ⚙️ Configuration files
├── default_config.yaml      # Default settings
└── config.yaml              # User overrides

tests/                        # 🧪 Test suite
docs/                         # 📚 Documentation
```

---

## 🔧 **Advanced Usage**

### 🚀 **Batch Processing**
```bash
# Process multiple files
for file in *.csv; do
    python main.py -c "$file"
done
```

### 📁 **Folder Watching**
```python
from src.dataset_cleaner.interfaces.folder_watcher import FolderWatcher

watcher = FolderWatcher(
    input_dir="./input",
    output_dir="./output",
    config_path="./config/config.yaml"
)
watcher.start()
```

### ⚙️ **Custom Configuration**
```python
from src.dataset_cleaner.utils.config_manager import ConfigManager

config = ConfigManager()
config.update({
    "cleaning": {
        "missing_values": {"strategy": "drop"},
        "outliers": {"method": "zscore", "threshold": 3}
    }
})
```

---

## 🧪 **Testing & Quality**

### 🧪 **Running Tests**
```bash
# All tests
python main.py -t

# Specific test file
python -m pytest tests/test_data_analyzer.py -v

# With coverage
python -m pytest --cov=src/dataset_cleaner tests/
```

### 📊 **Test Coverage**
- **Unit Tests**: Core functionality testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Large dataset handling
- **Edge Cases**: Error handling and boundary conditions

---

## 📚 **Documentation & Community**

### 📖 **Guides & References**
- **Contributing**: [CONTRIBUTING.md](./CONTRIBUTING.md) - Setup, code style, tests, PR rules
- **Code of Conduct**: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) - Community guidelines
- **Security**: [SECURITY.md](./SECURITY.md) - Vulnerability reporting
- **Code Owners**: [CODEOWNERS](./CODEOWNERS) - Review requirements

### 🤝 **Getting Help**
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Wiki**: Community-contributed guides and tips
- **Examples**: Sample datasets and use cases

---

## 🚀 **Performance & Scalability**

### ⚡ **Optimization Features**
- **Memory Management**: Efficient handling of large datasets
- **Parallel Processing**: Multi-threaded operations where possible
- **Lazy Evaluation**: On-demand computation for large operations
- **Caching**: Intelligent caching of intermediate results

### 📊 **Performance Benchmarks**
- **Small Datasets** (< 1MB): < 1 second
- **Medium Datasets** (1-100MB): 1-30 seconds
- **Large Datasets** (100MB-1GB): 30 seconds - 5 minutes
- **Very Large Datasets** (> 1GB): Configurable chunking

---

## 🔮 **Roadmap & Future Features**

### 🎯 **Short Term** (Next 3 months)
- [ ] **Database Support**: Direct database connections
- [ ] **API Endpoints**: RESTful API for integration
- [ ] **Real-time Processing**: Streaming data support
- [ ] **Enhanced Visualizations**: Interactive charts and dashboards

### 🚀 **Medium Term** (3-6 months)
- [ ] **Machine Learning Pipeline**: Automated ML model training
- [ ] **Cloud Integration**: AWS, Azure, GCP support
- [ ] **Collaborative Features**: Team workspaces and sharing
- [ ] **Advanced Rules**: Custom Python rule functions

### 🌟 **Long Term** (6+ months)
- [ ] **AI-Powered Cleaning**: Intelligent data transformation suggestions
- [ ] **Multi-language Support**: R, Julia, and other language bindings
- [ ] **Enterprise Features**: Role-based access control and audit trails
- [ ] **Mobile App**: iOS and Android applications

---

## 🤝 **Contributing**

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for:
- 🚀 **Setup Guide**: Local development environment
- 📝 **Code Style**: Python and project standards
- 🧪 **Testing**: Writing and running tests
- 🔄 **PR Process**: Pull request guidelines
- 🐛 **Bug Reports**: Issue reporting templates

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **Typer/Rich**: Beautiful CLI interfaces
- **Streamlit**: Web application framework

---

<div align="center">

**Made with ❤️ by the CleanEngine Community**

[![GitHub stars](https://img.shields.io/github/stars/I-invincib1e/CleanEngine?style=social)](https://github.com/I-invincib1e/CleanEngine)
[![GitHub forks](https://img.shields.io/github/forks/I-invincib1e/CleanEngine?style=social)](https://github.com/I-invincib1e/CleanEngine)
[![GitHub issues](https://img.shields.io/github/issues/I-invincib1e/CleanEngine)](https://github.com/I-invincib1e/CleanEngine/issues)

**⭐ Star this repository if it helped you! ⭐**

</div>

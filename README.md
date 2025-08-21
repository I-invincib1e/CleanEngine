# 🧹 Automated Dataset Cleaner

A comprehensive Python tool for automatically cleaning CSV and Excel datasets with detailed reporting and multiple interfaces (CLI, GUI, and automation).

## ✨ Features

### 🧹 Core Cleaning Capabilities
- **Smart Data Cleaning**: Handles missing values, duplicates, outliers, and data normalization
- **Multiple Interfaces**: CLI, Streamlit GUI, and folder watcher for automation
- **Flexible Configuration**: Customizable cleaning parameters and methods
- **File Format Support**: CSV and Excel files (.csv, .xlsx, .xls)
- **Organized Output**: Automatic folder creation with structured file organization

### 🔬 Advanced Data Analysis (NEW!)
- **Statistical Analysis**: Descriptive statistics, skewness, kurtosis analysis
- **Correlation Analysis**: Automatic detection of strong correlations between variables
- **Distribution Analysis**: Normality tests and data balance assessment
- **Feature Importance**: Mutual information-based feature ranking
- **Clustering Analysis**: Automatic optimal cluster detection using K-means
- **Anomaly Detection**: Isolation Forest-based outlier identification
- **Data Quality Scoring**: Comprehensive quality assessment with actionable scores
- **Automated Insights**: AI-generated insights and recommendations

### 📊 Professional Visualizations
- **Correlation Heatmaps**: Beautiful correlation matrices
- **Distribution Plots**: Histograms for all numeric variables
- **Feature Importance Charts**: Ranked feature importance visualization
- **Quality Dashboards**: Multi-panel data quality overview
- **Before/After Comparisons**: Visual cleaning impact assessment

### 📋 Comprehensive Reporting
- **Markdown Reports**: Professional analysis reports with insights
- **JSON Data**: Structured analysis results for further processing
- **Cleaning Summaries**: Human-readable cleaning process documentation
- **Actionable Recommendations**: Data-driven suggestions for improvement

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project files
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Clean a dataset with default settings
python dataset_cleaner.py data.csv

# Clean with custom parameters
python dataset_cleaner.py data.csv -o cleaned_data.csv --outlier-method zscore --encoding onehot

# Quick cleaning (minimal options)
python run_cleaner.py data.csv
```

### GUI Interface

```bash
# Launch the Streamlit web interface
streamlit run streamlit_app.py
```

### Automated Folder Watching

```bash
# Watch a folder for new files and auto-clean them
python folder_watcher.py ./input_folder -o ./cleaned_output
```

## 🛠️ Cleaning Pipeline

The tool performs the following cleaning steps:

1. **Data Loading**: Supports CSV and Excel files
2. **Missing Values**: Fill with mean/median/mode or drop columns with too many missing values
3. **Duplicates**: Remove duplicate rows
4. **Outliers**: Detect and remove using IQR or Z-score methods
5. **Categorical Encoding**: Label encoding or one-hot encoding
6. **Normalization**: MinMax scaling or standardization
7. **Reporting**: Generate detailed before/after reports

## 📊 Configuration Options

### Missing Values
- **Threshold**: Drop columns with missing values above this ratio (default: 0.5)
- **Strategy**: Auto-fill numeric with median, categorical with mode

### Outlier Detection
- **IQR Method**: Remove values outside 1.5 * IQR range
- **Z-Score Method**: Remove values with |z-score| > 3

### Categorical Encoding
- **Label Encoding**: Convert categories to numeric labels
- **One-Hot Encoding**: Create binary columns for each category

### Normalization
- **MinMax Scaling**: Scale values to 0-1 range
- **Standard Scaling**: Standardize to mean=0, std=1

## 📋 CLI Arguments

```bash
python dataset_cleaner.py input_file [options]

Options:
  -o, --output              Output file path
  --missing-threshold       Threshold for dropping columns (0.1-1.0)
  --outlier-method         'iqr' or 'zscore'
  --encoding               'label' or 'onehot'
  --normalization          'minmax' or 'standard'
```

## 📂 Organized Output Structure

The tool creates an organized folder structure for each dataset:

```
Cleans-{dataset_name}/
├── cleaned_{dataset_name}.csv/xlsx           # Cleaned dataset
├── {dataset_name}_cleaning_report.json       # Detailed cleaning report
├── {dataset_name}_cleaning_summary.txt       # Human-readable cleaning summary
├── {dataset_name}_analysis_report.md         # Comprehensive analysis report
├── {dataset_name}_analysis_results.json      # Structured analysis data
└── visualizations/                           # Analysis visualizations
    ├── correlation_heatmap.png
    ├── distributions.png
    ├── feature_importance.png
    └── quality_dashboard.png
```

**Example for `sample_data.csv`:**
```
Cleans-sample_data/
├── cleaned_sample_data.csv
├── sample_data_cleaning_report.json
├── sample_data_cleaning_summary.txt
├── sample_data_analysis_report.md
├── sample_data_analysis_results.json
└── visualizations/
    ├── correlation_heatmap.png
    ├── distributions.png
    ├── feature_importance.png
    └── quality_dashboard.png
```

This organized structure keeps all related files together and prevents clutter in your working directory.

## 🔬 Advanced Analysis Features

### Statistical Insights
- **Descriptive Statistics**: Mean, median, std, quartiles for all numeric columns
- **Skewness & Kurtosis**: Distribution shape analysis
- **Normality Testing**: Statistical tests for normal distribution

### Correlation & Relationships
- **Correlation Matrix**: Pearson correlation between all numeric variables
- **Strong Correlation Detection**: Automatic identification of highly correlated pairs (>0.7)
- **Feature Importance**: Mutual information scores for feature ranking

### Data Quality Assessment
- **Completeness Score**: Percentage of non-missing data
- **Uniqueness Analysis**: Unique value ratios for each column
- **Consistency Checks**: Detection of formatting inconsistencies
- **Overall Quality Score**: Composite score (0-100) for data quality

### Advanced Analytics
- **Clustering Analysis**: K-means clustering with optimal cluster detection
- **Anomaly Detection**: Isolation Forest for outlier identification
- **Distribution Analysis**: Statistical distribution fitting and testing

### Automated Insights
The system generates actionable insights such as:
- "High anomaly rate detected: 10.0% of data points"
- "Strong correlation found between Age and Salary (0.972)"
- "Data naturally groups into 6 clusters"
- "Small dataset - consider collecting more data for robust analysis"

## 🖥️ Streamlit GUI Features

- **File Upload**: Drag and drop CSV/Excel files
- **Interactive Configuration**: Adjust cleaning parameters with sliders and dropdowns
- **Real-time Visualizations**: Missing value heatmaps, outlier boxplots
- **Before/After Comparisons**: Visual charts showing cleaning impact
- **Download Results**: Get cleaned datasets and reports instantly

## 🔄 Folder Watcher

The folder watcher automatically processes new files:

```bash
python folder_watcher.py watch_folder -o output_folder
```

- Monitors a folder for new CSV/Excel files
- Automatically cleans them when detected
- Creates organized `Cleans-{dataset_name}` folders in the output directory
- Generates complete reports for each processed file

## 📊 Example Report Output

```
==================================================
DATASET CLEANING REPORT
==================================================

Timestamp: 2025-01-21T10:30:45

Dataset Shape:
  Original: 10,000 rows × 15 columns
  Final: 9,234 rows × 12 columns
  Rows removed: 766

Missing Values:
  Before: 1,245
  After: 0
  Cleaned: 1,245

Duplicates Removed: 156

Outliers Removed: 234
  price: 89
  age: 67
  income: 78

Categorical Encoding:
  category: label_encoded
  region: label_encoded

Normalization: minmax
Normalized columns: 8
```

## 🔧 Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- openpyxl >= 3.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- streamlit >= 1.20.0
- plotly >= 5.0.0
- watchdog >= 2.1.0

## 🎯 Use Cases

- **Data Science Projects**: Clean datasets before analysis
- **Machine Learning**: Prepare data for model training
- **Business Analytics**: Standardize reporting data
- **Data Migration**: Clean data during system transfers
- **Automated Pipelines**: Set up automated data cleaning workflows

## 🚀 Future Extensions

- Email/Telegram notifications
- More file format support (JSON, Parquet)
- Custom cleaning rules
- Database connectivity
- Docker containerization
- Web API with FastAPI
- Data profiling and validation

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the tool!
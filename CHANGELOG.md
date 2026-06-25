# Changelog

All notable changes to CleanEngine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-06-25

### Added

- **🗂️ Domain Templates** (`domain_templates.py`)
  - 6 pre-built cleaning configurations: General, Healthcare, Finance, E-commerce, Survey, IoT
  - Each template includes: default parameters, column hints, domain-specific tips, validation rules
  - Selecting a template in the GUI auto-populates all sidebar settings
  - Programmatic access via `get_template(key)` and `list_templates()`

- **🧾 Cleaning Recipe Export** (`recipe_exporter.py`)
  - `generate_python_recipe(report, filename)` — produces a standalone, production-ready Python cleaning script from a cleaning report
  - `generate_sql_recipe(report, table_name)` — produces an ANSI SQL `CREATE TABLE AS` query with deduplication CTEs, IQR outlier bounds, encoding, and normalization
  - `generate_yaml_config(...)` — exports a shareable YAML config reproducing the exact cleaning settings
  - All three export formats available as download buttons in the GUI after cleaning

- **📡 Data Drift Monitor** (`drift_monitor.py`)
  - `compare_datasets(df_ref, df_new)` — statistical comparison of two dataset versions
  - Numeric drift: Kolmogorov-Smirnov 2-sample test (p < 0.05 = drifted)
  - Categorical drift: Chi-squared goodness-of-fit test
  - Missing value shift detection (flagged when delta > 5 percentage points)
  - Schema diff: added and removed columns
  - New category detection in categorical columns
  - Severity rating: None / Low / Medium / High
  - Full drift report exportable as JSON

- **🔍 Anomaly Explorer** (`anomaly_explainer.py`)
  - `explain_anomalies(df, contamination, max_anomalies)` — Isolation Forest detection with per-row plain-English explanations
  - Per-feature deviation analysis: z-score, percentile rank, IQR fence breaches, direction
  - Column contribution chart showing which features drive the most anomalies
  - Anomaly score distribution histogram
  - Download anomalous rows as CSV or full report as JSON

- **Multi-tab Streamlit GUI**
  - Rebuilt as three-tab interface: Clean & Export / Drift Monitor / Anomaly Explorer
  - Domain template selector in sidebar with live tips expander
  - Cleaning Recipe Export section (Python / SQL / YAML tabs) rendered after every clean
  - File format support extended to JSON and Parquet in the GUI uploader

### Changed

- **Streamlit app** (`streamlit_app.py`) — complete rewrite as a multi-tab application
- **Import paths** — fixed `from data_analyzer import DataAnalyzer` → `from dataset_cleaner.analysis.analyzer import DataAnalyzer` for correct PYTHONPATH operation
- **Comparison table** in README updated to reflect new features vs. competing tools
- **docs/README.md** — updated to reflect new module structure and API
- **docs/ADVANCED_FEATURES.md** — full rewrite covering all four new features with examples

### Fixed

- Streamlit uploader now supports JSON and Parquet in addition to CSV/Excel
- Temp file cleanup uses `tempfile.NamedTemporaryFile` for safer cross-platform handling

---

## [0.1.0] - 2024-01-XX

### Added

- **Core Data Cleaning Pipeline**
  - Missing value handling: drop above threshold, median imputation (numeric), mode imputation (categorical)
  - Duplicate detection and removal
  - Outlier detection: IQR and Z-score methods
  - Categorical encoding: Label and One-Hot
  - Normalization: Min-Max and Standard scaling

- **Advanced Analytics**
  - Descriptive statistics, skewness, kurtosis
  - Correlation analysis (Pearson, Spearman, Kendall)
  - Feature importance via Mutual Information
  - Distribution analysis and Shapiro-Wilk normality tests
  - Data quality scoring (completeness, uniqueness, consistency → 0–100)

- **Machine Learning Capabilities**
  - Clustering: K-Means (Elbow method), DBSCAN, Hierarchical
  - Anomaly detection: Isolation Forest, Local Outlier Factor

- **Rule Engine & Validation**
  - YAML-driven validation rules
  - Pre/post cleaning validation
  - Custom rule system and data governance support

- **Multiple Interfaces**
  - Rich CLI with Typer
  - Streamlit web interface
  - Folder watcher for automation

- **File Format Support**
  - CSV (multi-encoding detection), Excel (XLSX/XLS), JSON, Parquet
  - PyArrow and FastParquet engines

- **Output & Reporting**
  - JSON cleaning reports and human-readable `.txt` summaries
  - Markdown analysis reports
  - Visualization gallery (PNG): correlation heatmap, distributions, feature importance, quality dashboard

- **Configuration Management**
  - YAML configuration files with sensible defaults

### Changed

- **Project Rebranding**: Renamed from "AD Cleaner" to "CleanEngine"
- **Architecture**: Refactored to modular `src/` structure
- **CLI**: Replaced basic argparse with Typer + Rich
- **Packaging**: Modern Python packaging with `pyproject.toml`

### Fixed

- Import errors in modular structure
- Logger configuration issues
- Exception handling improvements

---

## Version History

| Version | Date | Highlights |
|---------|------|-----------|
| **0.2.0** | 2026-06-25 | Domain Templates, Recipe Export, Drift Monitor, Anomaly Explorer |
| **0.1.0** | 2024-01-XX | Initial release — core cleaning, analytics, CLI, Streamlit GUI |
| **Pre-0.1.0** | — | Development and alpha versions |

---

## Contributing to the Changelog

When adding new entries, follow these guidelines:
1. Use sections: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`
2. Be descriptive — explain what changed and why, not just that it changed
3. Group related changes with bullet points
4. Reference issue/PR numbers where applicable

For the release process, see [CONTRIBUTING.md](CONTRIBUTING.md).

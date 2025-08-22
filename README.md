# 🧹 CleanEngine

[![Tests](https://img.shields.io/badge/tests-36%20passing-brightgreen)](./scripts/run_tests.py)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](./LICENSE)
[![CLI](https://img.shields.io/badge/CLI-Rich%20%2B%20Typer-purple)](#)
[![Main Branch](https://img.shields.io/badge/branch-main-blue)](#)

CleanEngine is a fast, modular toolkit for data cleaning, profiling, and analysis. It supports CSV/Excel/JSON/Parquet, produces reports and visualizations, and provides a YAML-driven rule engine for validation.

**🚀 Current Status**: Active development on `main` branch with enhanced CLI features, advanced analytics, and modular architecture.

## ✨ Features
- Cleaning: missing values, duplicates, outliers, encoding, normalization
- Profiling & Analysis: stats, distributions, correlations, feature importance, clustering, anomalies
- Rule Engine: YAML-defined validation rules (pre/post clean)
- Multi-format IO: CSV, Excel, JSON, Parquet (pyarrow/fastparquet)
- Interfaces: CLI, Streamlit (GUI), folder watcher

## 🚀 Quick Start
```bash
# 1) Install dependencies
python setup.py

# 2) Interactive menu
python main.py

# Short commands
python main.py -s                  # create sample datasets
python main.py -c sample_mixed.csv # clean a dataset
python main.py -t                  # run tests
python main.py -g                  # launch Streamlit GUI
```

## 🧰 CLI (Typer + Rich)
- clean (alias: `c`): `python main.py c <file> [out]`
- samples (alias: `s`): `python main.py s`
- tests (alias: `t`): `python main.py t`
- gui (alias: `g`): `python main.py g`
- No args shows an interactive menu.

## 🧪 Rule Engine (YAML)
Enable in `config/default_config.yaml`:
```yaml
validation:
  enable: true
  rules:
    - type: column_exists
      params: { column: "age" }
    - type: dtype_in
      params: { column: "age", dtypes: ["int64","float64"] }
    - type: max_missing_pct
      params: { column: "income", threshold: 20 }
    - type: value_range
      params: { column: "age", min: 0, max: 120 }
    - type: allowed_values
      params: { column: "status", values: ["A","B","C"] }
    - type: max_duplicates_pct
      params: { threshold: 10 }
```
Pre/post validation results are stored in the cleaning report under `validation_pre` and `validation_post`.

## 📁 Structure
```
src/dataset_cleaner/
  core/cleaner.py        # Cleaning pipeline
  analysis/              # Profiling & analysis
  interfaces/            # CLI/Streamlit/folder watcher
  utils/                 # Config, file IO, logging, rule_engine
scripts/                 # Entrypoints
config/                  # Config YAML (defaults & overrides)
```

## 📝 Notes
- Formats: CSV, XLSX, JSON, Parquet
- Engines: pyarrow (default) or fastparquet
- Outputs: reports and visualizations under `Cleans-<dataset>/`

## 📜 License
MIT

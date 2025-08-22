# CleanEngine Docs

CleanEngine is a modular toolkit for data cleaning, profiling, and analysis with a YAML-driven rule engine.

## Overview
- Cleaning: missing values, duplicates, outliers, encoding, normalization
- Profiling & Analysis: stats, distributions, correlations, feature importance, clustering, anomalies
- Rule Engine: YAML-defined validation (pre/post clean) with stable results
- IO: CSV, Excel, JSON, Parquet (pyarrow/fastparquet)
- Interfaces: CLI, Streamlit

## Quick Start
```bash
python setup.py       # install dependencies
python main.py        # interactive menu
python main.py -c <file> [out]
python main.py -s     # sample datasets
python main.py -t     # tests
python main.py -g     # GUI (Streamlit)
```

For full usage, configuration, and examples, see the top-level README.
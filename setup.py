#!/usr/bin/env python3
"""
CleanEngine - The Ultimate Data Cleaning & Analysis Toolkit
"""

import os
import sys
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = this_directory / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Package metadata
PACKAGE_NAME = "cleanengine"
VERSION = "0.1.0"
DESCRIPTION = "The Ultimate Data Cleaning & Analysis Toolkit"
AUTHOR = "CleanEngine Community"
AUTHOR_EMAIL = "contact@cleanengine.com"
URL = "https://github.com/I-invincib1e/CleanEngine"
PROJECT_URLS = {
    "Bug Reports": "https://github.com/I-invincib1e/CleanEngine/issues",
    "Source": "https://github.com/I-invincib1e/CleanEngine",
    "Documentation": "https://github.com/I-invincib1e/CleanEngine#readme",
    "Changelog": "https://github.com/I-invincib1e/CleanEngine/blob/main/CHANGELOG.md",
}

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Data Scientists",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Text Processing :: Filters",
    "Topic :: Database",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Visualization",
]

# Keywords for PyPI search
KEYWORDS = [
    "data-cleaning",
    "data-analysis",
    "data-profiling",
    "machine-learning",
    "data-science",
    "pandas",
    "scikit-learn",
    "data-validation",
    "rule-engine",
    "cli-tool",
    "data-quality",
    "outlier-detection",
    "clustering",
    "anomaly-detection",
    "correlation-analysis",
    "feature-importance",
    "data-visualization",
    "streamlit",
    "yaml-config",
    "automation",
]

# Dependencies
INSTALL_REQUIRES = read_requirements()

# Development dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
        "tox>=4.0.0",
    ],
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=1.0.0",
        "sphinx-autodoc-typehints>=1.23.0",
    ],
    "full": [
        "jupyter>=1.0.0",
        "ipython>=8.0.0",
        "notebook>=6.5.0",
    ],
}

# Entry points for CLI
ENTRY_POINTS = {
    "console_scripts": [
        "cleanengine=src.dataset_cleaner.cli:main",
        "ce=src.dataset_cleaner.cli:main",  # Short alias
    ],
}

# Package data
PACKAGE_DATA = {
    "dataset_cleaner": [
        "config/*.yaml",
        "config/*.yml",
        "*.py",
    ],
}

# Python version requirement
PYTHON_REQUIRES = ">=3.9"

# Setup configuration
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    project_urls=PROJECT_URLS,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data=PACKAGE_DATA,
    include_package_data=True,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points=ENTRY_POINTS,
    zip_safe=False,
    license="MIT",
    platforms=["any"],
    download_url=f"{URL}/archive/v{VERSION}.tar.gz",
    bugtrack_url=f"{URL}/issues",
    exclude_package_data={
        "": [
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "__pycache__",
            "*.so",
            "*.dll",
            "*.dylib",
        ]
    },
)

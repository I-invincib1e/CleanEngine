#!/usr/bin/env python3
"""
Domain-Specific Cleaning Templates
Pre-built cleaning configurations and validation rules for common dataset types.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainTemplate:
    name: str
    icon: str
    description: str
    missing_threshold: float
    outlier_method: str
    encoding_method: str
    normalization_method: str
    tips: list[str]
    column_hints: dict[str, str]
    validation_checks: list[dict[str, Any]] = field(default_factory=list)


TEMPLATES: dict[str, DomainTemplate] = {
    "general": DomainTemplate(
        name="General Purpose",
        icon="🔧",
        description="Balanced defaults suitable for any dataset. Good starting point.",
        missing_threshold=0.5,
        outlier_method="iqr",
        encoding_method="label",
        normalization_method="minmax",
        tips=[
            "Works well for most tabular datasets.",
            "IQR outlier detection is robust to skewed distributions.",
            "Label encoding preserves ordinal information where applicable.",
        ],
        column_hints={},
    ),
    "healthcare": DomainTemplate(
        name="Healthcare / Clinical",
        icon="🏥",
        description="Optimized for patient records, clinical trials, and medical data with strict completeness.",
        missing_threshold=0.2,
        outlier_method="zscore",
        encoding_method="label",
        normalization_method="standard",
        tips=[
            "Low missing-value tolerance (20%) — clinical data must be nearly complete.",
            "Z-score outliers are used as extreme vital-sign values can be real events.",
            "Standard normalization works better for normally distributed medical measurements.",
            "Watch for age < 0 or > 130, negative weights, and impossible dates.",
            "Check for PII columns (names, SSNs, DOBs) before sharing outputs.",
        ],
        column_hints={
            "age": "Expected range: 0–130",
            "weight": "Expected range: 0.5–500 kg",
            "bmi": "Expected range: 10–80",
            "blood_pressure": "Format: systolic/diastolic",
            "diagnosis": "Verify against ICD-10 code list",
        },
        validation_checks=[
            {"column": "age", "rule": "range", "min": 0, "max": 130},
            {"column": "bmi", "rule": "range", "min": 10, "max": 80},
        ],
    ),
    "finance": DomainTemplate(
        name="Finance / Transactions",
        icon="💰",
        description="For transaction logs, account data, and financial records requiring accuracy and consistency.",
        missing_threshold=0.1,
        outlier_method="iqr",
        encoding_method="label",
        normalization_method="standard",
        tips=[
            "Very low missing tolerance (10%) — financial records should be complete.",
            "IQR detects extreme transaction amounts that may indicate fraud.",
            "Standard normalization is preferred for monetary values (fat-tailed distributions).",
            "Check for negative balances where unexpected, duplicate transaction IDs.",
            "Verify currency columns have consistent formatting (strip $ € £ symbols).",
        ],
        column_hints={
            "amount": "Should be positive unless credits allowed",
            "transaction_id": "Must be unique",
            "account_id": "Should not be null",
            "currency": "Standardize to 3-letter ISO codes",
            "date": "Ensure consistent date format",
        },
        validation_checks=[
            {"column": "amount", "rule": "range", "min": 0, "max": 1e9},
        ],
    ),
    "ecommerce": DomainTemplate(
        name="E-commerce / Retail",
        icon="🛒",
        description="Product catalogues, order data, and customer records for online retail.",
        missing_threshold=0.3,
        outlier_method="iqr",
        encoding_method="onehot",
        normalization_method="minmax",
        tips=[
            "One-hot encoding works well for product categories and sizes.",
            "IQR catches price extremes and unusually large quantities.",
            "Min-Max normalization maps prices and quantities to [0,1] for ML.",
            "Deduplicate product SKUs — same item often appears with slight name variations.",
            "Check for orders with 0 or negative prices/quantities.",
        ],
        column_hints={
            "price": "Should be non-negative",
            "quantity": "Should be positive integer",
            "sku": "Standardize format, remove whitespace",
            "category": "Standardize capitalization",
            "rating": "Expected range: 1–5",
        },
        validation_checks=[
            {"column": "price", "rule": "range", "min": 0, "max": 1e6},
            {"column": "quantity", "rule": "range", "min": 1, "max": 1e4},
            {"column": "rating", "rule": "range", "min": 1, "max": 5},
        ],
    ),
    "survey": DomainTemplate(
        name="Survey / Research",
        icon="📋",
        description="Questionnaire responses, Likert scales, and qualitative research data.",
        missing_threshold=0.4,
        outlier_method="iqr",
        encoding_method="label",
        normalization_method="minmax",
        tips=[
            "Higher missing tolerance (40%) — respondents often skip optional questions.",
            "Label encoding preserves Likert scale order (Strongly Disagree → Strongly Agree).",
            "Watch for straight-liners: respondents who picked the same answer every time.",
            "Free-text columns may need separate NLP processing.",
            "Check that all Likert responses fall within defined scale range.",
        ],
        column_hints={
            "likert_*": "Expected range: 1–5 or 1–7",
            "age": "Validate as numeric, expected 18–100 for adults",
            "response_time": "Suspiciously fast responses may indicate bots",
            "open_ended": "Flag for manual review — not auto-cleaned",
        },
        validation_checks=[],
    ),
    "iot": DomainTemplate(
        name="IoT / Sensor Data",
        icon="📡",
        description="Time-series sensor readings, device telemetry, and operational data.",
        missing_threshold=0.15,
        outlier_method="zscore",
        encoding_method="label",
        normalization_method="standard",
        tips=[
            "Low missing tolerance (15%) — gaps in sensor data indicate device failures.",
            "Z-score is preferred: sensor readings are usually normally distributed.",
            "Standard normalization preserves the physical meaning of deviations.",
            "Timestamps must be monotonically increasing — check for clock drift.",
            "Consider rolling median imputation for short sensor gaps instead of global median.",
        ],
        column_hints={
            "timestamp": "Must be monotonically increasing, no duplicates",
            "temperature": "Validate against device operating range",
            "humidity": "Expected range: 0–100%",
            "device_id": "Group analysis by device ID",
            "status": "Map error codes to descriptions",
        },
        validation_checks=[
            {"column": "humidity", "rule": "range", "min": 0, "max": 100},
        ],
    ),
}


def get_template(key: str) -> DomainTemplate:
    return TEMPLATES.get(key, TEMPLATES["general"])


def list_templates() -> list[tuple[str, DomainTemplate]]:
    return list(TEMPLATES.items())

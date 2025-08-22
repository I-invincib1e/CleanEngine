#!/usr/bin/env python3
"""
Rule Engine: Evaluate YAML-defined validation rules against a pandas DataFrame.

Supported rule types (examples):
- column_exists: { column: "age" }
- dtype_in: { column: "age", dtypes: ["int64", "float64"] }
- max_missing_pct: { column: "age", threshold: 20 }            # percent
- value_range: { column: "age", min: 0, max: 120 }
- allowed_values: { column: "status", values: ["A","B","C"] }
- max_duplicates_pct: { threshold: 20 }                          # percent of duplicate rows

Returns a dict with summary and violations list. Designed to be stable and safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class Rule:
    type: str
    params: Dict[str, Any]


class RuleEngine:
    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None):
        self.rules: List[Rule] = [
            Rule(type=r.get("type", ""), params=r.get("params", {}))
            for r in (rules or [])
        ]

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []

        for idx, rule in enumerate(self.rules):
            rtype = (rule.type or "").lower()
            params = rule.params or {}
            try:
                if rtype == "column_exists":
                    col = params.get("column")
                    if col is None or col not in df.columns:
                        violations.append(
                            {
                                "rule_index": idx,
                                "type": rtype,
                                "column": col,
                                "reason": f"Column '{col}' not found",
                            }
                        )

                elif rtype == "dtype_in":
                    col = params.get("column")
                    allowed = set(params.get("dtypes", []))
                    if col not in df.columns:
                        violations.append(
                            {
                                "rule_index": idx,
                                "type": rtype,
                                "column": col,
                                "reason": f"Column '{col}' not found",
                            }
                        )
                    else:
                        actual = str(df[col].dtype)
                        if allowed and actual not in allowed:
                            violations.append(
                                {
                                    "rule_index": idx,
                                    "type": rtype,
                                    "column": col,
                                    "reason": f"dtype '{actual}' not in {sorted(allowed)}",
                                }
                            )

                elif rtype == "max_missing_pct":
                    col = params.get("column")
                    threshold = float(params.get("threshold", 0))
                    if col not in df.columns:
                        violations.append(
                            {
                                "rule_index": idx,
                                "type": rtype,
                                "column": col,
                                "reason": f"Column '{col}' not found",
                            }
                        )
                    else:
                        pct = (df[col].isna().sum() / max(len(df), 1)) * 100
                        if pct > threshold:
                            violations.append(
                                {
                                    "rule_index": idx,
                                    "type": rtype,
                                    "column": col,
                                    "reason": f"Missing {pct:.1f}% exceeds {threshold:.1f}%",
                                }
                            )

                elif rtype == "value_range":
                    col = params.get("column")
                    vmin = params.get("min", None)
                    vmax = params.get("max", None)
                    if col not in df.columns:
                        violations.append(
                            {
                                "rule_index": idx,
                                "type": rtype,
                                "column": col,
                                "reason": f"Column '{col}' not found",
                            }
                        )
                    else:
                        series = pd.to_numeric(df[col], errors="coerce")
                        below = int((series < vmin).sum()) if vmin is not None else 0
                        above = int((series > vmax).sum()) if vmax is not None else 0
                        if below > 0 or above > 0:
                            violations.append(
                                {
                                    "rule_index": idx,
                                    "type": rtype,
                                    "column": col,
                                    "reason": f"{below} values < {vmin}, {above} values > {vmax}",
                                }
                            )

                elif rtype == "allowed_values":
                    col = params.get("column")
                    allowed = set(params.get("values", []))
                    if col not in df.columns:
                        violations.append(
                            {
                                "rule_index": idx,
                                "type": rtype,
                                "column": col,
                                "reason": f"Column '{col}' not found",
                            }
                        )
                    elif allowed:
                        uniques = set(pd.Series(df[col]).dropna().astype(str).unique())
                        invalid = sorted(
                            [v for v in uniques if v not in set(map(str, allowed))]
                        )
                        if invalid:
                            violations.append(
                                {
                                    "rule_index": idx,
                                    "type": rtype,
                                    "column": col,
                                    "reason": f"Invalid values present: {invalid}",
                                }
                            )

                elif rtype == "max_duplicates_pct":
                    threshold = float(params.get("threshold", 0))
                    dup_count = int(df.duplicated().sum())
                    pct = (dup_count / max(len(df), 1)) * 100
                    if pct > threshold:
                        violations.append(
                            {
                                "rule_index": idx,
                                "type": rtype,
                                "reason": f"Duplicate rows {pct:.1f}% exceeds {threshold:.1f}%",
                            }
                        )

                else:
                    # Unknown rule types are ignored but recorded
                    violations.append(
                        {
                            "rule_index": idx,
                            "type": rtype,
                            "reason": "Unknown rule type",
                        }
                    )
            except Exception as e:
                violations.append(
                    {
                        "rule_index": idx,
                        "type": rtype,
                        "reason": f"Rule evaluation error: {e}",
                    }
                )

        return {
            "rules_evaluated": len(self.rules),
            "violations_count": len(violations),
            "violations": violations,
        }

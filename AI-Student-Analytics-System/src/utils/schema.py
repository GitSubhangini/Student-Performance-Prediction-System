"""
Shared feature schema and validation for StudentAI.

The same schema is used by training, form submissions, CSV uploads, and the
JSON API so the app does not drift into several slightly different contracts.
"""
from __future__ import annotations

from math import isfinite


class InputValidationError(ValueError):
    """Raised when incoming prediction data cannot be safely scored."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


NUMERICAL_FEATURES = [
    "study_hours_per_week",
    "attendance_rate",
    "math_score",
    "reading_score",
    "writing_score",
    "science_score",
]

CATEGORICAL_FEATURES = [
    "gender",
    "race_ethnicity",
    "parental_education",
    "lunch",
    "test_prep_course",
    "extracurricular",
    "internet_access",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "final_score"

NUMERIC_RANGES = {
    "study_hours_per_week": (0.0, 40.0),
    "attendance_rate": (0.0, 100.0),
    "math_score": (0.0, 100.0),
    "reading_score": (0.0, 100.0),
    "writing_score": (0.0, 100.0),
    "science_score": (0.0, 100.0),
}

CATEGORICAL_OPTIONS = {
    "gender": ["male", "female"],
    "race_ethnicity": ["Group A", "Group B", "Group C", "Group D", "Group E"],
    "parental_education": [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ],
    "lunch": ["standard", "free/reduced"],
    "test_prep_course": ["completed", "none"],
    "extracurricular": ["yes", "no"],
    "internet_access": ["yes", "no"],
}

FEATURE_LABELS = {
    "study_hours_per_week": "Study hours per week",
    "attendance_rate": "Attendance rate",
    "math_score": "Current math score",
    "reading_score": "Current reading score",
    "writing_score": "Current writing score",
    "science_score": "Current science score",
    "gender": "Gender",
    "race_ethnicity": "Race / ethnicity",
    "parental_education": "Parental education",
    "lunch": "Lunch type",
    "test_prep_course": "Test prep course",
    "extracurricular": "Extracurricular activity",
    "internet_access": "Internet access",
    TARGET: "Predicted final score",
}

CSV_OPTIONAL_COLUMNS = ["student_name", "student_id"]
DROP_COLUMNS = ["student_id", "risk_level", "average_score", "current_average_score"]


def feature_schema_payload() -> dict:
    """Return a JSON-serializable schema snapshot for model metadata."""
    return {
        "target": TARGET,
        "numerical": {
            name: {
                "label": FEATURE_LABELS[name],
                "min": NUMERIC_RANGES[name][0],
                "max": NUMERIC_RANGES[name][1],
            }
            for name in NUMERICAL_FEATURES
        },
        "categorical": {
            name: {
                "label": FEATURE_LABELS[name],
                "options": CATEGORICAL_OPTIONS[name],
            }
            for name in CATEGORICAL_FEATURES
        },
    }


def _canonical_category(field: str, value) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    for option in CATEGORICAL_OPTIONS[field]:
        if raw.casefold() == option.casefold():
            return option
    return raw


def validate_student_input(data: dict) -> dict:
    """
    Validate and normalize one prediction payload.

    Returns a cleaned dict with the exact model feature names and canonical
    categorical values. Raises InputValidationError with all discovered errors.
    """
    errors: list[str] = []
    cleaned: dict = {}

    for field in NUMERICAL_FEATURES:
        label = FEATURE_LABELS[field]
        value = data.get(field)
        if value is None or str(value).strip() == "":
            errors.append(f"{label} is required")
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number")
            continue
        lo, hi = NUMERIC_RANGES[field]
        if not isfinite(numeric):
            errors.append(f"{label} must be finite")
        elif numeric < lo or numeric > hi:
            errors.append(f"{label} must be between {lo:g} and {hi:g}")
        else:
            cleaned[field] = numeric

    for field in CATEGORICAL_FEATURES:
        label = FEATURE_LABELS[field]
        value = _canonical_category(field, data.get(field))
        if value is None:
            errors.append(f"{label} is required")
            continue
        if value not in CATEGORICAL_OPTIONS[field]:
            allowed = ", ".join(CATEGORICAL_OPTIONS[field])
            errors.append(f"{label} must be one of: {allowed}")
        else:
            cleaned[field] = value

    if errors:
        raise InputValidationError(errors)
    return cleaned


def missing_csv_columns(columns) -> list[str]:
    """Return required prediction columns absent from an uploaded CSV."""
    existing = {str(col).strip() for col in columns}
    return [field for field in ALL_FEATURES if field not in existing]


def display_feature_name(name: str) -> str:
    """Make transformed feature names presentable in charts and reports."""
    cleaned = name.replace("num__", "").replace("cat__", "")
    for field, label in FEATURE_LABELS.items():
        if cleaned == field:
            return label
        prefix = f"{field}_"
        if cleaned.startswith(prefix):
            return f"{label}: {cleaned[len(prefix):]}"
    return cleaned.replace("_", " ").title()

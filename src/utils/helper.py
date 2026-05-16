"""
Shared utilities used across the StudentAI project.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

import joblib
import numpy as np

from src.utils.schema import FEATURE_LABELS, NUMERICAL_FEATURES


def get_logger(name: str = "StudentAnalytics") -> logging.Logger:
    """Return a logger that writes to both console and a daily log file."""
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"{datetime.now().strftime('%Y-%m-%d')}.log")

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s]  %(levelname)-8s  %(name)s  >>  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = get_logger()


def save_object(file_path: str, obj) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)
    logger.info(f"Artifact saved  -> {file_path}")


def load_object(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artifact not found: {file_path}")
    obj = joblib.load(file_path)
    logger.info(f"Artifact loaded <- {file_path}")
    return obj


def evaluate_model(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    r2 = round(float(r2_score(y_true, y_pred)), 4)
    mae = round(float(mean_absolute_error(y_true, y_pred)), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
    return {"r2": r2, "mae": mae, "rmse": rmse}


def get_grade(score: float) -> tuple:
    """Return (letter_grade, description, bootstrap_color)."""
    s = float(score)
    if s >= 90:
        return "A+", "Outstanding", "success"
    if s >= 80:
        return "A", "Excellent", "success"
    if s >= 70:
        return "B", "Good", "info"
    if s >= 60:
        return "C", "Average", "warning"
    if s >= 50:
        return "D", "Below Average", "warning"
    return "F", "Needs Improvement", "danger"


def get_risk_level(score: float) -> str:
    if score < 50:
        return "High"
    if score < 65:
        return "Medium"
    return "Low"


def get_confidence(score: float, rmse: float | None = None) -> float:
    """
    Heuristic confidence score. It combines grade-boundary distance with model
    error, so a high-RMSE model does not claim exaggerated certainty.
    """
    boundaries = [0, 50, 60, 70, 80, 90, 100]
    boundary_conf = 80.0
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        if lo <= score <= hi:
            mid = (lo + hi) / 2
            dist = abs(score - mid) / ((hi - lo) / 2 + 1e-9)
            boundary_conf = (1 - dist * 0.35) * 100
            break

    if rmse is None:
        return round(max(50.0, min(98.0, boundary_conf)), 1)

    error_penalty = min(28.0, max(0.0, float(rmse) * 2.2))
    return round(max(45.0, min(97.0, boundary_conf - error_penalty)), 1)


def get_prediction_interval(score: float, rmse: float | None = None) -> dict:
    margin = max(3.0, min(18.0, float(rmse or 5.0) * 1.96))
    return {
        "low": round(max(0.0, score - margin), 2),
        "high": round(min(100.0, score + margin), 2),
        "margin": round(margin, 2),
    }


def _importance_by_raw_feature(metrics: dict) -> dict:
    raw = {name: 0.0 for name in NUMERICAL_FEATURES}
    raw.update(
        {
            "gender": 0.0,
            "race_ethnicity": 0.0,
            "parental_education": 0.0,
            "lunch": 0.0,
            "test_prep_course": 0.0,
            "extracurricular": 0.0,
            "internet_access": 0.0,
        }
    )
    for item in metrics.get("feature_importance", []):
        feature = item.get("feature", "")
        importance = float(item.get("importance", 0.0))
        for name in raw:
            if feature == name or feature.startswith(f"{name}_"):
                raw[name] += importance
                break
    return raw


def build_prediction_insights(input_data: dict, score: float, metrics: dict) -> dict:
    """Return top drivers and practical recommendations for a prediction."""
    importance = _importance_by_raw_feature(metrics)
    drivers: list[dict] = []
    recommendations: list[str] = []

    def add_driver(field: str, impact: str, message: str, recommendation: str | None = None):
        drivers.append(
            {
                "feature": FEATURE_LABELS.get(field, field),
                "value": input_data.get(field),
                "impact": impact,
                "message": message,
                "importance": round(float(importance.get(field, 0.0)), 4),
            }
        )
        if recommendation and recommendation not in recommendations:
            recommendations.append(recommendation)

    attendance = float(input_data["attendance_rate"])
    study = float(input_data["study_hours_per_week"])
    current_scores = [
        float(input_data["math_score"]),
        float(input_data["reading_score"]),
        float(input_data["writing_score"]),
        float(input_data["science_score"]),
    ]
    current_avg = float(np.mean(current_scores))

    if attendance < 75:
        add_driver(
            "attendance_rate",
            "negative",
            "Attendance is below the stable-learning range.",
            "Create an attendance recovery plan and review missed lessons weekly.",
        )
    elif attendance >= 92:
        add_driver("attendance_rate", "positive", "Attendance is a strong support signal.")

    if study < 3:
        add_driver(
            "study_hours_per_week",
            "negative",
            "Study time is low for the expected final-score target.",
            "Increase focused study time to at least 5 hours per week.",
        )
    elif study >= 8:
        add_driver("study_hours_per_week", "positive", "Study time is above the class baseline.")

    if current_avg < 60:
        add_driver(
            "math_score",
            "negative",
            "Current assessment average is below the pass-safety band.",
            "Start remedial practice on the two weakest subjects before the next assessment.",
        )
    elif current_avg >= 80:
        add_driver("math_score", "positive", "Current assessment scores are already strong.")

    weakest = min(
        [
            ("math_score", float(input_data["math_score"])),
            ("reading_score", float(input_data["reading_score"])),
            ("writing_score", float(input_data["writing_score"])),
            ("science_score", float(input_data["science_score"])),
        ],
        key=lambda item: item[1],
    )
    if weakest[1] < 65:
        add_driver(
            weakest[0],
            "negative",
            f"{FEATURE_LABELS[weakest[0]]} is the weakest current score.",
            f"Prioritize {FEATURE_LABELS[weakest[0]].replace('Current ', '').lower()} practice first.",
        )

    if input_data["test_prep_course"] == "none":
        add_driver(
            "test_prep_course",
            "negative",
            "No test-prep course is recorded.",
            "Enroll in a structured test-prep or revision session.",
        )
    else:
        add_driver("test_prep_course", "positive", "Completed test prep supports the forecast.")

    if input_data["internet_access"] == "no":
        add_driver(
            "internet_access",
            "negative",
            "Limited internet access can reduce practice consistency.",
            "Provide offline material or scheduled lab access for digital practice.",
        )

    if input_data["lunch"] == "free/reduced":
        add_driver(
            "lunch",
            "support",
            "The profile indicates a possible support-needs segment.",
            "Check whether food, schedule, or mentoring support would remove learning friction.",
        )

    if input_data["extracurricular"] == "no" and score < 70:
        add_driver(
            "extracurricular",
            "support",
            "No extracurricular engagement is recorded.",
            "Pair academic support with a motivating activity or peer-learning group.",
        )

    if not recommendations:
        if score >= 80:
            recommendations.append("Maintain the current study rhythm and use enrichment practice.")
        else:
            recommendations.append("Review attendance, study schedule, and weakest subject weekly.")

    drivers.sort(
        key=lambda item: (
            0 if item["impact"] == "negative" else 1,
            -item["importance"],
        )
    )
    return {"drivers": drivers[:5], "recommendations": recommendations[:4]}

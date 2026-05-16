"""
Loads saved artifacts and predicts final_score for a new student.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.exception import StudentAnalyticsException
from src.utils.helper import (
    build_prediction_insights,
    get_confidence,
    get_grade,
    get_logger,
    get_prediction_interval,
    get_risk_level,
    load_object,
)
from src.utils.schema import ALL_FEATURES, InputValidationError, validate_student_input

logger = get_logger("PredictionPipeline")

ARTIFACTS = {
    "model": os.path.join("artifacts", "model.pkl"),
    "preprocessor": os.path.join("artifacts", "preprocessor.pkl"),
    "metrics": os.path.join("artifacts", "metrics.json"),
}


class PredictPipeline:
    """Loads model + preprocessor and returns a structured prediction."""

    def __init__(self):
        self.model = load_object(ARTIFACTS["model"])
        self.preprocessor = load_object(ARTIFACTS["preprocessor"])
        self.metrics = self._load_metrics()
        logger.info("PredictPipeline ready.")

    def _load_metrics(self) -> dict:
        if not os.path.exists(ARTIFACTS["metrics"]):
            return {}
        with open(ARTIFACTS["metrics"], encoding="utf-8") as f:
            return json.load(f)

    def _best_rmse(self) -> float | None:
        if self.metrics.get("best_rmse") is not None:
            return float(self.metrics["best_rmse"])
        best_model = self.metrics.get("best_model")
        all_models = self.metrics.get("all_models", {})
        if best_model in all_models:
            return float(all_models[best_model].get("rmse", 0))
        return None

    def predict(self, input_data: dict) -> dict:
        """
        Return predicted_score, grade, risk_level, confidence, interval,
        top drivers, recommendations, and model metadata.
        """
        try:
            cleaned = validate_student_input(input_data)
            df = pd.DataFrame([cleaned], columns=ALL_FEATURES)
            X = self.preprocessor.transform(df)
            raw_score = self.model.predict(X)[0]

            score = round(float(max(0, min(100, raw_score))), 2)
            rmse = self._best_rmse()
            grade, desc, color = get_grade(score)
            insights = build_prediction_insights(cleaned, score, self.metrics)

            return {
                "predicted_score": score,
                "grade": grade,
                "grade_desc": desc,
                "grade_color": color,
                "risk_level": get_risk_level(score),
                "confidence": get_confidence(score, rmse),
                "interval": get_prediction_interval(score, rmse),
                "drivers": insights["drivers"],
                "recommendations": insights["recommendations"],
                "model": {
                    "name": self.metrics.get("best_model", self.model.__class__.__name__),
                    "version": self.metrics.get("model_version", "unknown"),
                    "trained_at": self.metrics.get("trained_at"),
                    "target": self.metrics.get("target", "final_score"),
                },
            }

        except InputValidationError:
            raise
        except Exception as e:
            raise StudentAnalyticsException(e, sys)

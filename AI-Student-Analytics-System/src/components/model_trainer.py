"""
Step 3: train multiple models, compare them, save the winner, and write
metadata used by the dashboard and prediction explanations.
"""
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.exception import StudentAnalyticsException
from src.utils.helper import evaluate_model, get_logger, save_object
from src.utils.schema import TARGET, display_feature_name, feature_schema_payload

warnings.filterwarnings("ignore")
logger = get_logger("ModelTrainer")


class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    metrics_path: str = os.path.join("artifacts", "metrics.json")


class ModelTrainer:
    """Trains and compares multiple regression models."""

    def __init__(self):
        self.config = ModelTrainerConfig()

    def _get_models(self) -> dict:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import LinearRegression, Ridge

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=180,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=3,
            ),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }

        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(
                n_estimators=220,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )
        except ImportError:
            logger.warning("XGBoost not installed - skipping.")

        return models

    def _cross_validate(self, model, X, y, cv: int = 5) -> float:
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        return round(float(scores.mean()), 4)

    def _feature_importance(self, model, feature_names: list[str] | None) -> list[dict]:
        if not feature_names:
            return []

        raw_values = None
        signed_values = None
        if hasattr(model, "feature_importances_"):
            raw_values = np.asarray(model.feature_importances_, dtype=float)
            signed_values = raw_values.copy()
        elif hasattr(model, "coef_"):
            signed_values = np.ravel(np.asarray(model.coef_, dtype=float))
            raw_values = np.abs(signed_values)

        if raw_values is None or len(raw_values) != len(feature_names):
            return []

        total = float(np.sum(np.abs(raw_values))) or 1.0
        items = []
        for name, value, signed in zip(feature_names, raw_values, signed_values):
            items.append(
                {
                    "feature": name,
                    "label": display_feature_name(name),
                    "importance": round(float(abs(value) / total), 4),
                    "weight": round(float(signed), 4),
                }
            )

        items.sort(key=lambda item: item["importance"], reverse=True)
        return items[:20]

    def initiate(self, X_train, X_test, y_train, y_test, feature_names=None) -> dict:
        """
        Train all models, cross-validate them, pick the best by test R2, save
        the best model, and return a metadata-rich metrics payload.
        """
        logger.info("Model Training started.")
        try:
            models = self._get_models()
            all_metrics = {}

            for name, model in models.items():
                logger.info(f"Training: {name}")
                model.fit(X_train, y_train)

                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                train_m = evaluate_model(y_train, train_pred)
                test_m = evaluate_model(y_test, test_pred)
                cv_r2 = self._cross_validate(model, X_train, y_train)

                all_metrics[name] = {
                    "train_r2": train_m["r2"],
                    "test_r2": test_m["r2"],
                    "cv_r2": cv_r2,
                    "mae": test_m["mae"],
                    "rmse": test_m["rmse"],
                    "accuracy": round(max(test_m["r2"], 0) * 100, 2),
                }
                logger.info(
                    f"  {name}: test_R2={test_m['r2']}  "
                    f"MAE={test_m['mae']}  RMSE={test_m['rmse']}  CV_R2={cv_r2}"
                )

            best_name = max(all_metrics, key=lambda n: all_metrics[n]["test_r2"])
            best_model = models[best_name]
            best_score = all_metrics[best_name]["test_r2"]
            best_rmse = all_metrics[best_name]["rmse"]

            logger.info(f"Best model: {best_name} (R2 = {best_score})")
            if best_score < 0.6:
                logger.warning("Best R2 is below 0.60 - consider adding better features.")

            save_object(self.config.model_path, best_model)

            os.makedirs("artifacts", exist_ok=True)
            trained_at = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            metrics_payload = {
                "target": TARGET,
                "trained_at": trained_at,
                "model_version": trained_at.replace(":", "").replace("-", ""),
                "best_model": best_name,
                "best_r2": best_score,
                "best_rmse": best_rmse,
                "feature_schema": feature_schema_payload(),
                "feature_names": feature_names or [],
                "feature_importance": self._feature_importance(best_model, feature_names),
                "all_models": all_metrics,
            }
            with open(self.config.metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, indent=4)
            logger.info("Metrics saved to artifacts/metrics.json")

            return metrics_payload

        except Exception as e:
            raise StudentAnalyticsException(e, sys)

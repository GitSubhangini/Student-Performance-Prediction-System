"""
Step 2 of the ML pipeline: build and save the preprocessing pipeline.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.exception import StudentAnalyticsException
from src.utils.helper import get_logger, save_object
from src.utils.schema import (
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET,
)

logger = get_logger("DataTransformation")


class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """Fits and applies the sklearn preprocessing pipeline."""

    def __init__(self):
        self.config = DataTransformationConfig()

    def _build_preprocessor(self) -> ColumnTransformer:
        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", num_pipeline, NUMERICAL_FEATURES),
                ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ]
        )

    def initiate(self, train_path: str, test_path: str) -> tuple:
        """
        Load train/test CSVs, fit the preprocessor on training data, transform
        both sets, and return transformed arrays plus feature names.
        """
        logger.info("Data Transformation started.")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            for df in [train_df, test_df]:
                for col in DROP_COLUMNS:
                    if col in df.columns:
                        df.drop(columns=[col], inplace=True)

            X_train = train_df.drop(columns=[TARGET])
            y_train = train_df[TARGET]
            X_test = test_df.drop(columns=[TARGET])
            y_test = test_df[TARGET]

            preprocessor = self._build_preprocessor()
            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t = preprocessor.transform(X_test)
            feature_names = [
                name.replace("num__", "").replace("cat__", "")
                for name in preprocessor.get_feature_names_out()
            ]

            logger.info(f"X_train shape after transform: {X_train_t.shape}")
            logger.info(f"X_test  shape after transform: {X_test_t.shape}")

            save_object(self.config.preprocessor_path, preprocessor)

            return (
                np.array(X_train_t),
                np.array(X_test_t),
                np.array(y_train),
                np.array(y_test),
                feature_names,
            )

        except Exception as e:
            raise StudentAnalyticsException(e, sys)

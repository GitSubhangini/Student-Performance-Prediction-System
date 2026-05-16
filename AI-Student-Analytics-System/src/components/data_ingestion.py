"""
src/components/data_ingestion.py
=================================
Step 1 of the ML pipeline: Load raw CSV, validate it,
split into train/test sets, and save them.

Beginner Note:
  "Data ingestion" simply means reading the raw data and
  preparing it for the next steps (cleaning, modelling).
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path so we can import src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.exception import StudentAnalyticsException
from src.utils.helper import get_logger, get_risk_level
from src.utils.schema import ALL_FEATURES, TARGET

logger = get_logger("DataIngestion")

# Columns we need to train the model.
REQUIRED_COLUMNS = ALL_FEATURES + [TARGET]


class DataIngestionConfig:
    """Holds all file-path configuration for the ingestion step."""
    raw_data_path: str   = os.path.join("data",      "student_data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str  = os.path.join("artifacts", "test.csv")


class DataIngestion:
    """Reads raw CSV and produces train / test CSVs in artifacts/."""

    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate(self) -> tuple:
        """
        Returns (train_path, test_path).
        Raises StudentAnalyticsException on any failure.
        """
        logger.info("Data Ingestion started.")
        try:
            # ── Load ─────────────────────────────────────────────────
            if not os.path.exists(self.config.raw_data_path):
                raise FileNotFoundError(
                    f"Dataset not found at {self.config.raw_data_path}. "
                    "Run  python generate_data.py  first."
                )
            df = pd.read_csv(self.config.raw_data_path)
            logger.info(f"Raw dataset loaded: {df.shape[0]} rows x {df.shape[1]} cols")

            # ── Validate ─────────────────────────────────────────────
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            # Drop rows where target is null
            df.dropna(subset=[TARGET], inplace=True)
            if "risk_level" not in df.columns:
                df["risk_level"] = df[TARGET].apply(get_risk_level)
            logger.info(f"After validation: {df.shape[0]} rows")

            # ── Split ─────────────────────────────────────────────────
            train, test = train_test_split(df, test_size=0.2, random_state=42)
            logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows")

            # ── Save ──────────────────────────────────────────────────
            os.makedirs("artifacts", exist_ok=True)
            train.to_csv(self.config.train_data_path, index=False)
            test.to_csv(self.config.test_data_path,   index=False)
            logger.info("Train / Test CSVs saved to artifacts/")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise StudentAnalyticsException(e, sys)

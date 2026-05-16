"""
src/pipeline/train_pipeline.py
================================
Orchestrates the full training pipeline:
  DataIngestion -> DataTransformation -> ModelTrainer

Run from project root:
  python src/pipeline/train_pipeline.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.components.data_ingestion      import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer       import ModelTrainer
from src.utils.helper                   import get_logger
from src.exception                      import StudentAnalyticsException

logger = get_logger("TrainPipeline")


def run_training() -> dict:
    """Execute the full pipeline and return metrics."""
    logger.info("=" * 55)
    logger.info("  TRAINING PIPELINE STARTED")
    logger.info("=" * 55)
    try:
        # Step 1 — Ingest
        ingestion          = DataIngestion()
        train_path, test_path = ingestion.initiate()

        # Step 2 — Transform
        transformation     = DataTransformation()
        X_train, X_test, y_train, y_test, feature_names = transformation.initiate(
            train_path, test_path
        )

        # Step 3 — Train
        trainer            = ModelTrainer()
        metrics            = trainer.initiate(
            X_train, X_test, y_train, y_test, feature_names
        )

        logger.info("=" * 55)
        logger.info("  TRAINING PIPELINE COMPLETE")
        logger.info(f"  Best Model : {metrics['best_model']}")
        logger.info(f"  Best R2    : {metrics['best_r2']}")
        logger.info("=" * 55)

        return metrics

    except Exception as e:
        raise StudentAnalyticsException(e, sys)


if __name__ == "__main__":
    results = run_training()
    print("\nAll model metrics:")
    for name, m in results["all_models"].items():
        print(f"  {name:20s}  R2={m['test_r2']}  MAE={m['mae']}  RMSE={m['rmse']}")

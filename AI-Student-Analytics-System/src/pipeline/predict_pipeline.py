"""
src/pipeline/predict_pipeline.py
=================================
Wraps PredictPipeline for easy use from the Flask app.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.components.prediction_pipeline import PredictPipeline
from src.utils.helper                   import get_logger

logger = get_logger("PredictPipelineWrapper")

_pipeline = None  # singleton — loaded once

def get_pipeline() -> PredictPipeline:
    """Return a singleton PredictPipeline (loaded once per process)."""
    global _pipeline
    if _pipeline is None:
        logger.info("Loading prediction pipeline (first call).")
        _pipeline = PredictPipeline()
    return _pipeline


def predict(input_data: dict) -> dict:
    """Convenience function: predict from a dict of feature values."""
    pipe = get_pipeline()
    return pipe.predict(input_data)

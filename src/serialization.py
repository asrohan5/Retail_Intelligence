"""Persist and load production model artifacts."""

import logging
import os

import joblib

from src import config


from logger import logging as logger


def save_production_artifacts(model, feature_columns, metadata: dict | None = None) -> None:
    """Save the fitted model (preprocessing included) with its expected feature order."""
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    artifacts = {
        "model": model,
        "features": list(feature_columns),
        "metadata": {"version": config.MODEL_VERSION, **(metadata or {})},
    }
    joblib.dump(artifacts, config.MODEL_PATH)
    logger.info("Model artifacts saved to %s", config.MODEL_PATH)


def load_production_model() -> dict:
    """Load the production artifact bundle.

    Raises:
        FileNotFoundError: if no trained model exists at `config.MODEL_PATH`.
    """
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"No production model found at {config.MODEL_PATH}")
    logger.info("Loading model from %s", config.MODEL_PATH)
    return joblib.load(config.MODEL_PATH)

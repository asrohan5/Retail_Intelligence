"""FastAPI service exposing the customer-churn model."""

import logging
from contextlib import asynccontextmanager

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.serialization import load_production_model

from logging import logging as logger

model = None
required_features: list[str] = []
model_version: str = config.MODEL_VERSION


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model into memory once, at process startup."""
    global model, required_features, model_version
    try:
        artifacts = load_production_model()
        model = artifacts["model"]
        required_features = artifacts["features"]
        model_version = artifacts.get("metadata", {}).get("version", config.MODEL_VERSION)
        logger.info("Model %s loaded successfully", model_version)
    except Exception as e:
        logger.error("Model failed to load: %s", e)
        model = None
    yield


app = FastAPI(title="Retail Churn API", version="1.1", lifespan=lifespan)


class CustomerState(BaseModel):
    """Feature payload for a churn prediction. Ranges reject impossible values with a 422."""

    recency: float = Field(ge=0, description="Days since last order")
    frequency: float = Field(ge=0, description="Number of distinct invoices")
    monetary: float = Field(description="Total spend")
    spend_velocity: float = Field(description="log1p(spend last 30d) - log1p(spend prior 30d)")
    purchase_interval_std: float = Field(ge=0, description="Std of days between invoices")
    unique_products: int = Field(ge=0)
    unique_categories: int = Field(ge=0)
    weekend_ratio: float = Field(ge=0, le=1)
    avg_unit_price: float = Field(ge=0)
    avg_hour: float = Field(ge=0, le=24)
    whale_score: float = Field(ge=0, description="Avg line quantity / population avg")
    return_rate: float = Field(ge=0, le=1)


@app.post("/predict_churn")
def predict(customer: CustomerState):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        input_data = pd.DataFrame([customer.model_dump()])[required_features]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability),
        "risk_level": "CRITICAL" if probability >= config.RISK_THRESHOLD_CRITICAL else "Normal",
    }


@app.get("/health")
def health_check():
    return {"status": "active", "model_version": model_version, "model_loaded": model is not None}


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

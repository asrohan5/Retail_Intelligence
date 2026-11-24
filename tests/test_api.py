"""API tests. The model is mocked (see conftest.py), so no trained artifact is needed."""

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID_PAYLOAD = {
    "recency": 30,
    "frequency": 5,
    "monetary": 100,
    "spend_velocity": 0.4,
    "purchase_interval_std": 5,
    "unique_products": 10,
    "unique_categories": 3,
    "weekend_ratio": 0.2,
    "avg_unit_price": 20,
    "whale_score": 0.1,
    "return_rate": 0.05,
    "avg_hour": 12,
}


def test_health_check_reports_model_status(mock_model):
    body = client.get("/health").json()
    assert body["status"] == "active"
    assert body["model_loaded"] is True


def test_predict_churn_valid_payload(mock_model):
    response = client.post("/predict_churn", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] in (0, 1)
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] == "Normal"


def test_high_probability_is_flagged_critical(high_risk_model):
    body = client.post("/predict_churn", json=VALID_PAYLOAD).json()
    assert body["risk_level"] == "CRITICAL"


def test_missing_fields_return_422(mock_model):
    assert client.post("/predict_churn", json={"recency": 30}).status_code == 422


def test_out_of_range_values_return_422(mock_model):
    bad = {**VALID_PAYLOAD, "weekend_ratio": 1.5}
    assert client.post("/predict_churn", json=bad).status_code == 422
    bad = {**VALID_PAYLOAD, "recency": -3}
    assert client.post("/predict_churn", json=bad).status_code == 422


def test_predict_without_model_returns_503(no_model):
    assert client.post("/predict_churn", json=VALID_PAYLOAD).status_code == 503

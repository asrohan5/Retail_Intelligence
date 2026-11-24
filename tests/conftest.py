"""Shared pytest fixtures.

`mock_model` swaps in a dummy classifier so API tests exercise the real FastAPI routes
without needing a trained artifact on disk (this is what lets CI run with no training step).
"""

import numpy as np
import pytest

import src.api as api_module
from src.features import MODEL_FEATURES


class _DummyModel:
    def __init__(self, p_churn: float = 0.2):
        self.p = p_churn

    def predict(self, X):
        return np.array([int(self.p >= 0.5)])

    def predict_proba(self, X):
        assert list(X.columns) == MODEL_FEATURES, "API must pass features in training order"
        return np.array([[1 - self.p, self.p]])


@pytest.fixture
def mock_model(monkeypatch):
    monkeypatch.setattr(api_module, "model", _DummyModel(0.2))
    monkeypatch.setattr(api_module, "required_features", MODEL_FEATURES)


@pytest.fixture
def high_risk_model(monkeypatch):
    monkeypatch.setattr(api_module, "model", _DummyModel(0.9))
    monkeypatch.setattr(api_module, "required_features", MODEL_FEATURES)


@pytest.fixture
def no_model(monkeypatch):
    monkeypatch.setattr(api_module, "model", None)


@pytest.fixture(scope="session")
def transactions():
    from tests.synthetic import make_transactions

    return make_transactions()

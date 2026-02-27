"""Tests for FastAPI serving endpoint."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.features.ngram import NgramTokenizer
from src.serving.app import _state, app


@pytest.fixture
def trained_state():
    """Set up app state with a trained toy model."""
    # Fit a tiny tokenizer + model
    domains = (
        ["google.com", "github.com", "amazon.com"] * 20
        + ["evil123abc456.xyz", "a1b2c3d4e5.net"] * 15
        + ["aGVsbG8.evil.com", "4a6f686e.data.io"] * 10
    )
    labels = [0] * 60 + [1] * 30 + [2] * 20

    tok = NgramTokenizer(ngram_range=(2, 3), max_features=50, min_df=1)

    from src.training.train import build_features
    cfg = {"features": {"statistical": {"entropy": True, "length": True, "char_distribution": True, "subdomain_stats": True}}}
    X = build_features(domains, tok, cfg, fit=True)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X.toarray(), np.array(labels))

    _state["model"] = model
    _state["tokenizer"] = tok
    _state["config"] = cfg
    _state["start_time"] = 1000000.0

    yield
    _state.clear()


@pytest.fixture
def client(trained_state):
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def empty_client():
    _state["model"] = None
    _state["tokenizer"] = None
    _state["config"] = {}
    _state["start_time"] = 1000000.0
    yield TestClient(app, raise_server_exceptions=False)
    _state.clear()


class TestHealth:
    def test_healthy_with_model(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_degraded_without_model(self, empty_client):
        resp = empty_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


class TestPredict:
    def test_single_prediction(self, client):
        resp = client.post("/predict", json={"domains": ["google.com"]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 1
        pred = data["predictions"][0]
        assert pred["label"] in [0, 1, 2]
        assert pred["label_name"] in ["benign", "dga", "exfiltration"]
        assert pred["probabilities"] is not None

    def test_batch_prediction(self, client):
        domains = ["google.com", "evil123.xyz", "aGVsbG8.evil.com"]
        resp = client.post("/predict", json={"domains": domains})
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 3

    def test_no_model_503(self, empty_client):
        resp = empty_client.post("/predict", json={"domains": ["test.com"]})
        assert resp.status_code == 503

    def test_empty_domains_422(self, client):
        resp = client.post("/predict", json={"domains": []})
        assert resp.status_code == 422

    def test_inference_time_reported(self, client):
        resp = client.post("/predict", json={"domains": ["test.com"]})
        data = resp.json()
        assert "inference_time_ms" in data
        assert data["inference_time_ms"] >= 0


class TestModelInfo:
    def test_model_info(self, client):
        resp = client.get("/model/info")
        data = resp.json()
        assert data["loaded"] is True
        assert data["model_type"] == "RandomForestClassifier"
        assert "classes" in data

    def test_model_info_no_model(self, empty_client):
        resp = empty_client.get("/model/info")
        data = resp.json()
        assert data["loaded"] is False

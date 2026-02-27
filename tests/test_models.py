"""Tests for model factory and class weighting."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.training.models import LABEL_MAP, compute_class_weights, create_model


class TestCreateModel:
    def test_xgboost_default(self):
        model = create_model("xgboost")
        assert isinstance(model, XGBClassifier)

    def test_random_forest(self):
        model = create_model("random_forest")
        assert isinstance(model, RandomForestClassifier)

    def test_logistic_regression(self):
        model = create_model("logistic_regression")
        assert isinstance(model, LogisticRegression)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("deep_learning")

    def test_custom_params(self):
        model = create_model("xgboost", {"n_estimators": 50, "max_depth": 3})
        assert model.get_params()["n_estimators"] == 50
        assert model.get_params()["max_depth"] == 3

    def test_class_weights_rf(self):
        weights = {0: 1.0, 1: 2.0, 2: 3.0}
        model = create_model("random_forest", {}, weights)
        assert model.get_params()["class_weight"] == weights

    def test_class_weights_lr(self):
        weights = {0: 1.0, 1: 2.0, 2: 3.0}
        model = create_model("logistic_regression", {}, weights)
        assert model.get_params()["class_weight"] == weights


class TestComputeClassWeights:
    def test_balanced(self):
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        weights = compute_class_weights(y)
        # All classes equal → all weights ~1.0
        assert abs(weights[0] - 1.0) < 1e-6

    def test_imbalanced(self):
        y = np.array([0] * 80 + [1] * 15 + [2] * 5)
        weights = compute_class_weights(y)
        assert weights[2] > weights[1] > weights[0]


class TestFitPredict:
    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        y = rng.choice([0, 1, 2], size=200, p=[0.6, 0.25, 0.15])
        return X, y

    def test_xgboost_fit_predict(self, synthetic_data):
        X, y = synthetic_data
        model = create_model("xgboost", {"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1, 2})

    def test_rf_fit_predict(self, synthetic_data):
        X, y = synthetic_data
        model = create_model("random_forest", {"n_estimators": 10})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_lr_fit_predict(self, synthetic_data):
        X, y = synthetic_data
        model = create_model("logistic_regression")
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestLabelMap:
    def test_three_classes(self):
        assert len(LABEL_MAP) == 3
        assert LABEL_MAP[0] == "benign"
        assert LABEL_MAP[1] == "dga"
        assert LABEL_MAP[2] == "exfiltration"

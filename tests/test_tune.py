"""Tests for Optuna hyperparameter tuning module."""

from __future__ import annotations

from src.training.tune import SEARCH_SPACES, _suggest_param


class TestSearchSpaces:
    """Verify search space definitions for all model types."""

    def test_xgboost_space_defined(self):
        assert "xgboost" in SEARCH_SPACES
        assert len(SEARCH_SPACES["xgboost"]) >= 5

    def test_random_forest_space_defined(self):
        assert "random_forest" in SEARCH_SPACES
        assert len(SEARCH_SPACES["random_forest"]) >= 3

    def test_logistic_regression_space_defined(self):
        assert "logistic_regression" in SEARCH_SPACES
        assert len(SEARCH_SPACES["logistic_regression"]) >= 1

    def test_each_space_has_tuples(self):
        """Each search space entry should be a tuple of (type, arg1, arg2...)."""
        for model_type, space in SEARCH_SPACES.items():
            for key, spec in space.items():
                assert isinstance(spec, tuple), f"{model_type}.{key} is not a tuple"
                assert len(spec) >= 2, f"{model_type}.{key} too short"
                assert spec[0] in ("int", "float", "float_log", "categorical")


class TestSuggestParam:
    """Test the _suggest_param helper with a mock trial."""

    class MockTrial:
        """Minimal Optuna trial mock."""

        def suggest_int(self, name, low, high, **kw):
            return low

        def suggest_float(self, name, low, high, **kw):
            return low

        def suggest_categorical(self, name, choices):
            return choices[0]

    def test_int_param(self):
        trial = self.MockTrial()
        val = _suggest_param(trial, "n_est", ("int", 50, 500))
        assert val == 50

    def test_float_param(self):
        trial = self.MockTrial()
        val = _suggest_param(trial, "lr", ("float", 0.01, 1.0))
        assert val == 0.01

    def test_categorical_param(self):
        trial = self.MockTrial()
        val = _suggest_param(trial, "solver", ("categorical", ["lbfgs", "saga"]))
        assert val == "lbfgs"

    def test_loguniform_float(self):
        trial = self.MockTrial()
        val = _suggest_param(trial, "lr", ("float_log", 0.001, 1.0))
        assert val == 0.001

"""Tests for training pipeline helpers."""

import numpy as np
import pytest

from src.features.ngram import NgramTokenizer
from src.training.train import build_features, compute_metrics


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_metric_keys(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_benign" in metrics
        assert "f1_dga" in metrics
        assert "f1_exfiltration" in metrics


class TestBuildFeatures:
    def test_returns_sparse_matrix(self):
        domains = ["google.com", "evil.xyz", "aGVsbG8.evil.com"] * 10
        tok = NgramTokenizer(ngram_range=(2, 3), max_features=50, min_df=1)
        cfg = {"features": {"statistical": {"entropy": True, "length": True, "char_distribution": True, "subdomain_stats": True}}}
        X = build_features(domains, tok, cfg, fit=True)

        from scipy.sparse import issparse
        assert issparse(X)
        assert X.shape[0] == len(domains)
        # Should have n-gram features + statistical features
        assert X.shape[1] > 10

    def test_transform_after_fit(self):
        train = ["google.com", "github.com", "evil.xyz"] * 10
        test = ["new-domain.io"]
        tok = NgramTokenizer(ngram_range=(2, 3), max_features=50, min_df=1)
        cfg = {"features": {"statistical": {}}}

        X_train = build_features(train, tok, cfg, fit=True)
        X_test = build_features(test, tok, cfg, fit=False)

        assert X_train.shape[1] == X_test.shape[1]
        assert X_test.shape[0] == 1

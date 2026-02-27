"""Evaluation pipeline — test set metrics and classification report.

Loads the trained model + vectorizer, evaluates on the held-out test
split, and saves metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.features.statistical import compute_statistical_features
from src.training.models import LABEL_MAP
from src.training.train import build_features, compute_metrics

logger = logging.getLogger(__name__)


def evaluate_pipeline(cfg: dict) -> dict:
    """Evaluate the trained model on the test split.

    Returns
    -------
    dict
        Test set metrics (prefixed with ``test_``).
    """
    data_cfg = cfg.get("data", {})
    mlflow_cfg = cfg.get("mlflow", {})

    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    registry_dir = Path(mlflow_cfg.get("model_registry", "models/registry"))

    # Load
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    model = joblib.load(registry_dir / "model.pkl")
    tokenizer = joblib.load(registry_dir / "vectorizer.pkl")

    # Features
    test_domains = test_df["domain"].tolist()
    y_test = test_df["label"].values
    X_test = build_features(test_domains, tokenizer, cfg, fit=False)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)

    # Metrics
    metrics = compute_metrics(y_test, y_pred)
    test_metrics = {f"test_{k}": v for k, v in metrics.items()}

    # Per-class report
    from sklearn.metrics import classification_report

    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    report = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0, output_dict=True,
    )
    test_metrics["classification_report"] = report

    # Save
    with open(registry_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2, default=str)

    logger.info(
        "Test: accuracy=%.4f, F1_macro=%.4f",
        test_metrics["test_accuracy"],
        test_metrics["test_f1_macro"],
    )

    return test_metrics

"""Training pipeline with MLflow experiment tracking.

Loads split data, extracts n-gram + statistical features, trains a
classifier, and logs everything to MLflow.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.features.ngram import NgramTokenizer
from src.features.statistical import compute_statistical_features
from src.training.models import LABEL_MAP, compute_class_weights, create_model

logger = logging.getLogger(__name__)


# ── Feature extraction helpers ──────────────────────────────────────


def build_features(
    domains: list[str],
    tokenizer: NgramTokenizer,
    cfg: dict,
    *,
    fit: bool = False,
) -> csr_matrix:
    """Build combined n-gram + statistical feature matrix.

    Parameters
    ----------
    domains : list[str]
        Raw domain strings.
    tokenizer : NgramTokenizer
        N-gram tokenizer (fitted or to be fitted).
    cfg : dict
        Feature config section.
    fit : bool
        If True, fit the tokenizer on *domains*.

    Returns
    -------
    csr_matrix
        Combined sparse feature matrix.
    """
    # N-gram TF-IDF features
    ngram_matrix = tokenizer.fit_transform(domains) if fit else tokenizer.transform(domains)

    # Statistical features
    stat_cfg = cfg.get("features", {}).get("statistical", {})
    stat_df = compute_statistical_features(
        domains,
        entropy=stat_cfg.get("entropy", True),
        length=stat_cfg.get("length", True),
        char_dist=stat_cfg.get("char_distribution", True),
        subdomain=stat_cfg.get("subdomain_stats", True),
    )
    stat_matrix = csr_matrix(stat_df.values)

    # Combine
    return hstack([ngram_matrix, stat_matrix], format="csr")


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None,
) -> dict[str, float]:
    """Compute multiclass classification metrics."""
    labels = labels or sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        **{
            f"f1_{LABEL_MAP.get(lbl, str(lbl))}": float(
                f1_score(y_true, y_pred, labels=[lbl], average="micro", zero_division=0)
            )
            for lbl in labels
        },
    }


# ── Training pipeline ──────────────────────────────────────────────


def train_pipeline(cfg: dict) -> dict:
    """Run the full training pipeline.

    1. Load splits
    2. Extract n-gram + statistical features
    3. Train classifier
    4. Evaluate on val set
    5. Save model + vectorizer
    6. Log to MLflow (if available)

    Returns
    -------
    dict
        Validation metrics.
    """
    data_cfg = cfg.get("data", {})
    feat_cfg = cfg.get("features", {})
    model_cfg = cfg.get("model", {})
    mlflow_cfg = cfg.get("mlflow", {})

    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    registry_dir = Path(mlflow_cfg.get("model_registry", "models/registry"))
    registry_dir.mkdir(parents=True, exist_ok=True)

    # ── Load splits ─────────────────────────────────────────────────
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")

    train_domains = train_df["domain"].tolist()
    val_domains = val_df["domain"].tolist()
    y_train = train_df["label"].values
    y_val = val_df["label"].values

    # ── Build features ──────────────────────────────────────────────
    ngram_range = tuple(feat_cfg.get("ngram_range", [2, 4]))
    tokenizer = NgramTokenizer(
        ngram_range=ngram_range,
        max_features=feat_cfg.get("max_features", 5000),
        min_df=feat_cfg.get("min_df", 2),
        max_df=feat_cfg.get("max_df", 0.95),
        sublinear_tf=feat_cfg.get("sublinear_tf", True),
    )

    logger.info("Extracting features (n-gram range: %s)", ngram_range)
    t0 = time.time()
    X_train = build_features(train_domains, tokenizer, cfg, fit=True)
    X_val = build_features(val_domains, tokenizer, cfg, fit=False)
    feat_time = time.time() - t0
    logger.info("Features: %d train, %d val (%d cols, %.1fs)", X_train.shape[0], X_val.shape[0], X_train.shape[1], feat_time)

    # ── Train model ─────────────────────────────────────────────────
    class_weights = compute_class_weights(y_train)
    model = create_model(model_cfg.get("type", "xgboost"), model_cfg.get("params", {}), class_weights)

    logger.info("Training %s model...", model_cfg.get("type", "xgboost"))
    t0 = time.time()

    if model_cfg.get("type") == "xgboost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    train_time = time.time() - t0
    logger.info("Training completed in %.1fs", train_time)

    # ── Evaluate ────────────────────────────────────────────────────
    y_val_pred = model.predict(X_val)
    metrics = compute_metrics(y_val, y_val_pred)
    metrics["train_time_sec"] = train_time
    metrics["feature_time_sec"] = feat_time
    metrics["n_features"] = X_train.shape[1]
    metrics["n_ngram_features"] = tokenizer.n_features

    logger.info("Val metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, float)})

    # ── Save artifacts ──────────────────────────────────────────────
    joblib.dump(model, registry_dir / "model.pkl")
    joblib.dump(tokenizer, registry_dir / "vectorizer.pkl")

    with open(registry_dir / "val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── MLflow logging (optional) ───────────────────────────────────
    try:
        import mlflow

        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "seqthreat-dns"))

        with mlflow.start_run(run_name=f"train-{model_cfg.get('type', 'xgboost')}"):
            mlflow.log_params({
                "model_type": model_cfg.get("type"),
                "ngram_range": str(ngram_range),
                "max_features": feat_cfg.get("max_features"),
                **{f"model_{k}": v for k, v in model_cfg.get("params", {}).items()},
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
    except ImportError:
        logger.warning("MLflow not installed — skipping experiment tracking")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)

    return metrics

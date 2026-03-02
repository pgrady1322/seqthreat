"""Model factory for DNS threat classifiers.

Supports XGBoost, Random Forest, and Logistic Regression with
automatic class-weight balancing for the multiclass setting.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Label mapping
LABEL_MAP = {0: "benign", 1: "dga", 2: "exfiltration"}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute inverse-frequency class weights for imbalanced data."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts, strict=True)}
    return weights


def create_model(
    model_type: str,
    params: dict[str, Any] | None = None,
    class_weights: dict[int, float] | None = None,
) -> Any:
    """Create a classifier instance.

    Parameters
    ----------
    model_type : str
        One of ``"xgboost"``, ``"random_forest"``, ``"logistic_regression"``.
    params : dict
        Model hyperparameters (passed to constructor).
    class_weights : dict
        Optional class weight mapping ``{label: weight}``.

    Returns
    -------
    Fitted sklearn-compatible classifier.
    """
    params = dict(params or {})

    if model_type == "xgboost":
        # XGBoost handles multiclass via softmax by default
        params.setdefault("objective", "multi:softmax")
        params.setdefault("num_class", 3)
        params.setdefault("n_estimators", 200)
        params.setdefault("use_label_encoder", False)
        if class_weights:
            params["sample_weight"] = None  # handled at fit time
        return XGBClassifier(**params)

    elif model_type == "random_forest":
        params.setdefault("n_estimators", 100)
        params.setdefault("n_jobs", -1)
        if class_weights:
            params["class_weight"] = class_weights
        return RandomForestClassifier(**params)

    elif model_type == "logistic_regression":
        params.setdefault("max_iter", 1000)
        params.setdefault("solver", "lbfgs")
        if class_weights:
            params["class_weight"] = class_weights
        return LogisticRegression(**params)

    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: xgboost, random_forest, logistic_regression"
        )

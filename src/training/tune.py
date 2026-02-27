"""Optuna hyperparameter tuning with MLflow nested run tracking.

Bayesian optimization over n-gram parameters, statistical feature
flags, and model hyperparameters using TPE sampler with stratified
K-fold cross-validation.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.features.ngram import NgramTokenizer
from src.training.models import compute_class_weights, create_model
from src.training.train import build_features, compute_metrics

logger = logging.getLogger(__name__)

# ── Search spaces ───────────────────────────────────────────────────

SEARCH_SPACES = {
    "xgboost": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 3, 12),
        "learning_rate": ("float_log", 0.005, 0.3),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "min_child_weight": ("int", 1, 10),
        "gamma": ("float_log", 1e-4, 10.0),
        "reg_alpha": ("float_log", 1e-4, 10.0),
        "reg_lambda": ("float_log", 1e-4, 10.0),
    },
    "random_forest": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 3, 30),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
        "max_features": ("categorical", ["sqrt", "log2", None]),
    },
    "logistic_regression": {
        "C": ("float_log", 0.001, 100.0),
        "max_iter": ("categorical", [500, 1000, 2000]),
    },
}


def _suggest_param(trial, name: str, spec: tuple):
    """Suggest a parameter value from an Optuna trial."""
    kind = spec[0]
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    elif kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    elif kind == "float_log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"Unknown param type: {kind}")


# ── Objective ───────────────────────────────────────────────────────


def make_objective(
    X, y, model_type: str, n_splits: int = 5, seed: int = 42, metric: str = "f1_macro",
):
    """Create an Optuna objective function.

    Uses stratified K-fold CV on the combined n-gram + statistical
    feature matrix.
    """
    space = SEARCH_SPACES.get(model_type, {})
    class_weights = compute_class_weights(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial):
        # Also tune n-gram range (applied earlier, so we use the
        # pre-built matrix here and only tune the classifier)
        params = {name: _suggest_param(trial, name, spec) for name, spec in space.items()}

        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            model = create_model(model_type, params, class_weights)
            if model_type == "xgboost":
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            else:
                model.fit(X_tr, y_tr)

            y_pred = model.predict(X_va)
            metrics = compute_metrics(y_va, y_pred)
            scores.append(metrics[metric])

        mean_score = float(np.mean(scores))

        # Log to MLflow if available
        try:
            import mlflow

            with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
                mlflow.log_params(params)
                mlflow.log_metric(f"cv_{metric}", mean_score)
                mlflow.log_metric("cv_std", float(np.std(scores)))
        except Exception:
            pass

        return mean_score

    return objective


# ── Pipeline entry point ────────────────────────────────────────────


def tune_pipeline(cfg: dict) -> dict:
    """Run Optuna hyperparameter search.

    1. Load training data
    2. Build feature matrix
    3. Run Bayesian HP search with CV
    4. Save best params and history

    Returns
    -------
    dict
        Best params and best metric value.
    """
    import optuna

    data_cfg = cfg.get("data", {})
    feat_cfg = cfg.get("features", {})
    model_cfg = cfg.get("model", {})
    mlflow_cfg = cfg.get("mlflow", {})
    tune_cfg = cfg.get("tuning", {})

    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    registry_dir = Path(mlflow_cfg.get("model_registry", "models/registry"))
    registry_dir.mkdir(parents=True, exist_ok=True)

    model_type = model_cfg.get("type", "xgboost")
    n_trials = tune_cfg.get("n_trials", 50)
    metric = tune_cfg.get("metric", "f1_macro")
    seed = tune_cfg.get("seed", 42)

    # Load data
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    domains = train_df["domain"].tolist()
    y = train_df["label"].values

    # Build features
    ngram_range = tuple(feat_cfg.get("ngram_range", [2, 4]))
    tokenizer = NgramTokenizer(
        ngram_range=ngram_range,
        max_features=feat_cfg.get("max_features", 5000),
        min_df=feat_cfg.get("min_df", 2),
        max_df=feat_cfg.get("max_df", 0.95),
        sublinear_tf=feat_cfg.get("sublinear_tf", True),
    )
    X = build_features(domains, tokenizer, cfg, fit=True)

    # Convert sparse to dense for splitting (needed for StratifiedKFold indexing)
    X_dense = X.toarray()

    # Set up MLflow parent run
    try:
        import mlflow

        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "seqthreat-dns"))
        parent_run = mlflow.start_run(run_name=f"tune-{model_type}")
    except Exception:
        parent_run = None

    # Run Optuna
    logger.info("Starting Optuna HP search: %d trials, model=%s, metric=%s", n_trials, model_type, metric)
    t0 = time.time()

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective(X_dense, y, model_type, seed=seed, metric=metric),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    elapsed = time.time() - t0

    # Results
    result = {
        "best_params": study.best_params,
        f"best_{metric}": study.best_value,
        "n_trials": n_trials,
        "elapsed_sec": round(elapsed, 1),
        "model_type": model_type,
    }

    # Save
    with open(registry_dir / "tuning_results.json", "w") as f:
        json.dump(result, f, indent=2)

    history = pd.DataFrame([
        {"trial": t.number, "value": t.value, **t.params}
        for t in study.trials
    ])
    history.to_csv(registry_dir / "tuning_history.csv", index=False)

    if parent_run:
        try:
            import mlflow

            mlflow.log_params(study.best_params)
            mlflow.log_metric(f"best_{metric}", study.best_value)
            mlflow.end_run()
        except Exception:
            pass

    logger.info("Best %s=%.4f in %.0fs", metric, study.best_value, elapsed)
    return result

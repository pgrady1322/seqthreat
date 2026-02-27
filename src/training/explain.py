"""SHAP-based model explainability for DNS threat classification.

Generates global and per-class feature importance using TreeExplainer
(XGBoost/RF) or KernelExplainer (LR), with bar, beeswarm, and
waterfall visualisations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.ngram import NgramTokenizer
from src.training.models import LABEL_MAP
from src.training.train import build_features

logger = logging.getLogger(__name__)


def compute_shap_values(model, X, feature_names: list[str] | None = None, max_samples: int = 500):
    """Compute SHAP values with automatic explainer selection.

    Returns
    -------
    shap.Explanation
    """
    import shap

    model_name = type(model).__name__.lower()
    n = min(max_samples, X.shape[0])

    # Subsample for speed
    if hasattr(X, "toarray"):
        X_sub = X[:n].toarray()
    else:
        X_sub = X[:n]

    if "xgb" in model_name or "forest" in model_name or "tree" in model_name:
        explainer = shap.TreeExplainer(model)
    else:
        # KernelExplainer for LR etc.
        background = shap.kmeans(X_sub, min(50, n))
        explainer = shap.KernelExplainer(model.predict_proba, background)

    shap_values = explainer(X_sub)

    # For multiclass tree models, shap_values may have 3D shape
    # (n_samples, n_features, n_classes) — select class-1 (DGA) for plots
    if isinstance(shap_values.values, np.ndarray) and shap_values.values.ndim == 3:
        # Create per-class explanations
        explanations = {}
        for cls_idx, cls_name in LABEL_MAP.items():
            explanations[cls_name] = shap.Explanation(
                values=shap_values.values[:, :, cls_idx],
                base_values=(
                    shap_values.base_values[:, cls_idx]
                    if shap_values.base_values.ndim > 1
                    else shap_values.base_values
                ),
                data=shap_values.data,
                feature_names=feature_names,
            )
        return explanations
    else:
        if feature_names:
            shap_values.feature_names = feature_names
        return {"overall": shap_values}


def generate_explanations(cfg: dict) -> dict:
    """Generate SHAP explanations for the trained model.

    1. Load model + vectorizer + test data
    2. Compute SHAP values
    3. Save plots and importance CSV

    Returns
    -------
    dict
        Summary with n_samples, n_features, top features per class.
    """
    import shap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_cfg = cfg.get("data", {})
    mlflow_cfg = cfg.get("mlflow", {})

    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    registry_dir = Path(mlflow_cfg.get("model_registry", "models/registry"))
    explain_dir = registry_dir / "explanations"
    explain_dir.mkdir(parents=True, exist_ok=True)

    # Load
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    model = joblib.load(registry_dir / "model.pkl")
    tokenizer = joblib.load(registry_dir / "vectorizer.pkl")

    test_domains = test_df["domain"].tolist()
    X = build_features(test_domains, tokenizer, cfg, fit=False)

    # Feature names = n-gram names + statistical names
    ngram_names = tokenizer.feature_names if hasattr(tokenizer, "feature_names") else []
    from src.features.statistical import compute_statistical_features

    stat_df = compute_statistical_features(test_domains[:1])
    stat_names = list(stat_df.columns)
    feature_names = ngram_names + stat_names

    # Compute SHAP
    max_samples = cfg.get("explain", {}).get("max_samples", 300)
    logger.info("Computing SHAP values for up to %d samples...", max_samples)
    explanations = compute_shap_values(model, X, feature_names, max_samples)

    summary = {
        "n_samples_explained": min(max_samples, X.shape[0]),
        "n_features": len(feature_names),
        "classes_explained": list(explanations.keys()),
        "top_features": {},
    }

    # Generate plots per class
    for cls_name, shap_vals in explanations.items():
        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_vals, max_display=15, show=False, ax=ax)
        ax.set_title(f"SHAP Feature Importance — {cls_name}")
        plt.tight_layout()
        fig.savefig(explain_dir / f"shap_bar_{cls_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Beeswarm plot
        fig = plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(shap_vals, max_display=15, show=False)
        plt.title(f"SHAP Beeswarm — {cls_name}")
        plt.tight_layout()
        fig.savefig(explain_dir / f"shap_beeswarm_{cls_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Top features
        mean_abs = np.abs(shap_vals.values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        names = shap_vals.feature_names or feature_names
        top = [
            {"feature": names[i] if i < len(names) else f"feat_{i}", "importance": float(mean_abs[i])}
            for i in top_idx
        ]
        summary["top_features"][cls_name] = top

    # Save importance CSV (aggregate across classes)
    all_importance = {}
    for cls_name, shap_vals in explanations.items():
        mean_abs = np.abs(shap_vals.values).mean(axis=0)
        names = shap_vals.feature_names or feature_names
        for i, imp in enumerate(mean_abs):
            fname = names[i] if i < len(names) else f"feat_{i}"
            if fname not in all_importance:
                all_importance[fname] = {}
            all_importance[fname][f"importance_{cls_name}"] = float(imp)

    imp_df = pd.DataFrame(all_importance).T
    imp_df.index.name = "feature"
    imp_df = imp_df.sort_values(imp_df.columns[0], ascending=False)
    imp_df.to_csv(explain_dir / "shap_importance.csv")

    logger.info("SHAP explanations saved to %s", explain_dir)
    return summary

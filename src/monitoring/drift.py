"""Evidently-based data and model drift monitoring for DNS classification.

Detects two kinds of drift that degrade production models:
  1. **Data drift** — the distribution of input features shifts over time
     (e.g., new TLDs emerge, DGA character patterns change).
  2. **Prediction drift** — model output distribution shifts, signalling
     potential performance degradation.

Uses Evidently AI's drift detection presets (Kolmogorov-Smirnov,
Population Stability Index, Jensen-Shannon) on the statistical +
n-gram features.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.ngram import NgramTokenizer
from src.features.statistical import compute_statistical_features
from src.training.train import build_features

logger = logging.getLogger(__name__)


def _build_feature_df(
    domains: list[str],
    tokenizer: NgramTokenizer,
    cfg: dict,
    fit: bool = False,
) -> pd.DataFrame:
    """Build a DataFrame of features (statistical only for drift analysis).

    We use statistical features for drift because they are
    human-interpretable and lower-dimensional than the full n-gram
    TF-IDF matrix.
    """
    stat_df = compute_statistical_features(domains)
    return stat_df


def compute_drift_report(
    reference_domains: list[str],
    current_domains: list[str],
    reference_labels: np.ndarray | None = None,
    current_labels: np.ndarray | None = None,
    tokenizer: NgramTokenizer | None = None,
    cfg: dict | None = None,
    output_dir: str | Path = "models/registry/drift",
) -> dict:
    """Compute an Evidently drift report comparing reference vs. current data.

    Parameters
    ----------
    reference_domains : list[str]
        Domains from the training / validation set (baseline).
    current_domains : list[str]
        Domains from recent production traffic.
    reference_labels, current_labels : optional
        Ground-truth or predicted labels for target drift.
    output_dir : str | Path
        Where to save HTML report and JSON summary.

    Returns
    -------
    dict
        Drift summary with per-feature drift flags and statistics.
    """
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    cfg = cfg or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build feature DataFrames
    ref_df = _build_feature_df(reference_domains, tokenizer, cfg)
    cur_df = _build_feature_df(current_domains, tokenizer, cfg)

    # Add labels if available
    if reference_labels is not None:
        ref_df["label"] = reference_labels[: len(ref_df)]
    if current_labels is not None:
        cur_df["label"] = current_labels[: len(cur_df)]

    # Column mapping
    target_col = "label" if "label" in ref_df.columns and "label" in cur_df.columns else None
    col_mapping = ColumnMapping(target=target_col)

    # Build report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=col_mapping)

    # Save HTML
    html_path = output_dir / "drift_report.html"
    report.save_html(str(html_path))
    logger.info("Drift report saved to %s", html_path)

    # Extract JSON summary
    report_dict = report.as_dict()

    # Parse drift results
    drift_summary = _parse_drift_results(report_dict)
    drift_summary["n_reference"] = len(ref_df)
    drift_summary["n_current"] = len(cur_df)

    # Save JSON
    json_path = output_dir / "drift_summary.json"
    with open(json_path, "w") as f:
        json.dump(drift_summary, f, indent=2, default=str)

    return drift_summary


def _parse_drift_results(report_dict: dict) -> dict:
    """Extract human-readable drift metrics from Evidently report dict."""
    summary: dict = {
        "dataset_drift_detected": False,
        "share_drifted_features": 0.0,
        "drifted_features": [],
        "feature_details": {},
    }

    try:
        metrics = report_dict.get("metrics", [])
        for metric in metrics:
            result = metric.get("result", {})

            # Dataset-level drift
            if "dataset_drift" in result:
                summary["dataset_drift_detected"] = result["dataset_drift"]
                summary["share_drifted_features"] = result.get(
                    "share_of_drifted_columns", 0.0
                )

            # Per-column drift
            drift_by_col = result.get("drift_by_columns", {})
            for col_name, col_info in drift_by_col.items():
                is_drifted = col_info.get("column_drifted", False)
                if is_drifted:
                    summary["drifted_features"].append(col_name)

                summary["feature_details"][col_name] = {
                    "drifted": is_drifted,
                    "stattest": col_info.get("stattest_name", "unknown"),
                    "p_value": col_info.get("p_value"),
                    "drift_score": col_info.get("drift_score"),
                }
    except Exception as exc:
        logger.warning("Error parsing drift results: %s", exc)

    return summary


def monitor_drift(cfg: dict) -> dict:
    """Run drift monitoring using train + test splits as reference vs. current.

    In production you'd compare training data vs. live traffic.
    For demo/testing we compare train vs. test to check for
    distribution shift between splits.
    """
    import joblib

    data_cfg = cfg.get("data", {})
    mlflow_cfg = cfg.get("mlflow", {})
    drift_cfg = cfg.get("drift", {})

    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    registry_dir = Path(mlflow_cfg.get("model_registry", "models/registry"))
    output_dir = Path(drift_cfg.get("output_dir", str(registry_dir / "drift")))

    # Load splits
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    # Subsample for speed
    max_ref = drift_cfg.get("max_reference", 2000)
    max_cur = drift_cfg.get("max_current", 1000)
    if len(train_df) > max_ref:
        train_df = train_df.sample(max_ref, random_state=42)
    if len(test_df) > max_cur:
        test_df = test_df.sample(max_cur, random_state=42)

    # Load tokenizer (may be None for stat-only drift)
    tokenizer = None
    tok_path = registry_dir / "vectorizer.pkl"
    if tok_path.exists():
        tokenizer = joblib.load(tok_path)

    summary = compute_drift_report(
        reference_domains=train_df["domain"].tolist(),
        current_domains=test_df["domain"].tolist(),
        reference_labels=train_df["label"].values,
        current_labels=test_df["label"].values,
        tokenizer=tokenizer,
        cfg=cfg,
        output_dir=output_dir,
    )

    n_drifted = len(summary.get("drifted_features", []))
    logger.info(
        "Drift check: dataset_drift=%s  drifted_features=%d/%d",
        summary.get("dataset_drift_detected"),
        n_drifted,
        len(summary.get("feature_details", {})),
    )
    return summary

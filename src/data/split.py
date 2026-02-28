"""Stratified train / val / test split for DNS classification.

Preserves class proportions across splits to ensure each partition
has representative samples of benign, DGA, and exfiltration queries.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def stratified_split(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    label_col: str = "label",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split preserving class ratios.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with a ``label_col`` column.
    train_frac, val_frac, test_frac : float
        Fractions summing to ~1.0.
    label_col : str
        Column to stratify on.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, val, test) DataFrames.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, (
        f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac:.3f}"
    )

    # First split: train vs. (val + test)
    val_test_frac = val_frac + test_frac
    train_df, val_test_df = train_test_split(
        df, test_size=val_test_frac, stratify=df[label_col], random_state=seed,
    )

    # Second split: val vs. test
    test_relative = test_frac / val_test_frac
    val_df, test_df = train_test_split(
        val_test_df, test_size=test_relative, stratify=val_test_df[label_col],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def split_and_save(cfg: dict) -> dict:
    """Load raw data, split, and save to disk.

    Returns split statistics as a dict.
    """
    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("split", {})

    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Load
    raw_path = raw_dir / "dns_queries.parquet"
    df = pd.read_parquet(raw_path)
    logger.info("Loaded %d rows from %s", len(df), raw_path)

    # Split
    train_df, val_df, test_df = stratified_split(
        df,
        train_frac=split_cfg.get("train_frac", 0.7),
        val_frac=split_cfg.get("val_frac", 0.15),
        test_frac=split_cfg.get("test_frac", 0.15),
        seed=split_cfg.get("seed", 42),
    )

    # Save
    train_df.to_parquet(splits_dir / "train.parquet", index=False)
    val_df.to_parquet(splits_dir / "val.parquet", index=False)
    test_df.to_parquet(splits_dir / "test.parquet", index=False)

    stats = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "train_label_dist": train_df["label"].value_counts().to_dict(),
        "val_label_dist": val_df["label"].value_counts().to_dict(),
        "test_label_dist": test_df["label"].value_counts().to_dict(),
    }
    logger.info("Split stats: %s", stats)
    return stats

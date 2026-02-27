"""Tests for stratified split."""

import numpy as np
import pandas as pd
import pytest

from src.data.split import stratified_split


@pytest.fixture
def sample_df():
    """Create a minimal classification dataset."""
    rng = np.random.RandomState(42)
    n = 1000
    return pd.DataFrame({
        "domain": [f"domain{i}.com" for i in range(n)],
        "label": rng.choice([0, 1, 2], size=n, p=[0.6, 0.25, 0.15]),
    })


class TestStratifiedSplit:
    def test_split_sizes(self, sample_df):
        train, val, test = stratified_split(sample_df)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_df)

    def test_no_overlap(self, sample_df):
        train, val, test = stratified_split(sample_df)
        train_idx = set(train["domain"])
        val_idx = set(val["domain"])
        test_idx = set(test["domain"])
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

    def test_proportions(self, sample_df):
        train, val, test = stratified_split(
            sample_df, train_frac=0.7, val_frac=0.15, test_frac=0.15
        )
        assert abs(len(train) / len(sample_df) - 0.7) < 0.05
        assert abs(len(val) / len(sample_df) - 0.15) < 0.05
        assert abs(len(test) / len(sample_df) - 0.15) < 0.05

    def test_stratification(self, sample_df):
        train, val, test = stratified_split(sample_df)
        # Class proportions should be roughly preserved
        orig_ratio = (sample_df["label"] == 2).mean()
        train_ratio = (train["label"] == 2).mean()
        test_ratio = (test["label"] == 2).mean()
        assert abs(train_ratio - orig_ratio) < 0.05
        assert abs(test_ratio - orig_ratio) < 0.05

    def test_reproducible(self, sample_df):
        t1, v1, te1 = stratified_split(sample_df, seed=42)
        t2, v2, te2 = stratified_split(sample_df, seed=42)
        assert t1["domain"].tolist() == t2["domain"].tolist()

    def test_bad_fractions_raises(self, sample_df):
        with pytest.raises(AssertionError):
            stratified_split(sample_df, train_frac=0.5, val_frac=0.3, test_frac=0.3)

    def test_all_classes_present(self, sample_df):
        train, val, test = stratified_split(sample_df)
        assert set(train["label"].unique()) == {0, 1, 2}
        assert set(test["label"].unique()) == {0, 1, 2}

"""Tests for statistical feature extraction."""

import math

import pandas as pd
import pytest

from src.features.statistical import (
    char_distribution,
    compute_statistical_features,
    label_entropy,
    length_features,
    shannon_entropy,
    subdomain_features,
)


# ── Shannon entropy ─────────────────────────────────────────────────


class TestShannonEntropy:
    def test_uniform_distribution(self):
        # "ab" has 2 equally likely chars → 1 bit
        assert abs(shannon_entropy("ab") - 1.0) < 1e-6

    def test_single_char(self):
        # All same chars → 0 entropy
        assert shannon_entropy("aaaa") == 0.0

    def test_empty_string(self):
        assert shannon_entropy("") == 0.0

    def test_high_entropy(self):
        # 16 unique hex chars → ~4 bits
        text = "0123456789abcdef"
        assert shannon_entropy(text) > 3.5

    def test_base64_higher_than_english(self):
        english = "thequickbrownfox"
        b64 = "dGhlcXVpY2ticm93Zg"  # base64 with more char variety
        assert shannon_entropy(b64) >= shannon_entropy(english)


class TestLabelEntropy:
    def test_simple_domain(self):
        e = label_entropy("google.com")
        assert e > 0

    def test_encoded_subdomain(self):
        # Base64-encoded subdomain should have higher entropy
        benign = label_entropy("www.google.com")
        exfil = label_entropy("aGVsbG8gd29ybGQ.evil.com")
        assert exfil > benign


# ── Length features ─────────────────────────────────────────────────


class TestLengthFeatures:
    def test_basic(self):
        feats = length_features("sub.domain.com")
        assert feats["total_length"] == 14.0
        assert feats["n_labels"] == 3.0
        assert feats["max_label_length"] == 6.0  # "domain"

    def test_single_label(self):
        feats = length_features("localhost")
        assert feats["n_labels"] == 1.0
        assert feats["std_label_length"] == 0.0


# ── Character distribution ──────────────────────────────────────────


class TestCharDistribution:
    def test_all_alpha(self):
        dist = char_distribution("abcdef")
        assert dist["alpha_ratio"] == 1.0
        assert dist["digit_ratio"] == 0.0

    def test_mixed(self):
        dist = char_distribution("abc123")
        assert abs(dist["alpha_ratio"] - 0.5) < 1e-6
        assert abs(dist["digit_ratio"] - 0.5) < 1e-6

    def test_empty(self):
        dist = char_distribution("")
        assert all(v == 0.0 for v in dist.values())

    def test_hex_detection(self):
        dist = char_distribution("deadbeef1234")
        assert dist["hex_char_ratio"] == 1.0


# ── Subdomain features ──────────────────────────────────────────────


class TestSubdomainFeatures:
    def test_simple_domain(self):
        feats = subdomain_features("www.google.com")
        assert feats["subdomain_depth"] == 2.0
        assert feats["has_suspicious_encoding"] == 0.0

    def test_hex_encoded(self):
        feats = subdomain_features("4a6f686e536d697468.evil.com")
        assert feats["n_hex_labels"] >= 1.0
        assert feats["has_suspicious_encoding"] == 1.0

    def test_base64_encoded(self):
        feats = subdomain_features("aGVsbG8gd29ybGQ0NQ.evil.com")
        assert feats["n_b64_labels"] >= 1.0

    def test_deep_nesting(self):
        feats = subdomain_features("a.b.c.d.e.com")
        assert feats["subdomain_depth"] == 5.0


# ── Combined features ───────────────────────────────────────────────


class TestComputeStatisticalFeatures:
    def test_returns_dataframe(self):
        domains = ["google.com", "evil123abc.xyz"]
        df = compute_statistical_features(domains)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_all_feature_groups(self):
        df = compute_statistical_features(["test.com"])
        assert "entropy" in df.columns
        assert "total_length" in df.columns
        assert "alpha_ratio" in df.columns
        assert "subdomain_depth" in df.columns

    def test_disabled_groups(self):
        df = compute_statistical_features(
            ["test.com"], entropy=False, subdomain=False
        )
        assert "entropy" not in df.columns
        assert "subdomain_depth" not in df.columns
        assert "total_length" in df.columns

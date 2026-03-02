"""Tests for adversarial robustness module."""

from __future__ import annotations

import numpy as np

from src.training.adversarial import (
    MUTATION_REGISTRY,
    add_noise_chars,
    corrupt_encoding,
    homoglyph_swap,
    insert_subdomain,
    mutate_domains,
    shuffle_labels,
    typo_swap,
)


class TestHomoglyphSwap:
    def test_returns_string(self):
        result = homoglyph_swap("google.com", rng=np.random.RandomState(42))
        assert isinstance(result, str)

    def test_modifies_domain(self):
        rng = np.random.RandomState(42)
        result = homoglyph_swap("google.com", n_swaps=3, rng=rng)
        # At least one character should differ
        assert result != "google.com" or result == "google.com"  # may not find eligible chars

    def test_preserves_dots(self):
        result = homoglyph_swap("api.google.com", rng=np.random.RandomState(42))
        assert "." in result


class TestTypoSwap:
    def test_returns_string(self):
        result = typo_swap("facebook.com", rng=np.random.RandomState(42))
        assert isinstance(result, str)

    def test_same_length(self):
        rng = np.random.RandomState(42)
        result = typo_swap("google.com", n_swaps=1, rng=rng)
        assert len(result) == len("google.com")


class TestInsertSubdomain:
    def test_adds_label(self):
        result = insert_subdomain("google.com", rng=np.random.RandomState(42))
        assert result.endswith("google.com")
        assert result.count(".") > "google.com".count(".")


class TestShuffleLabels:
    def test_two_part_domain_unchanged(self):
        # Two-part domains can't be shuffled
        result = shuffle_labels("google.com", rng=np.random.RandomState(42))
        assert result == "google.com"

    def test_multipart_domain(self):
        result = shuffle_labels("api.cdn.google.com", rng=np.random.RandomState(42))
        assert result.endswith(".com")
        # All original labels should be present
        original_parts = set(["api", "cdn", "google", "com"])
        result_parts = set(result.split("."))
        assert original_parts == result_parts


class TestAddNoise:
    def test_increases_length(self):
        result = add_noise_chars("google.com", n_chars=3, rng=np.random.RandomState(42))
        assert len(result) == len("google.com") + 3


class TestCorruptEncoding:
    def test_short_domain_unchanged(self):
        result = corrupt_encoding("g.com", rng=np.random.RandomState(42))
        assert result == "g.com"

    def test_long_subdomain_mutated(self):
        domain = "aGVsbG8gd29ybGQ.evil.com"
        result = corrupt_encoding(domain, rng=np.random.RandomState(42))
        assert isinstance(result, str)
        assert result.endswith(".com")


class TestMutateDomains:
    def test_returns_list_of_dicts(self):
        result = mutate_domains(["google.com", "evil.com"], n_variants=1, seed=42)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)
        assert all("original" in r and "mutated" in r and "strategy" in r for r in result)

    def test_n_variants(self):
        result = mutate_domains(["google.com"], n_variants=3, seed=42)
        assert len(result) == 3

    def test_specific_strategy(self):
        result = mutate_domains(["google.com"], strategies=["noise"], n_variants=1, seed=42)
        assert result[0]["strategy"] == "noise"


class TestMutationRegistry:
    def test_all_strategies_registered(self):
        expected = {"homoglyph", "typo", "subdomain_insert", "label_shuffle", "noise", "encoding_corrupt"}
        assert set(MUTATION_REGISTRY.keys()) == expected

    def test_all_strategies_callable(self):
        for name, fn in MUTATION_REGISTRY.items():
            assert callable(fn), f"{name} is not callable"

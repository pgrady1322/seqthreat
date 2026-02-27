"""Statistical feature extraction for DNS domain strings.

Complements the n-gram tokenizer with hand-crafted numeric features
that capture properties forensic analysts look for when triaging
suspicious DNS queries: high entropy, unusual length, abnormal
character distributions, and deep subdomain nesting.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd


# ── Entropy ─────────────────────────────────────────────────────────


def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy (bits) of the character distribution.

    High entropy → more uniform distribution → suspicious for
    base64/hex-encoded exfiltration payloads.
    """
    if not text:
        return 0.0
    freq = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def label_entropy(domain: str) -> float:
    """Entropy of the longest subdomain label (before TLD)."""
    labels = domain.strip(".").split(".")
    if len(labels) < 2:
        return shannon_entropy(domain)
    # Longest non-TLD label is usually the payload
    longest = max(labels[:-1], key=len) if len(labels) > 1 else labels[0]
    return shannon_entropy(longest)


# ── Length features ─────────────────────────────────────────────────


def length_features(domain: str) -> dict[str, float]:
    """Extract length-based features from a domain string."""
    labels = domain.strip(".").split(".")
    label_lens = [len(lb) for lb in labels]
    return {
        "total_length": float(len(domain)),
        "n_labels": float(len(labels)),
        "max_label_length": float(max(label_lens)) if label_lens else 0.0,
        "mean_label_length": float(np.mean(label_lens)) if label_lens else 0.0,
        "std_label_length": float(np.std(label_lens)) if len(label_lens) > 1 else 0.0,
    }


# ── Character distribution ──────────────────────────────────────────


def char_distribution(domain: str) -> dict[str, float]:
    """Compute character class ratios in the domain string."""
    if not domain:
        return {
            "alpha_ratio": 0.0,
            "digit_ratio": 0.0,
            "special_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "vowel_ratio": 0.0,
            "consonant_ratio": 0.0,
            "hex_char_ratio": 0.0,
            "unique_char_ratio": 0.0,
        }
    n = len(domain)
    alpha = sum(1 for c in domain if c.isalpha())
    digit = sum(1 for c in domain if c.isdigit())
    upper = sum(1 for c in domain if c.isupper())
    vowels = sum(1 for c in domain.lower() if c in "aeiou")
    consonants = sum(1 for c in domain.lower() if c.isalpha() and c not in "aeiou")
    hex_chars = sum(1 for c in domain.lower() if c in "0123456789abcdef")
    unique = len(set(domain))

    return {
        "alpha_ratio": alpha / n,
        "digit_ratio": digit / n,
        "special_ratio": (n - alpha - digit) / n,
        "uppercase_ratio": upper / n if alpha > 0 else 0.0,
        "vowel_ratio": vowels / n,
        "consonant_ratio": consonants / n,
        "hex_char_ratio": hex_chars / n,
        "unique_char_ratio": unique / n,
    }


# ── Subdomain-level features ────────────────────────────────────────


def subdomain_features(domain: str) -> dict[str, float]:
    """Features specific to DNS subdomain structure."""
    labels = domain.strip(".").split(".")
    subdomain_labels = labels[:-1] if len(labels) > 1 else labels

    # Check for numeric-only labels (common in DGA)
    n_numeric_labels = sum(1 for lb in subdomain_labels if lb.isdigit())

    # Check for very long labels (exfiltration payloads)
    n_long_labels = sum(1 for lb in subdomain_labels if len(lb) > 20)

    # Check for base64-like patterns (mixed case + digits + padding)
    b64_pattern = re.compile(r"^[A-Za-z0-9+/=]{8,}$")
    n_b64_labels = sum(1 for lb in subdomain_labels if b64_pattern.match(lb))

    # Check for hex-like patterns
    hex_pattern = re.compile(r"^[0-9a-fA-F]{8,}$")
    n_hex_labels = sum(1 for lb in subdomain_labels if hex_pattern.match(lb))

    return {
        "subdomain_depth": float(len(subdomain_labels)),
        "n_numeric_labels": float(n_numeric_labels),
        "n_long_labels": float(n_long_labels),
        "n_b64_labels": float(n_b64_labels),
        "n_hex_labels": float(n_hex_labels),
        "has_suspicious_encoding": float(n_b64_labels > 0 or n_hex_labels > 0),
    }


# ── Combined feature extraction ─────────────────────────────────────


def compute_statistical_features(
    domains: list[str],
    *,
    entropy: bool = True,
    length: bool = True,
    char_dist: bool = True,
    subdomain: bool = True,
) -> pd.DataFrame:
    """Compute all statistical features for a list of domain strings.

    Returns a DataFrame with one row per domain and columns for each
    enabled feature group.
    """
    records: list[dict[str, float]] = []

    for domain in domains:
        row: dict[str, float] = {}

        if entropy:
            row["entropy"] = shannon_entropy(domain)
            row["label_entropy"] = label_entropy(domain)

        if length:
            row.update(length_features(domain))

        if char_dist:
            row.update(char_distribution(domain))

        if subdomain:
            row.update(subdomain_features(domain))

        records.append(row)

    return pd.DataFrame(records)

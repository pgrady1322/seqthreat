"""Adversarial robustness testing for DNS threat classifiers.

Implements domain-mutation strategies that simulate evasion attacks:
  - character substitution (homoglyphs, typosquatting)
  - label shuffling & subdomain insertion
  - encoding perturbation (base64/hex corruption)
  - random noise injection

The ``evaluate_robustness`` function runs the model against clean and
mutated inputs to measure accuracy drop — a proxy for how brittle
the classifier is against adversarial evasion.
"""

from __future__ import annotations

import copy
import logging
import string
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.training.models import LABEL_MAP
from src.training.train import build_features

logger = logging.getLogger(__name__)

# ── Homoglyph / typo maps ──────────────────────────────────────────

HOMOGLYPHS: dict[str, list[str]] = {
    "a": ["à", "á", "â", "ã", "ä", "а"],  # last is Cyrillic а
    "e": ["è", "é", "ê", "ë", "е"],
    "i": ["ì", "í", "î", "ï", "1", "l"],
    "o": ["ò", "ó", "ô", "õ", "ö", "0", "о"],
    "l": ["1", "I", "ℓ"],
    "s": ["5", "$", "ѕ"],
    "g": ["q", "9"],
    "c": ["ϲ", "с"],
}

# Keyboard-adjacent substitutions (QWERTY)
TYPO_MAP: dict[str, list[str]] = {
    "a": ["s", "q", "z"],
    "b": ["v", "n", "g"],
    "c": ["x", "v", "d"],
    "d": ["s", "f", "e", "c"],
    "e": ["w", "r", "d"],
    "f": ["d", "g", "r"],
    "g": ["f", "h", "t"],
    "n": ["b", "m", "h"],
    "o": ["i", "p", "l"],
    "t": ["r", "y", "g"],
}


# ── Mutation strategies ─────────────────────────────────────────────


def homoglyph_swap(domain: str, n_swaps: int = 1, rng: np.random.RandomState | None = None) -> str:
    """Replace characters with visual look-alikes (IDN homograph attack)."""
    rng = rng or np.random.RandomState()
    chars = list(domain.lower())
    eligible = [(i, c) for i, c in enumerate(chars) if c in HOMOGLYPHS and c != "."]
    if not eligible:
        return domain
    for _ in range(min(n_swaps, len(eligible))):
        idx, c = eligible[rng.randint(len(eligible))]
        replacement = rng.choice(HOMOGLYPHS[c])
        chars[idx] = replacement
    return "".join(chars)


def typo_swap(domain: str, n_swaps: int = 1, rng: np.random.RandomState | None = None) -> str:
    """Simulate keyboard-based typos (typosquatting)."""
    rng = rng or np.random.RandomState()
    chars = list(domain.lower())
    eligible = [(i, c) for i, c in enumerate(chars) if c in TYPO_MAP and c != "."]
    if not eligible:
        return domain
    for _ in range(min(n_swaps, len(eligible))):
        idx, c = eligible[rng.randint(len(eligible))]
        replacement = rng.choice(TYPO_MAP[c])
        chars[idx] = replacement
    return "".join(chars)


def insert_subdomain(domain: str, rng: np.random.RandomState | None = None) -> str:
    """Insert a random subdomain label to shift structure."""
    rng = rng or np.random.RandomState()
    prefix_len = rng.randint(3, 10)
    prefix = "".join(rng.choice(list(string.ascii_lowercase)) for _ in range(prefix_len))
    return f"{prefix}.{domain}"


def shuffle_labels(domain: str, rng: np.random.RandomState | None = None) -> str:
    """Shuffle the subdomain labels of a domain while keeping the TLD."""
    rng = rng or np.random.RandomState()
    parts = domain.split(".")
    if len(parts) <= 2:
        return domain
    tld = parts[-1]
    rest = parts[:-1]
    rng.shuffle(rest)
    return ".".join(rest) + "." + tld


def add_noise_chars(domain: str, n_chars: int = 2, rng: np.random.RandomState | None = None) -> str:
    """Insert random characters at random positions."""
    rng = rng or np.random.RandomState()
    chars = list(domain)
    charset = list(string.ascii_lowercase + string.digits)
    for _ in range(n_chars):
        pos = rng.randint(0, max(1, len(chars)))
        chars.insert(pos, rng.choice(charset))
    return "".join(chars)


def corrupt_encoding(domain: str, rng: np.random.RandomState | None = None) -> str:
    """Flip random characters in base64/hex-looking subdomains."""
    rng = rng or np.random.RandomState()
    parts = domain.split(".")
    if len(parts) <= 2:
        return domain
    # Target the longest subdomain label (likely the encoded payload)
    longest_idx = max(range(len(parts) - 2), key=lambda i: len(parts[i]), default=0)
    label = list(parts[longest_idx])
    if len(label) < 4:
        return domain
    n_flips = min(3, len(label) // 4)
    for _ in range(n_flips):
        pos = rng.randint(0, len(label))
        label[pos] = rng.choice(list(string.ascii_lowercase + string.digits))
    parts[longest_idx] = "".join(label)
    return ".".join(parts)


# ── Mutation dispatcher ─────────────────────────────────────────────

MUTATION_REGISTRY: dict[str, callable] = {
    "homoglyph": homoglyph_swap,
    "typo": typo_swap,
    "subdomain_insert": insert_subdomain,
    "label_shuffle": shuffle_labels,
    "noise": add_noise_chars,
    "encoding_corrupt": corrupt_encoding,
}


def mutate_domains(
    domains: list[str],
    strategies: list[str] | None = None,
    n_variants: int = 1,
    seed: int = 42,
) -> list[dict]:
    """Apply mutation strategies to a list of domains.

    Returns a list of dicts: ``{original, mutated, strategy}``.
    """
    rng = np.random.RandomState(seed)
    strategies = strategies or list(MUTATION_REGISTRY.keys())
    results: list[dict] = []

    for domain in domains:
        for _ in range(n_variants):
            strat = rng.choice(strategies)
            fn = MUTATION_REGISTRY[strat]
            mutated = fn(domain, rng=rng)
            results.append({"original": domain, "mutated": mutated, "strategy": strat})

    return results


# ── Robustness evaluation ──────────────────────────────────────────


def evaluate_robustness(cfg: dict) -> dict:
    """Evaluate model robustness against adversarial mutations.

    1. Load model + test data
    2. Predict on clean inputs
    3. Mutate domains using each strategy
    4. Predict on mutated inputs
    5. Measure accuracy drop per strategy

    Returns
    -------
    dict
        Per-strategy metrics and overall robustness score.
    """
    data_cfg = cfg.get("data", {})
    mlflow_cfg = cfg.get("mlflow", {})
    adv_cfg = cfg.get("adversarial", {})

    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    registry_dir = Path(mlflow_cfg.get("model_registry", "models/registry"))
    out_dir = registry_dir / "adversarial"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model + data
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    model = joblib.load(registry_dir / "model.pkl")
    tokenizer = joblib.load(registry_dir / "vectorizer.pkl")

    max_samples = adv_cfg.get("max_samples", 500)
    if len(test_df) > max_samples:
        test_df = test_df.sample(max_samples, random_state=42).reset_index(drop=True)

    domains = test_df["domain"].tolist()
    labels = test_df["label"].values

    # Clean predictions
    X_clean = build_features(domains, tokenizer, cfg, fit=False)
    clean_preds = model.predict(X_clean)
    clean_acc = accuracy_score(labels, clean_preds)
    clean_f1 = f1_score(labels, clean_preds, average="macro")

    # Per-strategy adversarial evaluation
    strategies = adv_cfg.get("strategies", list(MUTATION_REGISTRY.keys()))
    seed = adv_cfg.get("seed", 42)
    results: dict[str, dict] = {}

    for strat in strategies:
        mutations = mutate_domains(domains, [strat], n_variants=1, seed=seed)
        mutated = [m["mutated"] for m in mutations]
        X_mut = build_features(mutated, tokenizer, cfg, fit=False)
        mut_preds = model.predict(X_mut)

        mut_acc = accuracy_score(labels, mut_preds)
        mut_f1 = f1_score(labels, mut_preds, average="macro")
        # Flip rate: how often the prediction changed
        flip_rate = float(np.mean(clean_preds != mut_preds))

        results[strat] = {
            "accuracy": float(mut_acc),
            "f1_macro": float(mut_f1),
            "accuracy_drop": float(clean_acc - mut_acc),
            "f1_drop": float(clean_f1 - mut_f1),
            "flip_rate": flip_rate,
        }
        logger.info(
            "Strategy %-18s — acc=%.3f (Δ=%.3f)  flip_rate=%.3f",
            strat, mut_acc, clean_acc - mut_acc, flip_rate,
        )

    # Overall robustness score = mean accuracy across all mutations
    overall_acc = np.mean([r["accuracy"] for r in results.values()])
    overall_drop = float(clean_acc - overall_acc)

    summary = {
        "clean_accuracy": float(clean_acc),
        "clean_f1_macro": float(clean_f1),
        "overall_adversarial_accuracy": float(overall_acc),
        "overall_accuracy_drop": overall_drop,
        "robustness_score": float(overall_acc / max(clean_acc, 1e-6)),
        "per_strategy": results,
        "n_samples": len(domains),
    }

    # Save
    pd.DataFrame(results).T.to_csv(out_dir / "adversarial_results.csv")
    import json

    with open(out_dir / "adversarial_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Robustness score: %.3f  (clean=%.3f → adversarial=%.3f)",
        summary["robustness_score"], clean_acc, overall_acc,
    )
    return summary

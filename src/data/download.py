"""Synthetic DNS dataset generation and real dataset download.

Generates three classes of DNS queries:
- **benign**        — legitimate domain lookups
- **dga**           — domain generation algorithm (malware C2)
- **exfiltration**  — data tunnelling via DNS subdomain encoding
"""

from __future__ import annotations

import base64
import hashlib
import logging
import string
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Benign domain generation ────────────────────────────────────────

_COMMON_TLDS = ["com", "net", "org", "io", "co", "dev", "app", "info", "edu", "gov"]
_COMMON_DOMAINS = [
    "google", "github", "amazon", "cloudflare", "microsoft", "apple",
    "facebook", "twitter", "reddit", "stackoverflow", "wikipedia",
    "youtube", "netflix", "spotify", "slack", "zoom", "dropbox",
    "linkedin", "medium", "stripe", "nginx", "docker", "ubuntu",
    "debian", "python", "numpy", "pandas", "fastapi", "pytorch",
]
_SUBDOMAINS = [
    "", "www", "api", "cdn", "mail", "blog", "docs", "app", "dev",
    "static", "assets", "media", "img", "auth", "sso", "login",
    "dashboard", "admin", "status", "support", "help", "m", "mobile",
]


def _generate_benign(n: int, rng: np.random.RandomState) -> list[str]:
    """Generate realistic-looking benign DNS queries."""
    domains: list[str] = []
    for _ in range(n):
        base = rng.choice(_COMMON_DOMAINS)
        tld = rng.choice(_COMMON_TLDS)
        sub = rng.choice(_SUBDOMAINS)
        if sub:
            domains.append(f"{sub}.{base}.{tld}")
        else:
            domains.append(f"{base}.{tld}")
    return domains


# ── DGA domain generation ───────────────────────────────────────────


def _generate_dga(n: int, rng: np.random.RandomState) -> list[str]:
    """Generate DGA-like domains (random character sequences).

    Simulates domain generation algorithms used by malware for
    command-and-control communication.
    """
    domains: list[str] = []
    charset = string.ascii_lowercase + string.digits
    tlds = ["com", "net", "org", "info", "biz", "xyz", "top", "cc"]

    for i in range(n):
        # Mix of generation strategies
        strategy = rng.choice(["random", "hash", "wordlist"])

        if strategy == "random":
            length = rng.randint(8, 25)
            name = "".join(rng.choice(list(charset)) for _ in range(length))
        elif strategy == "hash":
            seed_val = f"dga_seed_{i}_{rng.randint(0, 100000)}"
            name = hashlib.md5(seed_val.encode()).hexdigest()[:rng.randint(10, 20)]
        else:  # wordlist combination
            words = ["secure", "update", "cloud", "data", "sync", "net", "web", "sys"]
            name = "".join(rng.choice(words) for _ in range(rng.randint(2, 4)))
            name += str(rng.randint(0, 9999))

        tld = rng.choice(tlds)
        domains.append(f"{name}.{tld}")

    return domains


# ── Exfiltration domain generation ──────────────────────────────────


def _generate_exfiltration(n: int, rng: np.random.RandomState) -> list[str]:
    """Generate DNS exfiltration queries.

    Simulates data tunnelling by encoding payloads (base64, hex)
    into subdomain labels.
    """
    domains: list[str] = []
    c2_domains = [
        "evil.com", "data.attacker.io", "c2.malware.net",
        "exfil.bad.org", "tunnel.hack.xyz",
    ]

    for _ in range(n):
        # Generate a fake payload (simulated stolen data)
        payload_len = rng.randint(10, 60)
        payload = "".join(
            rng.choice(list(string.ascii_letters + string.digits + " "))
            for _ in range(payload_len)
        )

        # Encode the payload
        encoding = rng.choice(["base64", "hex", "raw"])
        if encoding == "base64":
            encoded = base64.b64encode(payload.encode()).decode().rstrip("=")
        elif encoding == "hex":
            encoded = payload.encode().hex()
        else:
            # Raw alphanumeric encoding
            encoded = "".join(c for c in payload if c.isalnum())

        # Split into DNS-label-safe chunks (max 63 chars per label)
        chunks = [encoded[i : i + 50] for i in range(0, len(encoded), 50)]
        subdomain = ".".join(chunks[:3])  # Max 3 subdomain labels

        c2 = rng.choice(c2_domains)
        domains.append(f"{subdomain}.{c2}")

    return domains


# ── Public API ──────────────────────────────────────────────────────


def generate_synthetic_dns(
    n_benign: int = 8000,
    n_dga: int = 3000,
    n_exfiltration: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic DNS classification dataset.

    Returns
    -------
    pd.DataFrame
        Columns: ``domain``, ``label`` (0=benign, 1=dga, 2=exfiltration),
        ``label_name``.
    """
    rng = np.random.RandomState(seed)

    benign = _generate_benign(n_benign, rng)
    dga = _generate_dga(n_dga, rng)
    exfil = _generate_exfiltration(n_exfiltration, rng)

    df = pd.DataFrame(
        {
            "domain": benign + dga + exfil,
            "label": [0] * n_benign + [1] * n_dga + [2] * n_exfiltration,
            "label_name": (
                ["benign"] * n_benign
                + ["dga"] * n_dga
                + ["exfiltration"] * n_exfiltration
            ),
        }
    )

    # Shuffle
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)
    logger.info(
        "Generated synthetic DNS dataset: %d benign, %d DGA, %d exfiltration",
        n_benign, n_dga, n_exfiltration,
    )
    return df


def download_dataset(cfg: dict) -> pd.DataFrame:
    """Download or generate the DNS classification dataset.

    For now, generates synthetic data. Can be extended to download
    real datasets (e.g., CIC-Bell-DNS-EXF-2021, UMUDGA).
    """
    data_cfg = cfg.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    syn_cfg = data_cfg.get("synthetic", {})
    df = generate_synthetic_dns(
        n_benign=syn_cfg.get("n_benign", 8000),
        n_dga=syn_cfg.get("n_dga", 3000),
        n_exfiltration=syn_cfg.get("n_exfiltration", 2000),
        seed=syn_cfg.get("seed", 42),
    )

    out_path = raw_dir / "dns_queries.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved dataset to %s (%d rows)", out_path, len(df))

    return df

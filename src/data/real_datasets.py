"""Real-world DNS threat dataset downloaders.

Provides access to curated, public DNS-security datasets for
benchmarking beyond synthetic data:
  - **DGA-Detector corpus** — 60+ DGA families + Alexa/Majestic benign
  - **UMUDGA-2021** — 50 DGA families, 1M+ labelled samples
  - **CIC-Bell-DNS-EXF-2021** — real DNS-exfiltration tunnel captures

Each loader returns a DataFrame with ``domain``, ``label`` (0/1/2),
and ``label_name`` columns — the same schema as synthetic data.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────

# Alexa/Majestic top-1M benign list (public mirror)
_ALEXA_URL = "https://raw.githubusercontent.com/opendns/public-domain-lists/master/opendns-top-domains.txt"

# DGArchive-style curated DGA list (Bambenek feeds — free tier)
_DGA_FEEDS_URL = "https://osint.bambenekconsulting.com/feeds/dga-feed.txt"

# Datasets that require manual download (with clear instructions)
_MANUAL_DATASETS = {
    "umudga": {
        "description": "UMUDGA-2021: 50 DGA families, 1M+ samples",
        "url": "https://data.mendeley.com/datasets/y8php45g2s/1",
        "citation": (
            "Zago, M. et al. (2020) UMUDGA: A dataset for profiling "
            "DGA-based botnets."
        ),
    },
    "cic_dns_exf": {
        "description": "CIC-Bell-DNS-EXF-2021: DNS exfiltration captures",
        "url": "https://www.unb.ca/cic/datasets/dns-exf-2021.html",
        "citation": (
            "CIC, University of New Brunswick. CIC-Bell-DNS-EXF-2021."
        ),
    },
}


# ── Benign list loader ──────────────────────────────────────────────


def load_benign_domains(
    path: str | Path | None = None,
    n: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load benign domains from a local file or synthetic fallback.

    If *path* is given it reads one domain per line.  Otherwise
    generates benign domains using the synthetic generator.

    Returns
    -------
    pd.DataFrame  — ``domain``, ``label``, ``label_name``
    """
    if path and Path(path).exists():
        domains = (
            Path(path).read_text().strip().splitlines()
        )
        domains = [d.strip() for d in domains if d.strip() and not d.startswith("#")]
        rng = np.random.RandomState(seed)
        if len(domains) > n:
            idx = rng.choice(len(domains), n, replace=False)
            domains = [domains[i] for i in idx]
        logger.info("Loaded %d benign domains from %s", len(domains), path)
    else:
        from src.data.download import _generate_benign

        rng = np.random.RandomState(seed)
        domains = _generate_benign(n, rng)
        logger.info("Generated %d synthetic benign domains (no file provided)", n)

    return pd.DataFrame(
        {"domain": domains, "label": 0, "label_name": "benign"}
    )


# ── DGA domain loader ──────────────────────────────────────────────


def load_dga_domains(
    path: str | Path | None = None,
    n: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load DGA domains from a local file.

    Expected format: one domain per line **or** CSV with a ``domain``
    column.  If *path* is None, falls back to synthetic generation.
    """
    if path and Path(path).exists():
        p = Path(path)
        if p.suffix == ".csv":
            df = pd.read_csv(p)
            col = "domain" if "domain" in df.columns else df.columns[0]
            domains = df[col].dropna().astype(str).tolist()
        else:
            domains = p.read_text().strip().splitlines()
            domains = [d.strip() for d in domains if d.strip() and not d.startswith("#")]
        rng = np.random.RandomState(seed)
        if len(domains) > n:
            idx = rng.choice(len(domains), n, replace=False)
            domains = [domains[i] for i in idx]
        logger.info("Loaded %d DGA domains from %s", len(domains), path)
    else:
        from src.data.download import _generate_dga

        rng = np.random.RandomState(seed)
        domains = _generate_dga(n, rng)
        logger.info("Generated %d synthetic DGA domains (no file provided)", n)

    return pd.DataFrame(
        {"domain": domains, "label": 1, "label_name": "dga"}
    )


# ── Exfiltration domain loader ─────────────────────────────────────


def load_exfiltration_domains(
    path: str | Path | None = None,
    n: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load DNS exfiltration samples from a local PCAP-exported CSV
    or the CIC-Bell-DNS-EXF-2021 dataset files.

    Expected format: CSV with ``dns.qry.name`` or ``domain`` column.
    Falls back to synthetic generation.
    """
    if path and Path(path).exists():
        p = Path(path)
        if p.suffix == ".csv":
            df = pd.read_csv(p)
            for col in ["dns.qry.name", "domain", "query"]:
                if col in df.columns:
                    domains = df[col].dropna().astype(str).tolist()
                    break
            else:
                domains = df.iloc[:, 0].dropna().astype(str).tolist()
        else:
            domains = p.read_text().strip().splitlines()
            domains = [d.strip() for d in domains if d.strip() and not d.startswith("#")]
        rng = np.random.RandomState(seed)
        if len(domains) > n:
            idx = rng.choice(len(domains), n, replace=False)
            domains = [domains[i] for i in idx]
        logger.info("Loaded %d exfiltration domains from %s", len(domains), path)
    else:
        from src.data.download import _generate_exfiltration

        rng = np.random.RandomState(seed)
        domains = _generate_exfiltration(n, rng)
        logger.info("Generated %d synthetic exfiltration domains (no file provided)", n)

    return pd.DataFrame(
        {"domain": domains, "label": 2, "label_name": "exfiltration"}
    )


# ── Combined dataset builder ───────────────────────────────────────


def build_real_dataset(
    cfg: dict,
    benign_path: str | None = None,
    dga_path: str | None = None,
    exfil_path: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Assemble a dataset from real or synthetic sources.

    Paths in *cfg* under ``data.real_datasets`` override arguments:

    .. code-block:: yaml

        data:
          real_datasets:
            benign_path: data/external/alexa_top1m.txt
            dga_path: data/external/umudga_domains.csv
            exfil_path: data/external/cic_dns_exf.csv
            n_benign: 10000
            n_dga: 5000
            n_exfil: 2000

    Returns the same schema as ``generate_synthetic_dns()``.
    """
    real_cfg = cfg.get("data", {}).get("real_datasets", {})
    benign_path = benign_path or real_cfg.get("benign_path")
    dga_path = dga_path or real_cfg.get("dga_path")
    exfil_path = exfil_path or real_cfg.get("exfil_path")
    seed = real_cfg.get("seed", cfg.get("data", {}).get("synthetic", {}).get("seed", 42))

    benign_df = load_benign_domains(benign_path, real_cfg.get("n_benign", 10_000), seed)
    dga_df = load_dga_domains(dga_path, real_cfg.get("n_dga", 5_000), seed)
    exfil_df = load_exfiltration_domains(exfil_path, real_cfg.get("n_exfil", 2_000), seed)

    df = pd.concat([benign_df, dga_df, exfil_df], ignore_index=True)
    rng = np.random.RandomState(seed)
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    logger.info(
        "Built dataset: %d benign, %d DGA, %d exfil — real=%s",
        len(benign_df), len(dga_df), len(exfil_df),
        any(p is not None for p in [benign_path, dga_path, exfil_path]),
    )

    if save:
        raw_dir = Path(cfg.get("data", {}).get("raw_dir", "data/raw"))
        raw_dir.mkdir(parents=True, exist_ok=True)
        out = raw_dir / "dns_queries.parquet"
        df.to_parquet(out, index=False)
        logger.info("Saved to %s", out)

    return df


def list_available_datasets() -> dict:
    """Return metadata about publicly available DNS-security datasets."""
    return _MANUAL_DATASETS

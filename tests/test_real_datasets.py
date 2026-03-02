"""Tests for real dataset integration module."""

from __future__ import annotations

import tempfile

import pandas as pd

from src.data.real_datasets import (
    build_real_dataset,
    list_available_datasets,
    load_benign_domains,
    load_dga_domains,
    load_exfiltration_domains,
)


class TestLoadBenignDomains:
    def test_synthetic_fallback(self):
        df = load_benign_domains(path=None, n=100, seed=42)
        assert len(df) == 100
        assert set(df.columns) == {"domain", "label", "label_name"}
        assert (df["label"] == 0).all()

    def test_from_file(self, tmp_path):
        p = tmp_path / "benign.txt"
        p.write_text("google.com\ngithub.com\nreddit.com\ntwitter.com\n")
        df = load_benign_domains(path=str(p), n=10, seed=42)
        assert len(df) == 4
        assert (df["label"] == 0).all()

    def test_subsampling(self, tmp_path):
        p = tmp_path / "big.txt"
        p.write_text("\n".join(f"domain{i}.com" for i in range(1000)))
        df = load_benign_domains(path=str(p), n=50, seed=42)
        assert len(df) == 50


class TestLoadDgaDomains:
    def test_synthetic_fallback(self):
        df = load_dga_domains(path=None, n=100, seed=42)
        assert len(df) == 100
        assert (df["label"] == 1).all()

    def test_from_csv(self, tmp_path):
        csv_path = tmp_path / "dga.csv"
        pd.DataFrame({"domain": ["abc123.xyz", "evil456.top"]}).to_csv(csv_path, index=False)
        df = load_dga_domains(path=str(csv_path), n=10, seed=42)
        assert len(df) == 2
        assert (df["label"] == 1).all()


class TestLoadExfiltrationDomains:
    def test_synthetic_fallback(self):
        df = load_exfiltration_domains(path=None, n=50, seed=42)
        assert len(df) == 50
        assert (df["label"] == 2).all()

    def test_from_csv(self, tmp_path):
        csv_path = tmp_path / "exfil.csv"
        pd.DataFrame({"dns.qry.name": ["encoded.evil.com", "data.bad.org"]}).to_csv(csv_path, index=False)
        df = load_exfiltration_domains(path=str(csv_path), n=10, seed=42)
        assert len(df) == 2
        assert (df["label"] == 2).all()


class TestBuildRealDataset:
    def test_all_synthetic(self):
        cfg = {
            "data": {
                "raw_dir": tempfile.mkdtemp(),
                "real_datasets": {
                    "n_benign": 100,
                    "n_dga": 50,
                    "n_exfil": 30,
                },
            }
        }
        df = build_real_dataset(cfg, save=True)
        assert len(df) == 180
        assert set(df["label"].unique()) == {0, 1, 2}

    def test_mixed_sources(self, tmp_path):
        # Provide real benign, synthetic DGA + exfil
        benign_path = tmp_path / "benign.txt"
        benign_path.write_text("\n".join(f"real{i}.com" for i in range(20)))
        cfg = {
            "data": {
                "raw_dir": str(tmp_path / "raw"),
                "real_datasets": {"n_benign": 20, "n_dga": 30, "n_exfil": 10},
            }
        }
        df = build_real_dataset(cfg, benign_path=str(benign_path), save=True)
        assert len(df) == 60


class TestListAvailableDatasets:
    def test_returns_dict(self):
        datasets = list_available_datasets()
        assert isinstance(datasets, dict)
        assert "umudga" in datasets
        assert "cic_dns_exf" in datasets

    def test_has_metadata(self):
        datasets = list_available_datasets()
        for _name, info in datasets.items():
            assert "description" in info
            assert "url" in info

"""Tests for the drift monitoring module."""

from __future__ import annotations

from src.monitoring.drift import _build_feature_df, _parse_drift_results


class TestBuildFeatureDf:
    def test_returns_dataframe(self):
        domains = ["google.com", "evil.xyz", "abc.def.org"]
        df = _build_feature_df(domains, tokenizer=None, cfg={})
        assert len(df) == 3
        assert df.shape[1] > 0

    def test_consistent_columns(self):
        domains1 = ["google.com", "amazon.com"]
        domains2 = ["evil.xyz", "bad.org", "fake.net"]
        df1 = _build_feature_df(domains1, tokenizer=None, cfg={})
        df2 = _build_feature_df(domains2, tokenizer=None, cfg={})
        assert list(df1.columns) == list(df2.columns)


class TestParseDriftResults:
    def test_empty_report(self):
        result = _parse_drift_results({})
        assert result["dataset_drift_detected"] is False
        assert result["share_drifted_features"] == 0.0
        assert result["drifted_features"] == []

    def test_with_drift_data(self):
        report = {
            "metrics": [
                {
                    "result": {
                        "dataset_drift": True,
                        "share_of_drifted_columns": 0.5,
                        "drift_by_columns": {
                            "entropy": {
                                "column_drifted": True,
                                "stattest_name": "ks",
                                "p_value": 0.001,
                                "drift_score": 0.45,
                            },
                            "length": {
                                "column_drifted": False,
                                "stattest_name": "ks",
                                "p_value": 0.8,
                                "drift_score": 0.02,
                            },
                        },
                    }
                }
            ]
        }
        result = _parse_drift_results(report)
        assert result["dataset_drift_detected"] is True
        assert result["share_drifted_features"] == 0.5
        assert "entropy" in result["drifted_features"]
        assert "length" not in result["drifted_features"]
        assert result["feature_details"]["entropy"]["drifted"] is True
        assert result["feature_details"]["length"]["drifted"] is False

    def test_handles_malformed_data(self):
        """Should not crash on unexpected structure."""
        result = _parse_drift_results({"metrics": [{"result": {}}]})
        assert isinstance(result, dict)

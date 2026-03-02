"""End-to-end integration test with synthetic data."""



from src.data.download import generate_synthetic_dns
from src.data.split import stratified_split
from src.features.ngram import NgramTokenizer
from src.training.models import create_model
from src.training.train import build_features, compute_metrics


class TestEndToEnd:
    def test_full_pipeline(self):
        """Generate data → featurize → split → train → evaluate."""
        # 1. Generate
        df = generate_synthetic_dns(n_benign=300, n_dga=100, n_exfiltration=80, seed=42)
        assert len(df) == 480
        assert set(df["label"].unique()) == {0, 1, 2}

        # 2. Split
        train_df, val_df, test_df = stratified_split(df, seed=42)
        assert len(train_df) + len(val_df) + len(test_df) == len(df)

        # 3. Featurize
        tok = NgramTokenizer(ngram_range=(2, 3), max_features=200, min_df=1)
        cfg = {"features": {"statistical": {"entropy": True, "length": True, "char_distribution": True, "subdomain_stats": True}}}

        X_train = build_features(train_df["domain"].tolist(), tok, cfg, fit=True)
        X_test = build_features(test_df["domain"].tolist(), tok, cfg, fit=False)
        y_train = train_df["label"].values
        y_test = test_df["label"].values

        # 4. Train
        model = create_model("xgboost", {"n_estimators": 20, "max_depth": 4})
        model.fit(X_train, y_train)

        # 5. Evaluate
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        # With injected signal, model should do well
        assert metrics["accuracy"] > 0.7
        assert metrics["f1_macro"] > 0.5

    def test_generate_data_deterministic(self):
        df1 = generate_synthetic_dns(n_benign=100, n_dga=50, n_exfiltration=30, seed=42)
        df2 = generate_synthetic_dns(n_benign=100, n_dga=50, n_exfiltration=30, seed=42)
        assert df1["domain"].tolist() == df2["domain"].tolist()

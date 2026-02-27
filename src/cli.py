"""SeqThreat CLI — sequence-based threat detection pipeline.

Usage
-----
    seqthreat download     -c configs/pipeline.yaml
    seqthreat split        -c configs/pipeline.yaml
    seqthreat train        -c configs/pipeline.yaml
    seqthreat evaluate     -c configs/pipeline.yaml
    seqthreat tune         -c configs/pipeline.yaml
    seqthreat explain      -c configs/pipeline.yaml
    seqthreat adversarial  -c configs/pipeline.yaml
    seqthreat drift        -c configs/pipeline.yaml
    seqthreat serve        --port 8000
    seqthreat pipeline     -c configs/pipeline.yaml   # run all stages
"""

from __future__ import annotations

import click
import yaml


@click.group()
@click.version_option(version="0.2.0", prog_name="seqthreat")
def main():
    """SeqThreat — Sequence-based DNS threat detection."""


# ── Download ────────────────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True))
def download(config: str):
    """Generate / download the DNS classification dataset."""
    from src.data.download import download_dataset

    with open(config) as f:
        cfg = yaml.safe_load(f)

    df = download_dataset(cfg)
    click.echo(f"✓ Dataset: {len(df):,} queries ({df['label'].value_counts().to_dict()})")


# ── Split ───────────────────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True))
def split(config: str):
    """Stratified train / val / test split."""
    from src.data.split import split_and_save

    with open(config) as f:
        cfg = yaml.safe_load(f)

    stats = split_and_save(cfg)
    click.echo(
        f"✓ Split: train={stats['train']:,}, "
        f"val={stats['val']:,}, test={stats['test']:,}"
    )


# ── Train ───────────────────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True))
def train(config: str):
    """Train the DNS threat classifier."""
    from src.training.train import train_pipeline

    with open(config) as f:
        cfg = yaml.safe_load(f)

    metrics = train_pipeline(cfg)
    click.echo(
        f"✓ Train: accuracy={metrics['accuracy']:.4f}, "
        f"F1_macro={metrics['f1_macro']:.4f}"
    )


# ── Evaluate ────────────────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True))
def evaluate(config: str):
    """Evaluate the model on the test split."""
    from src.training.evaluate import evaluate_pipeline

    with open(config) as f:
        cfg = yaml.safe_load(f)

    metrics = evaluate_pipeline(cfg)
    click.echo(
        f"✓ Test: accuracy={metrics['test_accuracy']:.4f}, "
        f"F1_macro={metrics['test_f1_macro']:.4f}"
    )


# ── Serve ───────────────────────────────────────────────────────────


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8000, type=int, help="Bind port")
@click.option("-c", "--config", default=None, type=click.Path(exists=True))
def serve(host: str, port: int, config: str | None):
    """Launch the FastAPI serving endpoint."""
    import uvicorn

    from src.serving.app import _state, app

    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        _state["config"] = cfg.get("serving", {})
    else:
        _state["config"] = {}

    uvicorn.run(app, host=host, port=port)


# ── Pipeline (all stages) ──────────────────────────────────────────


# ── Tune ────────────────────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True))
@click.option("-n", "--n-trials", default=30, type=int, help="Number of Optuna trials")
@click.option("--model-type", default=None, help="Override model type for tuning")
def tune(config: str, n_trials: int, model_type: str | None):
    """Hyperparameter tuning with Optuna (TPE sampler)."""
    from src.training.tune import tune_pipeline

    with open(config) as f:
        cfg = yaml.safe_load(f)

    if model_type:
        cfg.setdefault("model", {})["type"] = model_type
    cfg.setdefault("tune", {})["n_trials"] = n_trials

    result = tune_pipeline(cfg)
    click.echo(
        f"✓ Tune: best {result['metric']}={result['best_score']:.4f} "
        f"in {result['n_trials']} trials"
    )


# ── Explain ─────────────────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True))
def explain(config: str):
    """Generate SHAP explainability analysis."""
    from src.training.explain import generate_explanations

    with open(config) as f:
        cfg = yaml.safe_load(f)

    summary = generate_explanations(cfg)
    click.echo(
        f"✓ Explain: {summary['n_samples_explained']} samples, "
        f"{summary['n_features']} features, "
        f"classes={summary['classes_explained']}"
    )


# ── Adversarial ─────────────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True))
def adversarial(config: str):
    """Run adversarial robustness evaluation."""
    from src.training.adversarial import evaluate_robustness

    with open(config) as f:
        cfg = yaml.safe_load(f)

    summary = evaluate_robustness(cfg)
    click.echo(
        f"✓ Adversarial: robustness_score={summary['robustness_score']:.3f} "
        f"(clean={summary['clean_accuracy']:.3f} → "
        f"adv={summary['overall_adversarial_accuracy']:.3f})"
    )


# ── Drift ───────────────────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True))
def drift(config: str):
    """Run Evidently data drift monitoring."""
    from src.monitoring.drift import monitor_drift

    with open(config) as f:
        cfg = yaml.safe_load(f)

    summary = monitor_drift(cfg)
    n_drifted = len(summary.get("drifted_features", []))
    n_total = len(summary.get("feature_details", {}))
    click.echo(
        f"✓ Drift: dataset_drift={summary['dataset_drift_detected']}, "
        f"drifted_features={n_drifted}/{n_total}"
    )


# ── Deep Learning ───────────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True))
@click.option("--arch", default="char_cnn", type=click.Choice(["char_cnn", "char_lstm"]))
def deep_train(config: str, arch: str):
    """Train a character-level deep learning model."""
    from src.training.deep_model import train_deep_model

    with open(config) as f:
        cfg = yaml.safe_load(f)

    summary = train_deep_model(cfg, arch=arch)
    click.echo(
        f"✓ Deep train ({arch}): val_acc={summary['final_val_acc']:.4f}, "
        f"epochs={summary['epochs_trained']}, saved={summary['model_path']}"
    )


# ── Real Datasets ───────────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True))
@click.option("--benign", default=None, type=click.Path(), help="Path to benign domain list")
@click.option("--dga", default=None, type=click.Path(), help="Path to DGA domain file")
@click.option("--exfil", default=None, type=click.Path(), help="Path to exfiltration CSV")
def real_data(config: str, benign: str | None, dga: str | None, exfil: str | None):
    """Build dataset from real-world DNS threat data."""
    from src.data.real_datasets import build_real_dataset

    with open(config) as f:
        cfg = yaml.safe_load(f)

    df = build_real_dataset(cfg, benign_path=benign, dga_path=dga, exfil_path=exfil)
    click.echo(f"✓ Real data: {len(df):,} queries ({df['label'].value_counts().to_dict()})")


# ── Full pipeline ──────────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True))
def pipeline(config: str):
    """Run the full pipeline: download → split → train → evaluate."""
    from src.data.download import download_dataset
    from src.data.split import split_and_save
    from src.training.evaluate import evaluate_pipeline
    from src.training.train import train_pipeline

    with open(config) as f:
        cfg = yaml.safe_load(f)

    click.echo("── Stage 1: Download ──")
    df = download_dataset(cfg)
    click.echo(f"  ✓ {len(df):,} queries")

    click.echo("── Stage 2: Split ──")
    stats = split_and_save(cfg)
    click.echo(f"  ✓ train={stats['train']:,}, val={stats['val']:,}, test={stats['test']:,}")

    click.echo("── Stage 3: Train ──")
    train_metrics = train_pipeline(cfg)
    click.echo(f"  ✓ accuracy={train_metrics['accuracy']:.4f}, F1_macro={train_metrics['f1_macro']:.4f}")

    click.echo("── Stage 4: Evaluate ──")
    test_metrics = evaluate_pipeline(cfg)
    click.echo(f"  ✓ accuracy={test_metrics['test_accuracy']:.4f}, F1_macro={test_metrics['test_f1_macro']:.4f}")

    click.echo("\n✓ Pipeline complete!")

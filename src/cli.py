"""SeqThreat CLI — sequence-based threat detection pipeline.

Usage
-----
    seqthreat download  -c configs/pipeline.yaml
    seqthreat split     -c configs/pipeline.yaml
    seqthreat train     -c configs/pipeline.yaml
    seqthreat evaluate  -c configs/pipeline.yaml
    seqthreat serve     --port 8000
    seqthreat pipeline  -c configs/pipeline.yaml   # run all stages
"""

from __future__ import annotations

import click
import yaml


@click.group()
@click.version_option(version="0.1.0", prog_name="seqthreat")
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

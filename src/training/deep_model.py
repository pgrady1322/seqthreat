"""Character-level deep learning baselines for DNS classification.

Provides a lightweight character-level CNN and LSTM that ingest raw
domain strings (no hand-crafted features) so you can compare
representation-learning vs. n-gram approaches.

Both architectures share the same ``CharVocab`` character encoder,
training loop, and evaluation API so they are drop-in replacements
for the sklearn models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ── Character vocabulary ────────────────────────────────────────────

PAD_IDX = 0
UNK_IDX = 1

CHARS = (
    list("abcdefghijklmnopqrstuvwxyz")
    + list("0123456789")
    + list(".-_:/")
)
CHAR2IDX: dict[str, int] = {c: i + 2 for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS) + 2  # +PAD, +UNK
DEFAULT_MAX_LEN = 128


def encode_domain(domain: str, max_len: int = DEFAULT_MAX_LEN) -> list[int]:
    """Encode a domain string into a fixed-length integer sequence."""
    domain = domain.lower()[:max_len]
    ids = [CHAR2IDX.get(c, UNK_IDX) for c in domain]
    # Pad to max_len
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids


# ── Dataset ─────────────────────────────────────────────────────────


class DomainDataset(Dataset):
    """PyTorch dataset wrapping domain strings + integer labels."""

    def __init__(self, domains: list[str], labels: list[int], max_len: int = DEFAULT_MAX_LEN):
        self.domains = domains
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.domains)

    def __getitem__(self, idx: int):
        ids = encode_domain(self.domains[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# ── Models ──────────────────────────────────────────────────────────


class CharCNN(nn.Module):
    """1-D convolutional network over character embeddings.

    Architecture: Embedding → Conv1d(×3 filter widths) → GlobalMaxPool → FC.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 32,
        num_filters: int = 64,
        filter_widths: tuple[int, ...] = (3, 4, 5),
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, w) for w in filter_widths]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_widths), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        conv_outs = [torch.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        cat = torch.cat(conv_outs, dim=1)
        return self.fc(self.dropout(cat))


class CharLSTM(nn.Module):
    """Bidirectional LSTM over character embeddings.

    Architecture: Embedding → BiLSTM → last-hidden concat → FC.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        # Concatenate forward and backward last hidden states
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(hidden))


# ── Factory ─────────────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, type] = {
    "char_cnn": CharCNN,
    "char_lstm": CharLSTM,
}


def create_deep_model(
    arch: Literal["char_cnn", "char_lstm"] = "char_cnn",
    **kwargs,
) -> nn.Module:
    """Instantiate a deep learning model by name."""
    if arch not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown arch: {arch}. Choose from {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[arch](**kwargs)


# ── Training loop ───────────────────────────────────────────────────


def train_deep_model(
    cfg: dict,
    arch: Literal["char_cnn", "char_lstm"] = "char_cnn",
) -> dict:
    """Train a character-level deep model end-to-end.

    Parameters
    ----------
    cfg : dict
        Pipeline config with ``data.splits_dir``, ``deep`` section, etc.
    arch : str
        ``"char_cnn"`` or ``"char_lstm"``.

    Returns
    -------
    dict
        Training summary (final train/val loss, accuracy, path to saved model).
    """
    data_cfg = cfg.get("data", {})
    deep_cfg = cfg.get("deep", {})
    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    out_dir = Path(deep_cfg.get("output_dir", "models/deep"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparams
    lr = deep_cfg.get("lr", 1e-3)
    epochs = deep_cfg.get("epochs", 10)
    batch_size = deep_cfg.get("batch_size", 128)
    max_len = deep_cfg.get("max_len", DEFAULT_MAX_LEN)
    device_str = deep_cfg.get("device", "cpu")
    device = torch.device(device_str)

    # Load splits
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")

    train_ds = DomainDataset(train_df["domain"].tolist(), train_df["label"].tolist(), max_len)
    val_ds = DomainDataset(val_df["domain"].tolist(), val_df["label"].tolist(), max_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model_kwargs = deep_cfg.get("model_kwargs", {})
    model = create_deep_model(arch, **model_kwargs).to(device)
    logger.info("Model: %s  |  params: %d", arch, sum(p.numel() for p in model.parameters()))

    # Class weights for imbalanced data
    counts = np.bincount(train_df["label"].values)
    weights = 1.0 / counts.astype(float)
    weights = weights / weights.sum() * len(weights)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    history: list[dict] = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += X_batch.size(0)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += X_batch.size(0)

        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(avg_val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })
        logger.info(
            "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  train_acc=%.3f  val_acc=%.3f",
            epoch, epochs, avg_train_loss, avg_val_loss, train_acc, val_acc,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), out_dir / f"{arch}_best.pt")

    # Save final
    torch.save(model.state_dict(), out_dir / f"{arch}_final.pt")
    pd.DataFrame(history).to_csv(out_dir / f"{arch}_history.csv", index=False)

    return {
        "arch": arch,
        "best_val_loss": best_val_loss,
        "final_val_acc": history[-1]["val_acc"] if history else 0.0,
        "epochs_trained": len(history),
        "model_path": str(out_dir / f"{arch}_best.pt"),
    }


# ── Prediction helper ──────────────────────────────────────────────


def predict_deep(
    model: nn.Module,
    domains: list[str],
    max_len: int = DEFAULT_MAX_LEN,
    device: str = "cpu",
) -> np.ndarray:
    """Run inference on a list of domains, return predicted labels."""
    model.eval()
    dev = torch.device(device)
    ids = [encode_domain(d, max_len) for d in domains]
    X = torch.tensor(ids, dtype=torch.long).to(dev)
    with torch.no_grad():
        logits = model(X)
    return logits.argmax(1).cpu().numpy()

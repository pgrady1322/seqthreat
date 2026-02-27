# SeqThreat

**Sequence-based DNS threat detection using character n-gram feature engineering** — applying genomics k-mer techniques to cybersecurity.

Classifies DNS queries as **benign**, **DGA** (domain generation algorithm), or **exfiltration** (data tunnelling) using character n-gram TF-IDF features and statistical analysis.

---

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Generate /  │───▶│  Stratified │───▶│    Train     │───▶│  Evaluate   │
│  Download    │    │    Split    │    │  (MLflow)    │    │  (Test Set) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                                     │                    │
       │                              ┌──────┤              ┌─────┤
       ▼                              ▼      ▼              ▼     ▼
┌─────────────┐                ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Real Dataset│                │ N-gram │ │ Stat   │ │ SHAP   │ │ Adver- │
│ Integration │                │ TF-IDF │ │Features│ │Explain │ │ sarial │
└─────────────┘                └────────┘ └────────┘ └────────┘ └────────┘
                                    │          │
       ┌────────────────────────────┤          │
       ▼                            ▼          ▼
┌─────────────┐              ┌─────────────┐ ┌─────────────┐
│  Optuna HP  │              │  FastAPI     │ │  Evidently  │
│  Tuning     │              │  Serving     │ │  Drift Mon  │
└─────────────┘              └─────────────┘ └─────────────┘
                                               
       ┌─────────────┐
       │  Deep Learn  │  ← Char-CNN / Char-LSTM baseline
       │  Baseline    │
       └─────────────┘
```

### Key Concept: K-mers → N-grams

| Genomics | SeqThreat (Cybersecurity) |
|----------|--------------------------|
| DNA sequence (ATCGATCG...) | DNS query (api.google.com) |
| K-mer extraction (k=3,4,5) | Character n-gram extraction (n=2,3,4) |
| K-mer frequency table | TF-IDF weighted n-gram vectors |
| Shannon entropy (genome quality) | Shannon entropy (exfiltration detection) |
| Imbalanced classes (rare variants) | Imbalanced classes (rare threats) |

### Stack

| Component             | Technology                                  |
|-----------------------|---------------------------------------------|
| Feature extraction    | Character n-gram TF-IDF (sklearn)           |
| Statistical features  | Shannon entropy, char distributions         |
| Models                | XGBoost, Random Forest, Logistic Regression |
| Deep learning         | Character-level CNN/LSTM (PyTorch)          |
| HP tuning             | Optuna (TPE sampler)                        |
| Explainability        | SHAP (TreeExplainer / KernelExplainer)      |
| Adversarial testing   | Homoglyph, typo, encoding mutation attacks  |
| Drift monitoring      | Evidently AI (KS, PSI, JS divergence)       |
| Experiment tracking   | MLflow 2.10+                                |
| Model serving         | FastAPI + Uvicorn                           |
| CI/CD                 | GitHub Actions                              |
| Containerization      | Docker (multi-stage)                        |

---

## Threat Classes

| Label | Class | Description |
|-------|-------|-------------|
| 0 | **Benign** | Legitimate DNS lookups (google.com, api.github.com) |
| 1 | **DGA** | Domain Generation Algorithm — random-looking domains for malware C2 |
| 2 | **Exfiltration** | Data tunnelling — base64/hex encoded payloads in subdomains |

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/pgrady1322/seqthreat.git
cd seqthreat
pip install -e ".[dev]"

# Optional extras
pip install -e ".[all]"    # tune + explain + drift + deep
pip install -e ".[tune]"   # Optuna only
pip install -e ".[deep]"   # PyTorch only
```

### 2. Run the Full Pipeline

```bash
# Via CLI (generates synthetic data — no external downloads needed)
seqthreat pipeline -c configs/pipeline.yaml

# Or step by step
seqthreat download -c configs/pipeline.yaml
seqthreat split    -c configs/pipeline.yaml
seqthreat train    -c configs/pipeline.yaml
seqthreat evaluate -c configs/pipeline.yaml

# Advanced features
seqthreat tune         -c configs/pipeline.yaml -n 50
seqthreat explain      -c configs/pipeline.yaml
seqthreat adversarial  -c configs/pipeline.yaml
seqthreat drift        -c configs/pipeline.yaml
seqthreat deep-train   -c configs/pipeline.yaml --arch char_cnn
seqthreat real-data    -c configs/pipeline.yaml --dga data/external/dga.csv
```

### 3. Serve the Model

```bash
seqthreat serve --port 8000

# Or via Makefile
make serve
```

### 4. Classify a Domain

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"domains": ["google.com", "a1b2c3d4e5f6g7h8.xyz", "aGVsbG8gd29ybGQ.evil.com"]}'
```

---

## Feature Engineering

### N-gram Features (analogous to k-mers)

Character n-grams of sizes 2–4 are extracted from each domain string, weighted by TF-IDF:

```
"api.google.com" → bigrams: [" a", "ap", "pi", "i ", " g", "go", "oo", ...]
                 → trigrams: [" ap", "api", "pi ", " go", "goo", "oog", ...]
                 → 4-grams:  [" api", "api ", " goo", "goog", "oogl", ...]
```

### Statistical Features

| Feature Group | Features |
|---------------|----------|
| **Entropy** | Shannon entropy (full domain + longest label) |
| **Length** | Total length, label count, max/mean/std label length |
| **Char distribution** | Alpha, digit, special, uppercase, vowel, consonant, hex char, unique char ratios |
| **Subdomain structure** | Depth, numeric labels, long labels, base64-like, hex-like patterns |

---

## Project Structure

```
seqthreat/
├── configs/
│   └── pipeline.yaml            # Central configuration
├── src/
│   ├── cli.py                   # Click CLI entry point
│   ├── data/
│   │   ├── download.py          # Synthetic dataset generation
│   │   ├── real_datasets.py     # Real-world dataset loaders
│   │   └── split.py             # Stratified train/val/test split
│   ├── features/
│   │   ├── ngram.py             # Character n-gram tokenizer (TF-IDF)
│   │   └── statistical.py       # Entropy, length, char distribution features
│   ├── training/
│   │   ├── models.py            # Model factory (XGBoost, RF, LR)
│   │   ├── train.py             # Training pipeline with MLflow
│   │   ├── evaluate.py          # Test evaluation + metrics
│   │   ├── tune.py              # Optuna hyperparameter tuning
│   │   ├── explain.py           # SHAP explainability analysis
│   │   ├── adversarial.py       # Adversarial robustness testing
│   │   └── deep_model.py        # Char-CNN / Char-LSTM baselines
│   ├── monitoring/
│   │   └── drift.py             # Evidently data drift detection
│   └── serving/
│       └── app.py               # FastAPI serving endpoint
├── notebooks/
│   └── 01_showcase.ipynb        # End-to-end demo notebook
├── tests/                       # pytest test suite
├── .github/workflows/ci.yml     # CI: lint + test + Docker build
├── Dockerfile                   # Multi-stage container
├── Makefile                     # Developer targets
└── pyproject.toml               # Python project config
```

---

## API Endpoints

| Method | Path          | Description               |
|--------|---------------|---------------------------|
| GET    | `/health`     | Service health + model status |
| POST   | `/predict`    | Classify DNS domains       |
| GET    | `/model/info` | Model metadata + params    |

---

## Advanced Features

### Hyperparameter Tuning (Optuna)

Bayesian optimization with TPE sampler, per-model search spaces, StratifiedKFold CV, and MLflow nested run logging:

```bash
seqthreat tune -c configs/pipeline.yaml -n 50
```

### SHAP Explainability

Per-class feature importance using TreeExplainer (XGBoost/RF) or KernelExplainer (LR). Generates bar plots, beeswarm plots, and importance CSVs:

```bash
seqthreat explain -c configs/pipeline.yaml
```

### Real Dataset Integration

Plug in real-world DNS threat datasets (UMUDGA-2021, CIC-Bell-DNS-EXF-2021, Alexa top-1M) via file paths or config:

```bash
seqthreat real-data -c configs/pipeline.yaml --dga data/external/umudga.csv
```

### Deep Learning Baselines

Character-level CNN and BiLSTM models that learn directly from raw domain strings (no hand-crafted features):

```bash
seqthreat deep-train -c configs/pipeline.yaml --arch char_cnn
seqthreat deep-train -c configs/pipeline.yaml --arch char_lstm
```

### Adversarial Robustness

Six mutation strategies (homoglyphs, typos, subdomain insertion, label shuffle, noise injection, encoding corruption) to test evasion resilience:

```bash
seqthreat adversarial -c configs/pipeline.yaml
```

### Evidently Drift Monitoring

Detect feature distribution shift using KS / PSI / JS divergence tests, with HTML reports:

```bash
seqthreat drift -c configs/pipeline.yaml
```

---

## Testing

```bash
pytest tests/ -v
```

Tests cover:
- **N-gram extraction** — character n-grams, frequency, preprocessing, TF-IDF tokenizer
- **Statistical features** — entropy, length, char distribution, subdomain analysis
- **Model factory** — creation, class weights, fit/predict for all model types
- **Stratified split** — proportions, no overlap, stratification, reproducibility
- **Training helpers** — metrics computation, feature building
- **FastAPI serving** — health, predict, batch, error handling, model info
- **Integration** — full end-to-end pipeline with synthetic data

---

## Configuration

All parameters in [`configs/pipeline.yaml`](configs/pipeline.yaml):

- **data**: synthetic generation params (n_benign, n_dga, n_exfiltration)
- **features**: n-gram range, max features, TF-IDF params, statistical feature flags
- **split**: train/val/test fractions, stratification, seed
- **model**: type + hyperparameters
- **mlflow**: tracking URI, experiment name
- **serving**: host, port, model/vectorizer paths

---

## License

MIT — see [LICENSE](LICENSE).

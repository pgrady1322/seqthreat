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
                                             │
                   ┌─────────────────────────┤
                   ▼                         ▼
            ┌─────────────┐          ┌─────────────┐
            │  N-gram     │          │ Statistical  │
            │  TF-IDF     │          │  Features    │
            └─────────────┘          └─────────────┘
                   │                         │
                   └────────┬────────────────┘
                            ▼
                     ┌─────────────┐
                     │  FastAPI    │
                     │  Serving    │
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

| Component           | Technology         |
|---------------------|--------------------|
| Feature extraction  | Character n-gram TF-IDF (sklearn) |
| Statistical features| Shannon entropy, char distributions |
| Experiment tracking | MLflow 2.10+       |
| Model serving       | FastAPI + Uvicorn   |
| CI/CD               | GitHub Actions      |
| Containerization    | Docker (multi-stage)|
| Models              | XGBoost, Random Forest, Logistic Regression |

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
│   └── pipeline.yaml          # Central configuration
├── src/
│   ├── cli.py                 # Click CLI entry point
│   ├── data/
│   │   ├── download.py        # Synthetic dataset generation
│   │   └── split.py           # Stratified train/val/test split
│   ├── features/
│   │   ├── ngram.py           # Character n-gram tokenizer (TF-IDF)
│   │   └── statistical.py     # Entropy, length, char distribution features
│   ├── training/
│   │   ├── models.py          # Model factory (XGBoost, RF, LR)
│   │   ├── train.py           # Training pipeline with MLflow
│   │   └── evaluate.py        # Test evaluation + metrics
│   └── serving/
│       └── app.py             # FastAPI serving endpoint
├── tests/                     # pytest test suite
├── .github/workflows/ci.yml   # CI: lint + test + Docker build
├── Dockerfile                 # Multi-stage container
├── Makefile                   # Developer targets
└── pyproject.toml             # Python project config
```

---

## API Endpoints

| Method | Path          | Description               |
|--------|---------------|---------------------------|
| GET    | `/health`     | Service health + model status |
| POST   | `/predict`    | Classify DNS domains       |
| GET    | `/model/info` | Model metadata + params    |

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

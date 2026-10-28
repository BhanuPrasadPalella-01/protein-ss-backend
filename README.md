# Protein Secondary Structure Prediction: API

FastAPI service that predicts, for every amino acid in a protein, whether it folds into a
**helix (H)**, **strand (E)** or **coil (C)**. It serves three trained PyTorch models and powers the
[web app](https://protein-secondary-structure-fronten.vercel.app/)
([frontend code](https://github.com/BhanuPrasadPalella-01/protein-ss-frontend)).

## Models

Trained on CullPDB (5,926 proteins) and tested on the CB513 benchmark. Q3 accuracy is the mean of 3 training
seeds; F1 scores are from a single run.

| Model | Input | Q3 accuracy | Helix F1 | Strand F1 | Coil F1 | Parameters |
|---|---|---:|---:|---:|---:|---:|
| Baseline BiLSTM | sequence | 68.73% ± 0.12 | 0.727 | 0.593 | 0.704 | 51,267 |
| + PSSM | sequence + evolutionary profile | 79.30% ± 0.10 | 0.843 | 0.739 | 0.780 | 62,531 |
| + PSSM + self-attention | sequence + profile | **80.09% ± 0.07** | 0.852 | 0.736 | 0.788 | 128,835 |

The evolutionary profile (PSSM) does most of the work (+10.6 points); 4-head self-attention with a residual
connection and LayerNorm adds a further +0.8.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/predict/baseline` | Predict a pasted sequence: `{"sequence": "MKTAYIAKQR..."}`. Only the 20 standard amino-acid letters, max 700 residues. |
| `GET` | `/samples` | The CB513 test proteins available in the explorer |
| `GET` | `/samples/{id}/predict` | PSSM and attention-model predictions for a sample, with match rates against the true structure |
| `GET` | `/samples/{id}/trace` | The attention model's intermediate values for a window of a sample (PSSM input, embeddings, BiLSTM output, per-head attention weights, output probabilities). Optional `start` and `length` (4–24, default 12); without `start` it picks a window with clean helix/strand/coil runs in the true structure |
| `POST` | `/fold` | 3D structure for a sequence from [ESMFold](https://esmatlas.com): `{"sequence": "..."}`, max 400 residues. Returns PDB text plus per-residue confidence (pLDDT, 0–100). Cached in memory |
| `GET` | `/comparison` | Benchmark numbers for the three models |
| `GET`/`HEAD` | `/health` | Liveness check (plain `ok`) |

Pasted sequences use the **baseline** model, because the stronger models need a PSSM, which can't be computed
for an arbitrary sequence without a database search. Its live output gets a cosmetic clean-up of isolated
one-residue helix/strand predictions; sample predictions are returned raw so the match rates line up with the
benchmark.

### About `/fold`

Our models predict a label per residue, not 3D coordinates. `/fold` forwards the sequence to Meta's public
ESMFold API (override with the `ESMFOLD_URL` environment variable) and returns its predicted structure, which the
web app colors with our model's labels. Sequences sent to `/fold` therefore leave this server. The public API
can be slow or unavailable; the endpoint then answers 502/503 with a readable message.

## Run locally

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

```bash
uvicorn main:app --reload
```

Then open <http://localhost:8000/docs> for the interactive API docs.

Run the tests (they also run on every push via GitHub Actions):

```bash
pip install pytest httpx && python -m pytest
```

## Deploy

On Render, create a Python web service with:

* **Build command:** `pip install -r requirements.txt`
* **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

`requirements.txt` installs the CPU-only PyTorch build. The free tier sleeps when idle; the frontend pings
`/health` on load to wake it early.

## Files

```
main.py                   API, input validation, model loading
model.py                  BiLSTM, BiLSTM + PSSM and BiLSTM + PSSM + attention models
models/*.pt               Trained weights
data/sample_proteins.json CB513 test proteins (sequence, true structure, PSSM) for the explorer
extract_samples.py        Rebuilds sample_proteins.json from the preprocessed test set
```

## Data

* **CullPDB** (training) and **CB513** (test) with PSI-BLAST profiles, as released by Zhou & Troyanskaya,
  *Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure
  Prediction*, ICML 2014.
* 8-state DSSP labels are reduced to 3 states: H, G, I → helix; E, B → strand; everything else → coil.

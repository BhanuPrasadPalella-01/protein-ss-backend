import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from model import BiLSTMTagger, BiLSTMTaggerPSSM, BiLSTMAttentionPSSM

app = FastAPI(title="Protein Secondary Structure Prediction API")

# Allow the frontend (any origin, for now) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")  # Render free tier has no GPU

residues = ['A','C','E','D','G','F','I','H','K','M','L','N','Q','P','S','R','T','W','V','Y','X','NoSeq']
aa_to_idx = {aa: i for i, aa in enumerate(residues)}
class_names = ['H', 'E', 'C']
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')

# ---- Load all 3 models once at startup ----
baseline_model = BiLSTMTagger().to(device)
baseline_model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
baseline_model.eval()

pssm_model = BiLSTMTaggerPSSM().to(device)
pssm_model.load_state_dict(torch.load('models/best_model_pssm.pt', map_location=device))
pssm_model.eval()

attention_model = BiLSTMAttentionPSSM().to(device)
attention_model.load_state_dict(torch.load('models/best_model_attention.pt', map_location=device))
attention_model.eval()

# ---- Load sample proteins ----
with open('data/sample_proteins.json') as f:
    SAMPLES = {s["id"]: s for s in json.load(f)}


class SequenceRequest(BaseModel):
    sequence: str


def compute_percentages(structure_str):
    total = len(structure_str)
    return {
        "H": round(100 * structure_str.count("H") / total, 1),
        "E": round(100 * structure_str.count("E") / total, 1),
        "C": round(100 * structure_str.count("C") / total, 1),
    }


def smooth_structure(structure_str):
    """Cosmetic post-processing: replace isolated single-residue H or E segments
    (e.g. the lone E in "HHEHH") with their neighbouring state, or C when the
    neighbours disagree. This only tidies the live /predict/baseline output; it is
    not part of the model. Sample predictions and /comparison use raw predictions.
    """
    s = list(structure_str)
    n = len(s)
    for i, label in enumerate(structure_str):
        if label not in "HE":
            continue
        left = structure_str[i - 1] if i > 0 else None
        right = structure_str[i + 1] if i < n - 1 else None
        if left == label or right == label:
            continue
        if left is None or right is None:
            s[i] = left or right or label  # sequence edge: take the only neighbour
        else:
            s[i] = left if left == right else "C"
    return "".join(s)


# Cheap liveness check: no auth, no model call. Covered by the wildcard CORS policy above.
# HEAD is accepted too so uptime pingers can keep the Render instance warm.
@app.api_route("/health", methods=["GET", "HEAD"], response_class=PlainTextResponse)
def health():
    return "ok"


def clean_sequence(raw, max_length):
    """Trim and upper-case a submitted sequence; reject anything but the 20 standard amino acids."""
    seq = raw.strip().upper()
    if not seq:
        raise HTTPException(400, "Sequence cannot be empty")
    invalid = sorted(set(seq) - STANDARD_AA)
    if invalid:
        raise HTTPException(
            400,
            f"Invalid characters in sequence: {' '.join(repr(c) for c in invalid)}. "
            "Only the 20 standard amino-acid letters (ACDEFGHIKLMNPQRSTVWY) are allowed.",
        )
    if len(seq) > max_length:
        raise HTTPException(400, f"Sequence too long (max {max_length} residues)")
    return seq


@app.post("/predict/baseline")
def predict_baseline(req: SequenceRequest):
    seq = clean_sequence(req.sequence, 700)

    indices = [aa_to_idx.get(ch, aa_to_idx['X']) for ch in seq]
    X = torch.tensor([indices], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = baseline_model(X)
        preds = logits.argmax(dim=-1)[0]

    predicted_structure = smooth_structure(''.join(class_names[p] for p in preds.cpu().numpy()))

    return {
        "sequence": seq,
        "predicted_structure": predicted_structure,
        "percentages": compute_percentages(predicted_structure),
        "model_used": "baseline",
    }


@app.get("/samples")
def list_samples():
    return [
        {"id": s["id"], "length": s["length"], "sequence": s["sequence"], "true_structure": s["true_structure"]}
        for s in SAMPLES.values()
    ]


@app.get("/samples/{sample_id}/predict")
def predict_sample(sample_id: str):
    if sample_id not in SAMPLES:
        raise HTTPException(404, f"Sample '{sample_id}' not found")

    sample = SAMPLES[sample_id]
    seq = sample["sequence"]
    true_structure = sample["true_structure"]
    pssm_array = np.array(sample["pssm"], dtype=np.float32)

    indices = [aa_to_idx.get(ch, aa_to_idx['X']) for ch in seq]
    X = torch.tensor([indices], dtype=torch.long).to(device)
    PSSM = torch.tensor([pssm_array], dtype=torch.float32).to(device)

    with torch.no_grad():
        pssm_logits = pssm_model(X, PSSM)
        pssm_preds = pssm_logits.argmax(dim=-1)[0]

        attn_logits = attention_model(X, PSSM, key_padding_mask=None)
        attn_preds = attn_logits.argmax(dim=-1)[0]

    # Raw (unsmoothed) predictions so match rates line up with the reported benchmark.
    pssm_structure = ''.join(class_names[p] for p in pssm_preds.cpu().numpy())
    attn_structure = ''.join(class_names[p] for p in attn_preds.cpu().numpy())

    def match_rate(pred_str):
        matches = sum(a == b for a, b in zip(pred_str, true_structure))
        return round(100 * matches / len(true_structure), 1)

    return {
        "id": sample_id,
        "sequence": seq,
        "true_structure": true_structure,
        "pssm_prediction": pssm_structure,
        "pssm_match_accuracy": match_rate(pssm_structure),
        "attention_prediction": attn_structure,
        "attention_match_accuracy": match_rate(attn_structure),
    }


TRACE_DEFAULT_LENGTH = 12


def _r(t, digits=3):
    """Tensor -> nested lists rounded for a compact JSON payload."""
    return np.round(t.cpu().numpy().astype(np.float64), digits).tolist()


def clearest_window(structure, length):
    """Default window: the one whose true structure has the most kinds of clean runs (3+ residues),
    then the fewest short fragments (earliest wins). Uses only the true labels, not the model's
    predictions, so the default isn't picked for accuracy."""
    def score(i):
        runs = re.findall(r"((.)\2*)", structure[i:i + length])
        clean = {label for run, label in runs if len(run) >= 3}
        fragments = sum(len(run) < 3 for run, _ in runs)
        return (len(clean), -fragments)
    return max(range(len(structure) - length + 1), key=lambda i: (score(i), -i))


@app.get("/samples/{sample_id}/trace")
def trace_sample(sample_id: str, start: int | None = None, length: int = TRACE_DEFAULT_LENGTH):
    """Intermediate values of the PSSM + attention model for a window of a sample protein.

    Runs the same layers as BiLSTMAttentionPSSM.forward one at a time so the activations between
    them can be returned: the PSSM input, embeddings, BiLSTM output, per-head attention weights and
    output probabilities. Attention weights are the model's own (each row sums to 1 over the whole
    protein); only the columns inside the window are returned, plus the share that falls outside it.
    """
    if sample_id not in SAMPLES:
        raise HTTPException(404, f"Sample '{sample_id}' not found")
    sample = SAMPLES[sample_id]
    seq, true_structure = sample["sequence"], sample["true_structure"]
    n = len(seq)
    if not 4 <= length <= 24:
        raise HTTPException(400, "length must be between 4 and 24")
    length = min(length, n)
    if start is None:
        start = clearest_window(true_structure, length)
    if not 0 <= start <= n - length:
        raise HTTPException(400, f"start must be between 0 and {n - length} for this sample")
    end = start + length

    indices = [aa_to_idx.get(ch, aa_to_idx['X']) for ch in seq]
    X = torch.tensor([indices], dtype=torch.long).to(device)
    PSSM = torch.tensor(np.array([sample["pssm"]], dtype=np.float32)).to(device)

    m = attention_model
    with torch.no_grad():
        embedded = m.embedding(X)
        combined = torch.cat([embedded, PSSM], dim=-1)
        lstm_out, _ = m.lstm(combined)
        attn_out, weights = m.attention(lstm_out, lstm_out, lstm_out,
                                        need_weights=True, average_attn_weights=False)
        logits = m.classifier(m.layer_norm(lstm_out + attn_out))
        probs = torch.softmax(logits, dim=-1)[0]

    window_weights = weights[0][:, start:end, :]          # (heads, length, n)
    inside = window_weights[:, :, start:end]              # (heads, length, length)
    predicted = ''.join(class_names[p] for p in probs.argmax(dim=-1).cpu().numpy())

    return {
        "id": sample_id,
        "sequence": seq,
        "true_structure": true_structure,
        "predicted_structure": predicted,
        "start": start,
        "length": length,
        "window": {
            "residues": seq[start:end],
            "true": true_structure[start:end],
            "predicted": predicted[start:end],
            "pssm": _r(PSSM[0, start:end]),
            "embedding": _r(embedded[0, start:end]),
            "bilstm": _r(lstm_out[0, start:end]),
            "attention": _r(inside),
            "attention_outside": _r(1 - inside.sum(dim=-1)),
            "probabilities": _r(probs[start:end]),
        },
    }


# ---- 3D structure via ESMFold ----
# Our models only predict a label per residue, not coordinates. For the 3D view we ask Meta's public
# ESMFold API for a predicted structure and return it as PDB text; the frontend colors it with our labels.
ESMFOLD_URL = os.environ.get("ESMFOLD_URL", "https://api.esmatlas.com/foldSequence/v1/pdb/")
FOLD_MAX_RESIDUES = 400   # the public ESMFold API's limit
FOLD_TIMEOUT_S = 60
FOLD_CACHE_SIZE = 64
_fold_cache = OrderedDict()
_fold_cache_lock = threading.Lock()


def fetch_esmfold(seq):
    """POST the sequence to ESMFold and return PDB text. Raises HTTPException on any upstream failure."""
    request = urllib.request.Request(ESMFOLD_URL, data=seq.encode(), method="POST",
                                     headers={"Content-Type": "text/plain", "User-Agent": "protein-ss-backend"})
    try:
        with urllib.request.urlopen(request, timeout=FOLD_TIMEOUT_S) as response:
            pdb = response.read().decode()
    except urllib.error.HTTPError as err:
        raise HTTPException(502, f"The ESMFold service returned an error ({err.code}). Try again later.")
    except (urllib.error.URLError, TimeoutError):
        raise HTTPException(503, "Couldn't reach the ESMFold structure service. Try again in a minute.")
    if "ATOM" not in pdb:
        raise HTTPException(502, "The ESMFold service returned an unexpected response.")
    return pdb


def plddt_per_residue(pdb):
    """ESMFold stores per-residue confidence (pLDDT) in the B-factor column; read it from the CA atoms, as 0-100."""
    values = [float(line[60:66]) for line in pdb.splitlines()
              if line.startswith("ATOM") and line[12:16] == " CA "]
    if values and max(values) <= 1:
        values = [v * 100 for v in values]
    return [round(v, 1) for v in values]


@app.post("/fold")
def fold(req: SequenceRequest):
    seq = clean_sequence(req.sequence, FOLD_MAX_RESIDUES)
    with _fold_cache_lock:
        if seq in _fold_cache:
            _fold_cache.move_to_end(seq)
            return _fold_cache[seq]
    pdb = fetch_esmfold(seq)
    plddt = plddt_per_residue(pdb)
    result = {
        "sequence": seq,
        "pdb": pdb,
        "plddt": plddt,
        "mean_plddt": round(sum(plddt) / len(plddt), 1) if plddt else None,
        "source": "ESMFold (Meta) via api.esmatlas.com",
    }
    with _fold_cache_lock:
        _fold_cache[seq] = result
        while len(_fold_cache) > FOLD_CACHE_SIZE:
            _fold_cache.popitem(last=False)
    return result


@app.get("/comparison")
def get_comparison():
    return {
        "models": [
            {
                "name": "Baseline",
                "q3_accuracy": 68.73,
                "std_dev": 0.12,
                "helix_f1": 0.7266,
                "strand_f1": 0.5925,
                "coil_f1": 0.7042,
                "macro_f1": 0.6744,
                "params": 51267,
            },
            {
                "name": "+PSSM",
                "q3_accuracy": 79.30,
                "std_dev": 0.10,
                "helix_f1": 0.8432,
                "strand_f1": 0.7390,
                "coil_f1": 0.7799,
                "macro_f1": 0.7874,
                "params": 62531,
            },
            {
                "name": "+Attention",
                "q3_accuracy": 80.09,
                "std_dev": 0.07,
                "helix_f1": 0.8518,
                "strand_f1": 0.7355,
                "coil_f1": 0.7881,
                "macro_f1": 0.7918,
                "params": 128835,
            },
        ]
    }

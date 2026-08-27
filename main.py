import json
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/baseline")
def predict_baseline(req: SequenceRequest):
    seq = req.sequence.strip().upper()
    if not seq:
        raise HTTPException(400, "Sequence cannot be empty")
    if len(seq) > 700:
        raise HTTPException(400, "Sequence too long (max 700 residues)")

    indices = [aa_to_idx.get(ch, aa_to_idx['X']) for ch in seq]
    X = torch.tensor([indices], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = baseline_model(X)
        preds = logits.argmax(dim=-1)[0]

    predicted_structure = ''.join(class_names[p] for p in preds.cpu().numpy())

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

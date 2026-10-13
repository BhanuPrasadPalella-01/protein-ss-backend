"""Extract a few CB513 test proteins (sequence, true structure, PSSM) into
data/sample_proteins.json for the API's /samples endpoints.

Usage:
    python extract_samples.py --data-dir path/to/data/processed

--data-dir is the folder with X_test.npy, Y_test.npy and PSSM_test.npy produced
by the training repo's preprocessing (default: $PROTEIN_SS_DATA, then
../protein-ss-prediction/data/processed).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

residues = ['A','C','E','D','G','F','I','H','K','M','L','N','Q','P','S','R','T','W','V','Y','X','NoSeq']
ss_labels_q3 = ['H', 'E', 'C']

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--data-dir", type=Path,
                    default=Path(os.environ.get("PROTEIN_SS_DATA", "../protein-ss-prediction/data/processed")))
parser.add_argument("--out", type=Path, default=Path("data/sample_proteins.json"))
args = parser.parse_args()

X_test = np.load(args.data_dir / 'X_test.npy')
Y_test = np.load(args.data_dir / 'Y_test.npy')
PSSM_test = np.load(args.data_dir / 'PSSM_test.npy')

samples = []
# Pick 10 proteins with varied lengths, skipping the very first few if too short
picked_indices = [0, 5, 12, 20, 33, 47, 58, 71, 85, 99]

for idx in picked_indices:
    raw_indices = X_test[idx]
    real_len = int((raw_indices != 21).sum())
    if real_len < 30 or real_len > 300:
        continue  # skip extreme lengths for a cleaner demo

    seq = ''.join(residues[i] for i in raw_indices[:real_len])
    true_ss = ''.join(ss_labels_q3[i] for i in Y_test[idx][:real_len])
    pssm = PSSM_test[idx][:real_len].tolist()  # real_len x 22

    samples.append({
        "id": f"SAMPLE_{idx:03d}",
        "sequence": seq,
        "true_structure": true_ss,
        "length": real_len,
        "pssm": pssm
    })

print(f"Extracted {len(samples)} samples")
for s in samples:
    print(f"  {s['id']}: length={s['length']}")

with open(args.out, 'w') as f:
    json.dump(samples, f)

print(f"\nSaved to {args.out}")

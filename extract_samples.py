import numpy as np
import json

residues = ['A','C','E','D','G','F','I','H','K','M','L','N','Q','P','S','R','T','W','V','Y','X','NoSeq']
ss_labels_q3 = ['H', 'E', 'C']

X_test = np.load('/Users/palellabhanuprasad/protein-ss-prediction/data/processed/X_test.npy')
Y_test = np.load('/Users/palellabhanuprasad/protein-ss-prediction/data/processed/Y_test.npy')
PSSM_test = np.load('/Users/palellabhanuprasad/protein-ss-prediction/data/processed/PSSM_test.npy')

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

with open('data/sample_proteins.json', 'w') as f:
    json.dump(samples, f)

print("\nSaved to data/sample_proteins.json")

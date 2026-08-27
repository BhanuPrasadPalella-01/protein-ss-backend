import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, X_path, Y_path, mask_path, pssm_path=None):
        self.X = torch.from_numpy(np.load(X_path)).long()
        self.Y = torch.from_numpy(np.load(Y_path)).long()
        self.mask = torch.from_numpy(np.load(mask_path)).bool()
        self.use_pssm = pssm_path is not None
        if self.use_pssm:
            self.PSSM = torch.from_numpy(np.load(pssm_path)).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.use_pssm:
            return self.X[idx], self.PSSM[idx], self.Y[idx], self.mask[idx]
        return self.X[idx], self.Y[idx], self.mask[idx]


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size=22, embed_dim=32, hidden_dim=64, num_classes=3, padding_idx=21):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.classifier(lstm_out)
        return logits


class BiLSTMTaggerPSSM(nn.Module):
    def __init__(self, vocab_size=22, embed_dim=32, pssm_dim=22, hidden_dim=64, num_classes=3, padding_idx=21):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        combined_input_dim = embed_dim + pssm_dim
        self.lstm = nn.LSTM(input_size=combined_input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, pssm):
        embedded = self.embedding(x)
        combined = torch.cat([embedded, pssm], dim=-1)
        lstm_out, _ = self.lstm(combined)
        logits = self.classifier(lstm_out)
        return logits


class BiLSTMAttentionPSSM(nn.Module):
    """Sequence + PSSM -> BiLSTM -> Self-Attention -> Classifier.
       Self-attention lets every position directly attend to every other
       position, helping capture long-range dependencies (esp. beta strands)
       that a plain BiLSTM captures only weakly.
    """
    def __init__(self, vocab_size=22, embed_dim=32, pssm_dim=22, hidden_dim=64,
                 num_heads=4, num_classes=3, padding_idx=21):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        combined_input_dim = embed_dim + pssm_dim
        self.lstm = nn.LSTM(input_size=combined_input_dim, hidden_size=hidden_dim,
                             batch_first=True, bidirectional=True)
        attn_dim = hidden_dim * 2
        self.attention = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(attn_dim)
        self.classifier = nn.Linear(attn_dim, num_classes)

    def forward(self, x, pssm, key_padding_mask=None):
        # key_padding_mask: (batch, seq_len), True at PADDED positions
        embedded = self.embedding(x)
        combined = torch.cat([embedded, pssm], dim=-1)
        lstm_out, _ = self.lstm(combined)                      # (batch, 700, 128)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out,
                                      key_padding_mask=key_padding_mask)
        fused = self.layer_norm(lstm_out + attn_out)            # residual + normalization
        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    dataset = ProteinDataset(
        'data/processed/X_train.npy',
        'data/processed/Y_train.npy',
        'data/processed/mask_train.npy',
        'data/processed/PSSM_train.npy'
    )
    print("Dataset size:", len(dataset))

    X_batch, PSSM_batch, Y_batch, mask_batch = dataset[0]
    X_batch = X_batch.unsqueeze(0)
    PSSM_batch = PSSM_batch.unsqueeze(0)
    mask_batch = mask_batch.unsqueeze(0)
    key_padding_mask = ~mask_batch  # True where padded

    print("\n--- Baseline model (sequence only) ---")
    baseline_model = BiLSTMTagger()
    baseline_logits = baseline_model(X_batch)
    print("Output shape:", baseline_logits.shape, "| Expected: torch.Size([1, 700, 3])")
    print(f"Params: {sum(p.numel() for p in baseline_model.parameters()):,}")

    print("\n--- PSSM model (sequence + profile) ---")
    pssm_model = BiLSTMTaggerPSSM()
    pssm_logits = pssm_model(X_batch, PSSM_batch)
    print("Output shape:", pssm_logits.shape, "| Expected: torch.Size([1, 700, 3])")
    print(f"Params: {sum(p.numel() for p in pssm_model.parameters()):,}")

    print("\n--- PSSM + Attention model (sequence + profile + self-attention) ---")
    attn_model = BiLSTMAttentionPSSM()
    attn_logits = attn_model(X_batch, PSSM_batch, key_padding_mask=key_padding_mask)
    print("Output shape:", attn_logits.shape, "| Expected: torch.Size([1, 700, 3])")
    print(f"Params: {sum(p.numel() for p in attn_model.parameters()):,}")

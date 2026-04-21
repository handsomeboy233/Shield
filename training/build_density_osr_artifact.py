from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class CharTextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        emb_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        hidden_dim: int = 128,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.transpose(1, 2)
        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(emb))
            p = torch.max(c, dim=2).values
            conv_outs.append(p)
        feat = torch.cat(conv_outs, dim=1)
        feat = self.dropout(feat)
        feat = torch.relu(self.fc1(feat))
        logits = self.classifier(feat)
        return logits, feat


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(ch, vocab.get("<unk>", 1)) for ch in str(text)]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [vocab.get("<pad>", 0)] * (max_len - len(ids))
    return ids


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def batched_extract_features(model, texts, vocab, max_len, device, batch_size=128):
    feats = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        x = torch.tensor(
            [encode_text(t, vocab, max_len) for t in batch_texts],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            _, feat = model(x)
        feats.append(feat.detach().cpu().numpy().astype(np.float32))
    feats = np.concatenate(feats, axis=0)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with text,label")
    ap.add_argument("--model_ckpt", required=True)
    ap.add_argument("--base_osr_ckpt", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--k1", type=int, default=50)
    ap.add_argument("--k2", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(args.data)
    assert "text" in df.columns and "label" in df.columns, "need text,label columns"
    texts = df["text"].astype(str).tolist()

    model_bundle = torch.load(args.model_ckpt, map_location=device, weights_only=False)
    base_osr = torch.load(args.base_osr_ckpt, map_location="cpu", weights_only=False)

    vocab = model_bundle["vocab"]
    config = model_bundle["config"]

    model = CharTextCNN(
        vocab_size=len(vocab),
        num_classes=len(model_bundle["label2id"]),
        emb_dim=int(config["emb_dim"]),
        num_filters=int(config["num_filters"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        pad_idx=vocab["<pad>"],
    ).to(device)
    model.load_state_dict(model_bundle["model_state"])
    model.eval()

    feats = batched_extract_features(
        model=model,
        texts=texts,
        vocab=vocab,
        max_len=int(config["max_len"]),
        device=device,
        batch_size=args.batch_size,
    )
    feats_norm = l2_normalize(feats)

    n = feats_norm.shape[0]
    dmat = np.linalg.norm(feats_norm[:, None, :] - feats_norm[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)

    k1 = min(args.k1, max(1, n - 1))
    knn_dists = np.sort(dmat, axis=1)[:, :k1]
    rho = knn_dists.mean(axis=1)
    delta = float(rho.mean())

    out = dict(base_osr)
    out["density_estimation"] = {
        "k1": int(k1),
        "k2": int(args.k2),
        "delta": float(delta),
        "train_features_norm": feats_norm.astype(np.float32),
    }

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, save_path)

    print("train_features_norm shape =", feats_norm.shape)
    print("k1 =", k1)
    print("k2 =", args.k2)
    print("delta =", delta)
    print("saved to:", save_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import re
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


def canonicalize_raw_text(raw: str) -> str:
    raw = str(raw or "").strip()
    if "METHOD=" in raw and "PATH=" in raw and "PROTOCOL=" in raw:
        return raw

    m = re.search(r'"([A-Z]+)\s+(\S+)\s+(HTTP/\d\.\d)"', raw)
    if not m:
        m = re.search(r'^([A-Z]+)\s+(\S+)\s+(HTTP/\d\.\d)$', raw)

    if m:
        method = m.group(1)
        target = m.group(2)
        proto = m.group(3)
        if "?" in target:
            path, query = target.split("?", 1)
        else:
            path, query = target, ""
        return f"METHOD={method} PATH={path} QUERY={query} PROTOCOL={proto}"

    return "METHOD=GET PATH=/ QUERY= PROTOCOL=HTTP/1.1"


def load_texts(df: pd.DataFrame):
    if "text" in df.columns:
        return df["text"].astype(str).tolist()
    if "raw_text" in df.columns:
        return [canonicalize_raw_text(x) for x in df["raw_text"].astype(str).tolist()]
    raise ValueError("CSV needs text or raw_text column")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_bundle = torch.load(args.model_ckpt, map_location=device, weights_only=False)

    vocab = model_bundle["vocab"]
    raw_id2label = model_bundle["id2label"]
    if isinstance(next(iter(raw_id2label.keys())), str):
        id2label = {int(k): v for k, v in raw_id2label.items()}
    else:
        id2label = raw_id2label

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

    df = pd.read_csv(args.data)
    texts = load_texts(df)

    rows = []
    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i:i+args.batch_size]
        x = torch.tensor(
            [encode_text(t, vocab, int(config["max_len"])) for t in batch_texts],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

        for j, text in enumerate(batch_texts):
            prob = probs[j]
            pred_id = int(np.argmax(prob))
            pred_label = id2label[pred_id]
            max_prob = float(prob[pred_id])
            rows.append({
                "row_idx": i + j,
                "text": text,
                "pred_label": pred_label,
                "max_prob": max_prob,
            })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print("Saved results to:", out_path)
    print(out_df.head(10))
    print("\npred_label counts:")
    print(out_df["pred_label"].value_counts())


if __name__ == "__main__":
    main()

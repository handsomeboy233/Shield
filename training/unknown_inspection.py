from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
    return np.concatenate(feats, axis=0)


def choose_k(x: np.ndarray, min_k: int, max_k: int) -> tuple[int, float]:
    n = len(x)
    max_k = min(max_k, max(2, n - 1))
    if n < 4:
        return 2, -1.0

    best_k = min_k
    best_score = -1.0
    for k in range(min_k, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = km.fit_predict(x)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(x, labels, metric="euclidean")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue
    return best_k, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--model_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--unknown_col", default="final_label")
    ap.add_argument("--unknown_value", default="unknown")
    ap.add_argument("--min_k", type=int, default=2)
    ap.add_argument("--max_k", type=int, default=12)
    ap.add_argument("--repr_per_cluster", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(args.input_csv)
    df_unknown = df[df[args.unknown_col].astype(str) == args.unknown_value].copy().reset_index(drop=True)

    if len(df_unknown) == 0:
        raise RuntimeError("No unknown samples found.")

    if "raw_text" in df_unknown.columns:
        df_unknown["text"] = df_unknown["raw_text"].astype(str).apply(canonicalize_raw_text)
    elif "text" in df_unknown.columns:
        df_unknown["text"] = df_unknown["text"].astype(str)
    else:
        raise RuntimeError("Need raw_text or text column.")

    model_bundle = torch.load(args.model_ckpt, map_location=device, weights_only=False)
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
        texts=df_unknown["text"].tolist(),
        vocab=vocab,
        max_len=int(config["max_len"]),
        device=device,
        batch_size=args.batch_size,
    )

    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    feats_norm = feats / norms

    k, sil = choose_k(feats_norm, args.min_k, args.max_k)
    print(f"chosen_k = {k}, silhouette = {sil:.4f}")

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    cluster_ids = km.fit_predict(feats_norm)
    centers = km.cluster_centers_

    dists = np.linalg.norm(feats_norm - centers[cluster_ids], axis=1)
    df_unknown["cluster_id"] = cluster_ids
    df_unknown["cluster_dist"] = dists
    df_unknown["pseudo_label"] = df_unknown["cluster_id"].apply(lambda x: f"unknown_cluster_{int(x):02d}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clustered_path = out_dir / "unknown_clustered.csv"
    summary_path = out_dir / "unknown_cluster_summary.csv"
    repr_path = out_dir / "unknown_representatives.csv"

    df_unknown.to_csv(clustered_path, index=False, encoding="utf-8")

    summary_rows = []
    repr_rows = []

    pred_col = "osr_pred_label" if "osr_pred_label" in df_unknown.columns else None
    reject_col = "osr_reject_reason" if "osr_reject_reason" in df_unknown.columns else ("reject_reason" if "reject_reason" in df_unknown.columns else None)
    record_col = "record_id" if "record_id" in df_unknown.columns else None

    for cid in sorted(df_unknown["cluster_id"].unique()):
        sub = df_unknown[df_unknown["cluster_id"] == cid].copy().sort_values("cluster_dist")
        size = len(sub)

        top_pred = ""
        top_reject = ""
        if pred_col:
            top_pred = Counter(sub[pred_col].astype(str)).most_common(1)[0][0]
        if reject_col:
            top_reject = Counter(sub[reject_col].astype(str)).most_common(1)[0][0]

        sample_ids = ",".join(sub[record_col].astype(str).head(min(5, size)).tolist()) if record_col else ""
        sample_paths = " | ".join(sub["text"].astype(str).head(min(3, size)).tolist())

        summary_rows.append({
            "cluster_id": int(cid),
            "pseudo_label": f"unknown_cluster_{int(cid):02d}",
            "size": int(size),
            "top_osr_pred_label": top_pred,
            "top_reject_reason": top_reject,
            "sample_record_ids": sample_ids,
            "sample_texts": sample_paths,
        })

        reps = sub.head(args.repr_per_cluster).copy()
        reps["representative_rank"] = range(1, len(reps) + 1)
        repr_rows.append(reps)

    summary_df = pd.DataFrame(summary_rows).sort_values(["size", "cluster_id"], ascending=[False, True])
    repr_df = pd.concat(repr_rows, ignore_index=True)

    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    repr_df.to_csv(repr_path, index=False, encoding="utf-8")

    print("saved:", clustered_path)
    print("saved:", summary_path)
    print("saved:", repr_path)
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

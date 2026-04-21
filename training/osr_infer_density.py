from __future__ import annotations

import argparse
import json
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


def l2_normalize_row(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)


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
    ap.add_argument("--osr_ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_bundle = torch.load(args.model_ckpt, map_location=device, weights_only=False)
    osr_bundle = torch.load(args.osr_ckpt, map_location="cpu", weights_only=False)

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

    if "centroids" in osr_bundle:
        centroids = {int(k): np.asarray(v, dtype=np.float32) for k, v in osr_bundle["centroids"].items()} if isinstance(osr_bundle["centroids"], dict) else {i: np.asarray(v, dtype=np.float32) for i, v in enumerate(osr_bundle["centroids"])}
    else:
        centroids = {int(k): np.asarray(v, dtype=np.float32) for k, v in osr_bundle["class_centroids"].items()} if isinstance(osr_bundle["class_centroids"], dict) else {i: np.asarray(v, dtype=np.float32) for i, v in enumerate(osr_bundle["class_centroids"])}

    thresholds = osr_bundle["thresholds"] if "thresholds" in osr_bundle else osr_bundle
    class_distance_thresholds = thresholds["class_distance_thresholds"]
    if isinstance(class_distance_thresholds, dict):
        class_distance_thresholds = {int(k): float(v) for k, v in class_distance_thresholds.items()}

    de = osr_bundle["density_estimation"]
    train_features_norm = np.asarray(de["train_features_norm"], dtype=np.float32)
    k1 = int(de["k1"])
    k2 = int(de["k2"])
    delta = float(de["delta"])

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
            logits, feats = model(x)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            feats = feats.detach().cpu().numpy().astype(np.float32)

        for j, text in enumerate(batch_texts):
            prob = probs[j]
            feat = feats[j]

            pred_id = int(np.argmax(prob))
            pred_label = id2label[pred_id]
            max_prob = float(prob[pred_id])

            dist_to_pred_centroid = float(np.linalg.norm(feat - centroids[pred_id]))
            base_prob_th = float(thresholds["global_prob_threshold"])
            base_dist_th = float(class_distance_thresholds[pred_id])

            prob_th = base_prob_th
            dist_th = base_dist_th
            if pred_label == "command_exec":
                prob_th = max(base_prob_th, 0.75)
                dist_th = min(base_dist_th, base_dist_th * 0.85)

            feat_norm = l2_normalize_row(feat.astype(np.float32))
            dists = np.linalg.norm(train_features_norm - feat_norm[None, :], axis=1)
            k = min(k1, len(dists))
            nn = np.sort(dists)[:k]
            reliable_count = int(np.sum(nn <= delta))
            density_unknown = reliable_count < k2

            low_conf = max_prob < prob_th
            far_from_centroid = dist_to_pred_centroid > dist_th

            if density_unknown:
                final_label = "unknown"
                reject_reason = "low_density"
                is_unknown = 1
            else:
                if pred_label == "benign":
                    if low_conf and far_from_centroid:
                        final_label = "unknown"
                        reject_reason = "low_confidence+far_from_centroid"
                        is_unknown = 1
                    else:
                        final_label = "benign"
                        reject_reason = "accepted"
                        is_unknown = 0
                else:
                    if low_conf and far_from_centroid:
                        final_label = "unknown"
                        reject_reason = "low_confidence+far_from_centroid"
                        is_unknown = 1
                    elif low_conf:
                        final_label = "unknown"
                        reject_reason = "low_confidence"
                        is_unknown = 1
                    elif far_from_centroid:
                        final_label = "unknown"
                        reject_reason = "far_from_centroid"
                        is_unknown = 1
                    else:
                        final_label = pred_label
                        reject_reason = "accepted"
                        is_unknown = 0

            rows.append({
                "row_idx": i + j,
                "text": text,
                "pred_label": pred_label,
                "max_prob": max_prob,
                "dist_to_pred_centroid": dist_to_pred_centroid,
                "prob_threshold": prob_th,
                "distance_threshold": dist_th,
                "reliable_count": reliable_count,
                "delta": delta,
                "k1": k1,
                "k2": k2,
                "is_unknown": is_unknown,
                "final_label": final_label,
                "reject_reason": reject_reason,
            })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print("Saved results to:", out_path)
    print(out_df.head(10))
    print("\nlabel counts:")
    print(out_df["final_label"].value_counts())


if __name__ == "__main__":
    main()

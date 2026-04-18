import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class CharTextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        emb_dim=64,
        num_filters=64,
        kernel_sizes=(3, 4, 5),
        hidden_dim=128,
        dropout=0.3,
        pad_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)              # [B, L, E]
        emb = emb.transpose(1, 2)           # [B, E, L]

        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(emb))       # [B, F, L-k+1]
            p = torch.max(c, dim=2).values  # [B, F]
            conv_outs.append(p)

        feat = torch.cat(conv_outs, dim=1)  # [B, F*K]
        feat = self.dropout(feat)
        feat = torch.relu(self.fc1(feat))   # embedding for OSR
        logits = self.classifier(feat)
        return logits, feat


def encode_text(text: str, vocab: dict, max_len: int):
    ids = [vocab.get(ch, vocab["<unk>"]) for ch in str(text)]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


class TextOnlyDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(
            encode_text(self.texts[idx], self.vocab, self.max_len),
            dtype=torch.long,
        )
        if self.labels is None:
            return x
        return x, self.labels[idx]


def load_model(model_ckpt: str, device: torch.device):
    ckpt = torch.load(model_ckpt, map_location=device)
    config = ckpt["config"]
    vocab = ckpt["vocab"]
    label2id = ckpt["label2id"]
    id2label = {int(k): v for k, v in ckpt["id2label"].items()} if isinstance(next(iter(ckpt["id2label"].keys())), str) else ckpt["id2label"]

    model = CharTextCNN(
        vocab_size=len(vocab),
        num_classes=len(label2id),
        emb_dim=config["emb_dim"],
        num_filters=config["num_filters"],
        kernel_sizes=(3, 4, 5),
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        pad_idx=vocab["<pad>"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, label2id, id2label, config


def run_inference_df(model, df, vocab, label2id, id2label, max_len, device, batch_size=64):
    ds = TextOnlyDataset(
        texts=df["text"].astype(str).tolist(),
        labels=[label2id[x] for x in df["label"].tolist()],
        vocab=vocab,
        max_len=max_len,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    rows = []
    offset = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, feat = model(x)
            probs = torch.softmax(logits, dim=1)
            max_probs, pred_ids = torch.max(probs, dim=1)

            feat_np = feat.cpu().numpy()
            max_prob_np = max_probs.cpu().numpy()
            pred_id_np = pred_ids.cpu().numpy()
            y_np = y.numpy()

            for i in range(len(y_np)):
                row = {
                    "row_idx": int(offset + i),
                    "true_id": int(y_np[i]),
                    "true_label": id2label[int(y_np[i])],
                    "pred_id": int(pred_id_np[i]),
                    "pred_label": id2label[int(pred_id_np[i])],
                    "max_prob": float(max_prob_np[i]),
                    "embedding": feat_np[i].tolist(),
                    "is_correct": int(pred_id_np[i] == y_np[i]),
                }
                rows.append(row)
            offset += len(y_np)
    return pd.DataFrame(rows)


def run_inference_texts(model, texts, vocab, max_len, device, batch_size=64):
    ds = TextOnlyDataset(texts=texts, labels=None, vocab=vocab, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    rows = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            logits, feat = model(x)
            probs = torch.softmax(logits, dim=1)
            max_probs, pred_ids = torch.max(probs, dim=1)
            rows.extend(
                zip(
                    pred_ids.cpu().numpy().tolist(),
                    max_probs.cpu().numpy().tolist(),
                    feat.cpu().numpy().tolist(),
                )
            )
    return rows


def compute_centroids(train_embed_df: pd.DataFrame, label2id: dict):
    centroids = {}
    for label, class_id in label2id.items():
        sub = train_embed_df[train_embed_df["true_id"] == class_id]
        arr = np.array(sub["embedding"].tolist(), dtype=np.float32)
        centroids[class_id] = arr.mean(axis=0)
    return centroids


def add_distance_columns(embed_df: pd.DataFrame, centroids: dict):
    pred_centroid_dist = []
    nearest_centroid_dist = []
    nearest_centroid_id = []

    centroid_items = sorted(centroids.items(), key=lambda x: x[0])
    centroid_ids = [cid for cid, _ in centroid_items]
    centroid_matrix = np.stack([vec for _, vec in centroid_items], axis=0)

    for emb, pred_id in zip(embed_df["embedding"], embed_df["pred_id"]):
        emb = np.array(emb, dtype=np.float32)
        dists = np.linalg.norm(centroid_matrix - emb[None, :], axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_centroid_id.append(int(centroid_ids[nearest_idx]))
        nearest_centroid_dist.append(float(dists[nearest_idx]))
        pred_centroid_dist.append(float(np.linalg.norm(emb - centroids[int(pred_id)])))

    embed_df = embed_df.copy()
    embed_df["dist_to_pred_centroid"] = pred_centroid_dist
    embed_df["nearest_centroid_id"] = nearest_centroid_id
    embed_df["nearest_centroid_dist"] = nearest_centroid_dist
    return embed_df


def build_thresholds(val_embed_df: pd.DataFrame, label2id: dict):
    correct_val = val_embed_df[val_embed_df["is_correct"] == 1].copy()
    if len(correct_val) == 0:
        raise ValueError("验证集上没有正确分类样本，无法构建 OSR 阈值。")

    global_prob_threshold = float(np.percentile(correct_val["max_prob"].values, 5))
    global_dist_threshold = float(np.percentile(correct_val["dist_to_pred_centroid"].values, 95))

    class_distance_thresholds = {}
    for label, class_id in label2id.items():
        sub = correct_val[correct_val["pred_id"] == class_id]
        if len(sub) >= 2:
            th = float(np.percentile(sub["dist_to_pred_centroid"].values, 95))
        else:
            th = global_dist_threshold
        class_distance_thresholds[class_id] = th

    return {
        "global_prob_threshold": global_prob_threshold,
        "global_dist_threshold": global_dist_threshold,
        "class_distance_thresholds": class_distance_thresholds,
    }


def apply_osr_rule(embed_df: pd.DataFrame, thresholds: dict, id2label: dict):
    out = embed_df.copy()

    class_distance_thresholds = thresholds["class_distance_thresholds"]
    prob_threshold = thresholds["global_prob_threshold"]

    reasons = []
    final_labels = []
    is_unknowns = []
    class_threshold_used = []

    for _, row in out.iterrows():
        pred_id = int(row["pred_id"])
        prob = float(row["max_prob"])
        dist = float(row["dist_to_pred_centroid"])
        dist_th = float(class_distance_thresholds[pred_id])

        low_conf = prob < prob_threshold
        far_from_centroid = dist > dist_th

        if low_conf and far_from_centroid:
            reason = "low_confidence+far_from_centroid"
            final_label = "unknown"
            is_unknown = 1
        elif low_conf:
            reason = "low_confidence"
            final_label = "unknown"
            is_unknown = 1
        elif far_from_centroid:
            reason = "far_from_centroid"
            final_label = "unknown"
            is_unknown = 1
        else:
            reason = "accepted"
            final_label = id2label[pred_id]
            is_unknown = 0

        reasons.append(reason)
        final_labels.append(final_label)
        is_unknowns.append(is_unknown)
        class_threshold_used.append(dist_th)

    out["class_distance_threshold"] = class_threshold_used
    out["is_unknown"] = is_unknowns
    out["final_label"] = final_labels
    out["reject_reason"] = reasons
    return out


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, vocab, label2id, id2label, config = load_model(args.model_ckpt, device)

    df = pd.read_csv(args.data)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("输入 CSV 必须包含 text,label 两列")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=args.seed, stratify=train_df["label"]
    )

    train_embed_df = run_inference_df(model, train_df, vocab, label2id, id2label, config["max_len"], device, args.batch_size)
    val_embed_df = run_inference_df(model, val_df, vocab, label2id, id2label, config["max_len"], device, args.batch_size)
    test_embed_df = run_inference_df(model, test_df, vocab, label2id, id2label, config["max_len"], device, args.batch_size)

    centroids = compute_centroids(train_embed_df, label2id)
    train_embed_df = add_distance_columns(train_embed_df, centroids)
    val_embed_df = add_distance_columns(val_embed_df, centroids)
    test_embed_df = add_distance_columns(test_embed_df, centroids)

    thresholds = build_thresholds(val_embed_df, label2id)
    test_osr_df = apply_osr_rule(test_embed_df, thresholds, id2label)

    known_accept_rate = 1.0 - float(test_osr_df["is_unknown"].mean())
    known_accuracy_after_reject = float((test_osr_df["final_label"] == test_osr_df["true_label"]).mean())

    print("\n=== OSR thresholds ===")
    print(json.dumps(thresholds, indent=2, ensure_ascii=False))
    print("\n=== Known-set OSR summary ===")
    print(f"known_accept_rate: {known_accept_rate:.4f}")
    print(f"known_accuracy_after_reject: {known_accuracy_after_reject:.4f}")
    print(test_osr_df[["true_label", "pred_label", "final_label", "max_prob", "dist_to_pred_centroid", "reject_reason"]].head(10))

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "centroids": {int(k): torch.tensor(v, dtype=torch.float32) for k, v in centroids.items()},
        "thresholds": {
            "global_prob_threshold": float(thresholds["global_prob_threshold"]),
            "global_dist_threshold": float(thresholds["global_dist_threshold"]),
            "class_distance_thresholds": {int(k): float(v) for k, v in thresholds["class_distance_thresholds"].items()},
        },
        "label2id": label2id,
        "id2label": id2label,
        "model_ckpt": args.model_ckpt,
        "seed": args.seed,
        "max_len": config["max_len"],
        "known_test_summary": {
            "known_accept_rate": known_accept_rate,
            "known_accuracy_after_reject": known_accuracy_after_reject,
        },
    }
    torch.save(artifact, save_path)
    print(f"\nSaved OSR artifact to: {save_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_embed_df.to_csv(out_dir / "train_embeddings_scored.csv", index=False)
    val_embed_df.to_csv(out_dir / "val_embeddings_scored.csv", index=False)
    test_osr_df.to_csv(out_dir / "test_osr_scored.csv", index=False)
    print(f"Saved scored CSVs to: {out_dir}")

    if args.unknown_data:
        udf = pd.read_csv(args.unknown_data)
        if "text" not in udf.columns:
            raise ValueError("unknown_data CSV 必须包含 text 列")
        utexts = udf["text"].astype(str).tolist()
        infer_rows = run_inference_texts(model, utexts, vocab, config["max_len"], device, args.batch_size)

        unknown_out = []
        for text, (pred_id, max_prob, emb) in zip(utexts, infer_rows):
            emb = np.array(emb, dtype=np.float32)
            pred_id = int(pred_id)
            pred_label = id2label[pred_id]
            dist_to_pred_centroid = float(np.linalg.norm(emb - centroids[pred_id]))
            dist_th = float(thresholds["class_distance_thresholds"][pred_id])
            low_conf = max_prob < thresholds["global_prob_threshold"]
            far_from_centroid = dist_to_pred_centroid > dist_th
            is_unknown = int(low_conf or far_from_centroid)

            if low_conf and far_from_centroid:
                reject_reason = "low_confidence+far_from_centroid"
                final_label = "unknown"
            elif low_conf:
                reject_reason = "low_confidence"
                final_label = "unknown"
            elif far_from_centroid:
                reject_reason = "far_from_centroid"
                final_label = "unknown"
            else:
                reject_reason = "accepted"
                final_label = pred_label

            unknown_out.append({
                "text": text,
                "pred_label": pred_label,
                "max_prob": float(max_prob),
                "dist_to_pred_centroid": dist_to_pred_centroid,
                "class_distance_threshold": dist_th,
                "is_unknown": is_unknown,
                "final_label": final_label,
                "reject_reason": reject_reason,
            })

        unknown_out_df = pd.DataFrame(unknown_out)
        unknown_reject_rate = float(unknown_out_df["is_unknown"].mean()) if len(unknown_out_df) else 0.0
        unknown_out_df.to_csv(out_dir / "unknown_pool_osr_scored.csv", index=False)
        print("\n=== Unknown-pool summary ===")
        print(f"unknown_reject_rate: {unknown_reject_rate:.4f}")
        print(f"Saved unknown-pool scores to: {out_dir / 'unknown_pool_osr_scored.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Known labeled CSV with text,label")
    parser.add_argument("--model_ckpt", type=str, required=True, help="checkpoints/textcnn_closedset.pt")
    parser.add_argument("--save_path", type=str, default="checkpoints/osr_artifacts.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/osr")
    parser.add_argument("--unknown_data", type=str, default=None, help="Optional unknown-pool CSV with text column")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

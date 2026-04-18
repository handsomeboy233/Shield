import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


def encode_text(text: str, vocab: dict, max_len: int):
    ids = [vocab.get(ch, vocab["<unk>"]) for ch in str(text)]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


class TextOnlyDataset(Dataset):
    def __init__(self, texts, vocab, max_len):
        self.texts = list(texts)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(
            encode_text(self.texts[idx], self.vocab, self.max_len),
            dtype=torch.long,
        )


def load_everything(model_ckpt: str, osr_ckpt: str, device: torch.device):
    model_raw = torch.load(model_ckpt, map_location=device)
    config = model_raw["config"]
    vocab = model_raw["vocab"]
    label2id = model_raw["label2id"]
    id2label = {int(k): v for k, v in model_raw["id2label"].items()} if isinstance(next(iter(model_raw["id2label"].keys())), str) else model_raw["id2label"]

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
    model.load_state_dict(model_raw["model_state"])
    model.eval()

    osr = torch.load(osr_ckpt, map_location=device)
    centroids = {int(k): v.cpu().numpy() for k, v in osr["centroids"].items()}
    thresholds = osr["thresholds"]
    thresholds["class_distance_thresholds"] = {
        int(k): float(v) for k, v in thresholds["class_distance_thresholds"].items()
    }
    return model, vocab, id2label, config["max_len"], centroids, thresholds


def infer_texts(model, texts, vocab, id2label, max_len, centroids, thresholds, device, batch_size=64):
    ds = TextOnlyDataset(texts, vocab, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    rows = []
    idx = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            logits, feat = model(x)
            probs = torch.softmax(logits, dim=1)
            max_probs, pred_ids = torch.max(probs, dim=1)

            feat_np = feat.cpu().numpy()
            pred_ids_np = pred_ids.cpu().numpy()
            max_probs_np = max_probs.cpu().numpy()

            for i in range(len(pred_ids_np)):
                pred_id = int(pred_ids_np[i])
                pred_label = id2label[pred_id]
                max_prob = float(max_probs_np[i])
                emb = np.array(feat_np[i], dtype=np.float32)
                dist_to_pred_centroid = float(np.linalg.norm(emb - centroids[pred_id]))
                dist_th = float(thresholds["class_distance_thresholds"][pred_id])
                prob_th = float(thresholds["global_prob_threshold"])

                low_conf = max_prob < prob_th
                far_from_centroid = dist_to_pred_centroid > dist_th

                if low_conf and far_from_centroid:
                    reject_reason = "low_confidence+far_from_centroid"
                    final_label = "unknown"
                    is_unknown = 1
                elif low_conf:
                    reject_reason = "low_confidence"
                    final_label = "unknown"
                    is_unknown = 1
                elif far_from_centroid:
                    reject_reason = "far_from_centroid"
                    final_label = "unknown"
                    is_unknown = 1
                else:
                    reject_reason = "accepted"
                    final_label = pred_label
                    is_unknown = 0

                rows.append({
                    "row_idx": idx,
                    "text": texts[idx],
                    "pred_label": pred_label,
                    "max_prob": max_prob,
                    "prob_threshold": prob_th,
                    "dist_to_pred_centroid": dist_to_pred_centroid,
                    "class_distance_threshold": dist_th,
                    "is_unknown": is_unknown,
                    "final_label": final_label,
                    "reject_reason": reject_reason,
                })
                idx += 1
    return pd.DataFrame(rows)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, vocab, id2label, max_len, centroids, thresholds = load_everything(
        args.model_ckpt, args.osr_ckpt, device
    )

    if args.text:
        texts = [args.text]
    elif args.data:
        df = pd.read_csv(args.data)
        if "text" not in df.columns:
            raise ValueError("输入 CSV 必须包含 text 列")
        texts = df["text"].astype(str).tolist()
    else:
        raise ValueError("必须提供 --text 或 --data")

    out_df = infer_texts(
        model=model,
        texts=texts,
        vocab=vocab,
        id2label=id2label,
        max_len=max_len,
        centroids=centroids,
        thresholds=thresholds,
        device=device,
        batch_size=args.batch_size,
    )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Saved results to: {out_path}")

    if len(out_df) == 1:
        print(json.dumps(out_df.iloc[0].to_dict(), indent=2, ensure_ascii=False))
    else:
        print(out_df.head(10))
        print("\nlabel counts:")
        print(out_df["final_label"].value_counts(dropna=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--osr_ckpt", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)

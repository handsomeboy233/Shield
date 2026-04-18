import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vocab(texts, min_freq=1, max_size=None):
    counter = Counter()
    for text in texts:
        counter.update(list(str(text)))

    vocab = {"<pad>": 0, "<unk>": 1}
    items = [(ch, cnt) for ch, cnt in counter.items() if cnt >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))

    if max_size is not None:
        items = items[:max_size]

    for ch, _ in items:
        vocab[ch] = len(vocab)
    return vocab


def encode_text(text, vocab, max_len):
    ids = [vocab.get(ch, vocab["<unk>"]) for ch in str(text)]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


class LogDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2id, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode_text(self.texts[idx], self.vocab, self.max_len)
        y = self.label2id[self.labels[idx]]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


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
        # x: [B, L]
        emb = self.embedding(x)              # [B, L, E]
        emb = emb.transpose(1, 2)           # [B, E, L]

        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(emb))       # [B, F, L-k+1]
            p = torch.max(c, dim=2).values  # [B, F]
            conv_outs.append(p)

        feat = torch.cat(conv_outs, dim=1)  # [B, F * K]
        feat = self.dropout(feat)
        feat = torch.relu(self.fc1(feat))   # 这一层后面会拿来做开放集
        logits = self.classifier(feat)
        return logits, feat


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, all_labels, all_preds


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(args.data)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV 必须包含 text 和 label 两列")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    print("Total samples:", len(df))
    print("Label distribution:")
    print(df["label"].value_counts())

    label_counts = df["label"].value_counts()
    rare_labels = label_counts[label_counts < 5].index.tolist()

    if rare_labels:
        print("Rare labels found, keep them only in training set:", rare_labels)
        rare_df = df[df["label"].isin(rare_labels)].copy()
        main_df = df[~df["label"].isin(rare_labels)].copy()

        train_df, test_df = train_test_split(
            main_df, test_size=0.2, random_state=args.seed, stratify=main_df["label"]
        )
        train_df, val_df = train_test_split(
            train_df, test_size=0.2, random_state=args.seed, stratify=train_df["label"]
        )
     
        train_df = pd.concat([train_df, rare_df], ignore_index=True)
        train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    else:
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=args.seed, stratify=df["label"]
        )
        train_df, val_df = train_test_split(
            train_df, test_size=0.2, random_state=args.seed, stratify=train_df["label"]
        )

    vocab = build_vocab(train_df["text"].tolist(), min_freq=1, max_size=args.max_vocab)
    labels = sorted(train_df["label"].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    print("Vocab size:", len(vocab))
    print("Labels:", label2id)

    train_ds = LogDataset(train_df["text"], train_df["label"], vocab, label2id, args.max_len)
    val_ds = LogDataset(val_df["text"], val_df["label"], vocab, label2id, args.max_len)
    test_ds = LogDataset(test_df["text"], test_df["label"], vocab, label2id, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = CharTextCNN(
        vocab_size=len(vocab),
        num_classes=len(label2id),
        emb_dim=args.emb_dim,
        num_filters=args.num_filters,
        kernel_sizes=(3, 4, 5),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pad_idx=vocab["<pad>"],
    ).to(device)

    # 类别不平衡时加权
    label_counts = train_df["label"].value_counts().to_dict()
    weights = []
    for label in labels:
        weights.append(1.0 / label_counts[label])
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "textcnn_closedset.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "label2id": label2id,
                    "id2label": id2label,
                    "config": {
                        "max_len": args.max_len,
                        "emb_dim": args.emb_dim,
                        "num_filters": args.num_filters,
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                    },
                },
                ckpt_path,
            )
            print(f"Saved best model to {ckpt_path}")

    print("\nLoad best model and evaluate on test set...")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = CharTextCNN(
        vocab_size=len(checkpoint["vocab"]),
        num_classes=len(checkpoint["label2id"]),
        emb_dim=checkpoint["config"]["emb_dim"],
        num_filters=checkpoint["config"]["num_filters"],
        kernel_sizes=(3, 4, 5),
        hidden_dim=checkpoint["config"]["hidden_dim"],
        dropout=checkpoint["config"]["dropout"],
        pad_idx=checkpoint["vocab"]["<pad>"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    y_true_names = [checkpoint["id2label"][i] for i in y_true]
    y_pred_names = [checkpoint["id2label"][i] for i in y_pred]

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test macro F1: {test_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true_names, y_pred_names, digits=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="CSV file with columns: text,label")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--max_vocab", type=int, default=200)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--num_filters", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

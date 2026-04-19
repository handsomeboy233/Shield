import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_cmd(cmd, workdir=None):
    print("\n[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=workdir, check=True)
    return result.returncode


def load_text_label(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{csv_path} 必须包含 text,label 两列")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df


def make_train_csv(base_train_csv: Path, hardneg_csv: Path, n: int, seed: int, out_csv: Path):
    base_df = load_text_label(base_train_csv)
    hard_df = load_text_label(hardneg_csv)

    hard_df = hard_df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if n > 0:
        sample_n = min(n, len(hard_df))
        sampled = hard_df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
        merged = pd.concat([base_df, sampled], ignore_index=True)
    else:
        sampled = hard_df.iloc[:0].copy()
        merged = base_df.copy()

    merged = merged.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    stats = {
        "base_size": len(base_df),
        "hardneg_pool_size": len(hard_df),
        "hardneg_added": len(sampled),
        "merged_size": len(merged),
        "label_counts": merged["label"].value_counts().to_dict(),
    }
    return stats


def compute_internal_metrics(test_scored_csv: Path):
    if not test_scored_csv.exists():
        return {
            "internal_known_accept_rate": None,
            "internal_known_accuracy_after_reject": None,
        }

    df = pd.read_csv(test_scored_csv)
    if "final_label" not in df.columns or "true_label" not in df.columns:
        return {
            "internal_known_accept_rate": None,
            "internal_known_accuracy_after_reject": None,
        }

    accept_mask = df["final_label"] != "unknown"
    accept_rate = float(accept_mask.mean())

    if accept_mask.sum() > 0:
        acc_after_reject = float(
            (df.loc[accept_mask, "final_label"] == df.loc[accept_mask, "true_label"]).mean()
        )
    else:
        acc_after_reject = 0.0

    return {
        "internal_known_accept_rate": accept_rate,
        "internal_known_accuracy_after_reject": acc_after_reject,
    }


def compute_external_metrics(scored_csv: Path, benign_label="benign"):
    df = pd.read_csv(scored_csv)
    vc = df["final_label"].value_counts().to_dict()

    metrics = {
        "total": len(df),
        "unknown_rate": float((df["final_label"] == "unknown").mean()),
        "benign_rate": float((df["final_label"] == benign_label).mean()),
        "command_exec_rate": float((df["final_label"] == "command_exec").mean())
        if "command_exec" in vc else 0.0,
        "suspicious_path_probe_rate": float((df["final_label"] == "suspicious_path_probe").mean())
        if "suspicious_path_probe" in vc else 0.0,
        "label_counts": vc,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="自动跑 hard negative 对比实验")
    parser.add_argument(
        "--base_train_csv",
        type=str,
        default="datasets/osr/processed/clean_closedset_v3_canonical_balanced_text_label.csv",
    )
    parser.add_argument(
        "--hardneg_csv",
        type=str,
        default="outputs/osr/hard_negative_benign_text_label.csv",
    )
    parser.add_argument(
        "--known_csv",
        type=str,
        default="datasets/osr/processed/csic2010_external_known_benign_canonical_text_label.csv",
    )
    parser.add_argument(
        "--unknown_csv",
        type=str,
        default="datasets/osr/processed/csic2010_external_unknown_attack_canonical_text_label.csv",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="60,70,75,80,90",
        help="逗号分隔，例如 60,70,75,80,90",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="experiments/hardneg_compare_refine",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    base_train_csv = Path(args.base_train_csv)
    hardneg_csv = Path(args.hardneg_csv)
    known_csv = Path(args.known_csv)
    unknown_csv = Path(args.unknown_csv)
    output_root = Path(args.output_root)

    for p in [base_train_csv, hardneg_csv, known_csv, unknown_csv]:
        if not p.exists():
            raise FileNotFoundError(f"找不到文件: {p}")

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    summary_rows = []

    for n in sizes:
        tag = f"hardneg{n}"
        run_dir = output_root / tag
        data_dir = run_dir / "data"
        ckpt_dir = run_dir / "checkpoints"
        out_dir = run_dir / "outputs"

        data_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_csv = data_dir / f"train_{tag}.csv"
        train_stats = make_train_csv(base_train_csv, hardneg_csv, n, args.seed, train_csv)

        run_cmd([
            sys.executable, "training/train_textcnn.py",
            "--data", str(train_csv),
            "--save_dir", str(ckpt_dir)
        ])

        model_ckpt = ckpt_dir / "textcnn_closedset.pt"
        if not model_ckpt.exists():
            raise FileNotFoundError(f"训练后未找到模型: {model_ckpt}")

        internal_osr_dir = out_dir / "osr_internal"
        run_cmd([
            sys.executable, "training/extract_embeddings.py",
            "--data", str(train_csv),
            "--model_ckpt", str(model_ckpt),
            "--save_path", str(ckpt_dir / "osr_artifacts.pt"),
            "--output_dir", str(internal_osr_dir)
        ])

        osr_ckpt = ckpt_dir / "osr_artifacts.pt"
        if not osr_ckpt.exists():
            raise FileNotFoundError(f"未找到 OSR 工件: {osr_ckpt}")

        known_scored = out_dir / "csic_known_scored.csv"
        run_cmd([
            sys.executable, "training/osr_infer.py",
            "--model_ckpt", str(model_ckpt),
            "--osr_ckpt", str(osr_ckpt),
            "--data", str(known_csv),
            "--out_csv", str(known_scored)
        ])

        unknown_scored = out_dir / "csic_unknown_scored.csv"
        run_cmd([
            sys.executable, "training/osr_infer.py",
            "--model_ckpt", str(model_ckpt),
            "--osr_ckpt", str(osr_ckpt),
            "--data", str(unknown_csv),
            "--out_csv", str(unknown_scored)
        ])

        internal_metrics = compute_internal_metrics(internal_osr_dir / "test_osr_scored.csv")
        known_metrics = compute_external_metrics(known_scored)
        unknown_metrics = compute_external_metrics(unknown_scored)

        row = {
            "tag": tag,
            "hardneg_added": n,
            "train_size": train_stats["merged_size"],
            "train_label_counts": json.dumps(train_stats["label_counts"], ensure_ascii=False),

            "internal_known_accept_rate": internal_metrics["internal_known_accept_rate"],
            "internal_known_accuracy_after_reject": internal_metrics["internal_known_accuracy_after_reject"],

            "external_known_unknown_rate": known_metrics["unknown_rate"],
            "external_known_benign_rate": known_metrics["benign_rate"],
            "external_known_command_exec_rate": known_metrics["command_exec_rate"],
            "external_known_suspicious_path_probe_rate": known_metrics["suspicious_path_probe_rate"],
            "external_known_label_counts": json.dumps(known_metrics["label_counts"], ensure_ascii=False),

            "external_unknown_unknown_rate": unknown_metrics["unknown_rate"],
            "external_unknown_benign_rate": unknown_metrics["benign_rate"],
            "external_unknown_command_exec_rate": unknown_metrics["command_exec_rate"],
            "external_unknown_suspicious_path_probe_rate": unknown_metrics["suspicious_path_probe_rate"],
            "external_unknown_label_counts": json.dumps(unknown_metrics["label_counts"], ensure_ascii=False),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== all experiments done ===")
    print(summary_df[[
        "tag",
        "hardneg_added",
        "internal_known_accept_rate",
        "internal_known_accuracy_after_reject",
        "external_known_unknown_rate",
        "external_known_benign_rate",
        "external_unknown_unknown_rate",
        "external_unknown_benign_rate",
    ]])
    print("\nsaved summary to:", summary_csv)


if __name__ == "__main__":
    main()
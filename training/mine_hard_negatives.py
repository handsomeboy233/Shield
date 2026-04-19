import argparse
from pathlib import Path

import pandas as pd


def ensure_text_label(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("输入文件必须包含 text 列")
    out = df.copy()
    if "label" not in out.columns:
        out["label"] = ""
    return out


def main():
    parser = argparse.ArgumentParser(
        description="从外部正常集评分结果中自动挖 hard negative benign，并可回流到训练集"
    )
    parser.add_argument(
        "--scored_csv",
        type=str,
        required=True,
        help="OSR 推理后的外部正常集评分结果，例如 csic2010_known_benign_canonical_scored_v2.csv",
    )
    parser.add_argument(
        "--base_train_csv",
        type=str,
        required=True,
        help="当前闭集训练集（text,label）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/osr",
        help="输出目录",
    )
    parser.add_argument(
        "--wrong_label",
        type=str,
        default="command_exec",
        help="把外部正常样本误吸到哪个类，就挖哪个类的 hard negatives",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=120,
        help="回流训练集的采样条数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--dedup_on_text",
        action="store_true",
        help="是否按 text 去重后再采样",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = pd.read_csv(args.scored_csv)
    base_train = pd.read_csv(args.base_train_csv)

    base_train = ensure_text_label(base_train)

    required_cols = {"text", "final_label"}
    missing = required_cols - set(scored.columns)
    if missing:
        raise ValueError(f"评分结果缺少必要列: {missing}")

    # 只保留“外部正常样本却被接收到错误已知类”的样本
    hardneg = scored[scored["final_label"] == args.wrong_label].copy()

    if len(hardneg) == 0:
        print(f"没有找到 final_label == {args.wrong_label} 的样本")
        return

    # 这些样本的真实语义是 benign
    hardneg["label"] = "benign"
    hardneg["hard_negative_source"] = args.wrong_label

    if args.dedup_on_text:
        before = len(hardneg)
        hardneg = hardneg.drop_duplicates(subset=["text"]).reset_index(drop=True)
        print(f"按 text 去重: {before} -> {len(hardneg)}")

    # 保存完整版
    full_path = out_dir / "hard_negative_benign_full.csv"
    hardneg.to_csv(full_path, index=False)

    # 保存最小训练版
    text_label = hardneg[["text", "label"]].copy()
    text_label_path = out_dir / "hard_negative_benign_text_label.csv"
    text_label.to_csv(text_label_path, index=False)

    # 采样一部分先回流，避免 benign 再次压制攻击类
    n = min(args.sample_size, len(text_label))
    sample_df = text_label.sample(n=n, random_state=args.seed).reset_index(drop=True)
    sample_path = out_dir / f"hard_negative_benign_sample{n}_text_label.csv"
    sample_df.to_csv(sample_path, index=False)

    # 合并回训练集
    merged = pd.concat([base_train[["text", "label"]], sample_df], ignore_index=True)

    # 可选：去重，避免完全相同文本重复太多
    merged = merged.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    merged_path = out_dir / f"{Path(args.base_train_csv).stem}_plus_hardneg{n}_text_label.csv"
    merged.to_csv(merged_path, index=False)

    print("=== hard negative mining done ===")
    print(f"wrong_label target: {args.wrong_label}")
    print(f"hard negatives found: {len(hardneg)}")
    print(f"sampled for replay: {n}")
    print(f"base train size: {len(base_train)}")
    print(f"merged train size: {len(merged)}")
    print()
    print("saved:")
    print(full_path)
    print(text_label_path)
    print(sample_path)
    print(merged_path)


if __name__ == "__main__":
    main()
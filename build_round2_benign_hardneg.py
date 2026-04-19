from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def canonicalize_apache_raw(raw: str) -> str | None:
    raw = str(raw)
    m = re.search(r'"([A-Z]+)\s+(\S+)\s+(HTTP/\d\.\d)"', raw)
    if not m:
        return None
    method, target, protocol = m.groups()
    if '?' in target:
        path, query = target.split('?', 1)
    else:
        path, query = target, ''
    return f"METHOD={method} PATH={path} QUERY={query} PROTOCOL={protocol}"


def main() -> None:
    parser = argparse.ArgumentParser(description='Build round2 benign hard-negative training CSV from annotated unknown samples.')
    parser.add_argument('--annotated_csv', required=True, help='Path to unknown_sample_50_annotated.csv')
    parser.add_argument('--base_hardneg_csv', default=None, help='Existing hard_negative_benign_text_label.csv path (optional)')
    parser.add_argument('--out_candidates_csv', required=True, help='Output CSV for reviewed benign candidates with metadata')
    parser.add_argument('--out_text_label_csv', required=True, help='Output CSV for new benign text,label pairs only')
    parser.add_argument('--out_merged_csv', default=None, help='Merged text,label CSV with existing base hardneg file (optional)')
    args = parser.parse_args()

    df = pd.read_csv(args.annotated_csv)
    benign = df[df['建议标注'].astype(str).str.strip() == '实际benign'].copy()
    benign['text'] = benign['raw_text'].apply(canonicalize_apache_raw)
    benign = benign[benign['text'].notna()].copy()

    out_candidates = benign.copy()
    Path(args.out_candidates_csv).parent.mkdir(parents=True, exist_ok=True)
    out_candidates.to_csv(args.out_candidates_csv, index=False, encoding='utf-8-sig')

    text_label = benign[['text']].copy()
    text_label['label'] = 'benign'
    text_label = text_label.drop_duplicates(subset=['text', 'label']).reset_index(drop=True)
    Path(args.out_text_label_csv).parent.mkdir(parents=True, exist_ok=True)
    text_label.to_csv(args.out_text_label_csv, index=False, encoding='utf-8-sig')

    print(f'[OK] reviewed benign candidates: {len(benign)}')
    print(f'[OK] new benign text,label rows: {len(text_label)}')
    print(f'[OK] wrote: {args.out_candidates_csv}')
    print(f'[OK] wrote: {args.out_text_label_csv}')

    if args.base_hardneg_csv and args.out_merged_csv:
        base = pd.read_csv(args.base_hardneg_csv)
        if not {'text', 'label'}.issubset(base.columns):
            raise ValueError('base_hardneg_csv must contain columns: text,label')
        merged = pd.concat([base[['text', 'label']], text_label], ignore_index=True)
        merged = merged.drop_duplicates(subset=['text', 'label']).reset_index(drop=True)
        Path(args.out_merged_csv).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.out_merged_csv, index=False, encoding='utf-8-sig')
        print(f'[OK] merged rows after dedup: {len(merged)}')
        print(f'[OK] wrote: {args.out_merged_csv}')


if __name__ == '__main__':
    main()

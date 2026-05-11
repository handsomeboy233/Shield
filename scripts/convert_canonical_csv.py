#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from pathlib import Path


def pick(pattern, text, default=""):
    m = re.search(pattern, text)
    return m.group(1) if m else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise SystemExit(f"[ERROR] 输入文件不存在: {src}")

    lines = []
    with src.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or args.text_column not in reader.fieldnames:
            raise SystemExit(f"[ERROR] CSV 中没有列 {args.text_column}，实际列为: {reader.fieldnames}")

        for i, row in enumerate(reader):
            if args.limit > 0 and len(lines) >= args.limit:
                break

            text = row.get(args.text_column, "")
            method = pick(r"METHOD=([A-Z]+)", text, "GET")
            path = pick(r"PATH=([^ ]+)", text, "/")
            query = pick(r"QUERY=([^ ]*)", text, "")
            proto = pick(r"PROTOCOL=([^ ]+)", text, "HTTP/1.1")

            target = path
            if query:
                target = f"{path}?{query}"

            ip = f"172.20.0.{(i % 250) + 1}"
            sec = i % 60
            size = str(800 + (i % 500))
            line = f'{ip} - - [10/May/2026:22:00:{sec:02d} +0800] "{method} {target} {proto}" 200 {size} "-" "Mozilla/5.0"'
            lines.append(line)

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out}")
    print(f"[OK] lines: {len(lines)}")
    if lines:
        print(lines[0])


if __name__ == "__main__":
    main()

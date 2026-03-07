#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from pipeline import DetectionPipeline
from storage.db import init_db


SUPPORTED_INPUT_TYPES = {"apache_log", "os_processes"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid IDS midterm prototype entrypoint"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input file",
    )
    parser.add_argument(
        "--input-type",
        required=True,
        choices=sorted(SUPPORTED_INPUT_TYPES),
        help="Input source type",
    )
    parser.add_argument(
        "--db",
        default="storage/ids.db",
        help="SQLite database path",
    )
    parser.add_argument(
        "--export",
        default="outputs/final_results.csv",
        help="Summary CSV export path",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional custom run name",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose summary",
    )
    return parser


def ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def build_run_id(run_name: str | None, input_file: Path, input_type: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        return f"{run_name}_{ts}"
    return f"{input_type}_{input_file.stem}_{ts}"


def print_summary(summary: dict, verbose: bool = False) -> None:
    print("\n========== RUN FINISHED ==========")
    if summary.get("warning"):
        print(f"Warning      : {summary['warning']}")
    print(f"Run ID       : {summary.get('run_id')}")
    print(f"Input type   : {summary.get('input_type')}")
    print(f"Input file   : {summary.get('input_path')}")
    print(f"Total records: {summary.get('total_records', 0)}")
    print(f"Rule hits    : {summary.get('rule_hits', 0)}")
    print(f"Anomalies    : {summary.get('anomalies', 0)}")
    print(f"Unknowns     : {summary.get('unknowns', 0)}")
    print(f"Benign       : {summary.get('benign', 0)}")
    print(f"DB path      : {summary.get('db_path')}")
    print(f"CSV export   : {summary.get('export_path')}")
    print("==================================\n")

    if verbose:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    db_path = Path(args.db)
    export_path = Path(args.export)

    if not input_path.exists():
        print(f"[ERROR] input file not found: {input_path}", file=sys.stderr)
        return 1

    ensure_parent_dir(db_path)
    ensure_parent_dir(export_path)

    init_db(str(db_path))

    run_id = build_run_id(args.run_name, input_path, args.input_type)

    pipeline = DetectionPipeline(
        db_path=str(db_path),
        export_path=str(export_path),
    )

    try:
        summary = pipeline.run(
            input_path=str(input_path),
            input_type=args.input_type,
            run_id=run_id,
        )
    except Exception as e:
        print(f"[ERROR] pipeline execution failed: {e}", file=sys.stderr)
        return 2

    summary["run_id"] = run_id
    summary["input_type"] = args.input_type
    summary["input_path"] = str(input_path)
    summary["db_path"] = str(db_path)
    summary["export_path"] = str(export_path)

    print_summary(summary, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
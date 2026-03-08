from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path.cwd()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CSV = OUTPUTS_DIR / "final_results.csv"
DEFAULT_SUMMARY_JSON = OUTPUTS_DIR / "final_results.summary.json"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pick_path(path_arg: Optional[str], default: Path) -> Path:
    if path_arg:
        candidate = Path(path_arg)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate
    return default


def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            enriched = dict(row)
            enriched["rule_hit_bool"] = _to_bool(row.get("rule_hit"))
            enriched["is_anomaly_bool"] = _to_bool(row.get("is_anomaly"))
            enriched["is_unknown_bool"] = _to_bool(row.get("is_unknown"))
            enriched["anomaly_score_num"] = _to_float(row.get("anomaly_score"))
            enriched["risk_score_num"] = _to_float(row.get("risk_score"))
            rows.append(enriched)
    return rows


def load_summary(summary_json_path: Path) -> Dict[str, Any]:
    if not summary_json_path.exists():
        return {}
    try:
        with summary_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def compute_summary(rows: List[Dict[str, Any]], summary_json: Dict[str, Any]) -> Dict[str, Any]:
    benign_count = sum(1 for r in rows if (r.get("final_label") or "").strip().lower() == "benign")
    result = {
        "total_records": len(rows),
        "rule_hits": sum(1 for r in rows if r.get("rule_hit_bool")),
        "anomalies": sum(1 for r in rows if r.get("is_anomaly_bool")),
        "unknowns": sum(1 for r in rows if r.get("is_unknown_bool")),
        "benign": benign_count,
        "run_id": rows[0].get("run_id", "N/A") if rows else summary_json.get("run_id", "N/A"),
        "input_type": rows[0].get("source_type", rows[0].get("input_type", "N/A")) if rows else summary_json.get("input_type", "N/A"),
        "anomaly_backend": summary_json.get("anomaly_backend") or "N/A",
        "anomaly_model": summary_json.get("anomaly_model") or (rows[0].get("anomaly_model", "N/A") if rows else "N/A"),
        "anomaly_version": summary_json.get("anomaly_version") or "N/A",
        "self_learning": summary_json.get("self_learning") or {},
    }
    return result


def apply_filters(rows: List[Dict[str, Any]], tab: str, q: str) -> List[Dict[str, Any]]:
    result = rows
    if tab == "rule":
        result = [r for r in result if r.get("rule_hit_bool")]
    elif tab == "anomaly":
        result = [r for r in result if r.get("is_anomaly_bool")]
    elif tab == "unknown":
        result = [r for r in result if r.get("is_unknown_bool")]
    elif tab == "normal":
        result = [r for r in result if (r.get("final_label") or "").strip().lower() == "benign"]

    if q:
        key = q.lower().strip()
        result = [
            r for r in result
            if key in (r.get("record_id") or "").lower()
            or key in (r.get("rule_name") or "").lower()
            or key in (r.get("final_label") or "").lower()
            or key in (r.get("raw_text") or "").lower()
            or key in (r.get("rule_reason") or "").lower()
            or key in (r.get("anomaly_reason") or "").lower()
            or key in (r.get("osr_reason") or "").lower()
        ]
    return result


def choose_active_row(rows: List[Dict[str, Any]], selected_record_id: str) -> Optional[Dict[str, Any]]:
    if selected_record_id:
        for row in rows:
            if row.get("record_id") == selected_record_id:
                return row
    if not rows:
        return None
    ranked = sorted(
        rows,
        key=lambda r: (
            0 if r.get("is_unknown_bool") else 1,
            0 if r.get("rule_hit_bool") else 1,
            -r.get("risk_score_num", 0.0),
            -r.get("anomaly_score_num", 0.0),
        ),
    )
    return ranked[0]


@app.route("/")
def index() -> str:
    csv_path = _pick_path(request.args.get("csv", "").strip(), DEFAULT_CSV)
    summary_json_path = _pick_path(request.args.get("summary", "").strip(), DEFAULT_SUMMARY_JSON)
    tab = request.args.get("tab", "all").strip() or "all"
    q = request.args.get("q", "").strip()
    selected_record_id = request.args.get("record_id", "").strip()

    rows = load_rows(csv_path)
    summary_json = load_summary(summary_json_path)
    summary = compute_summary(rows, summary_json)
    filtered_rows = apply_filters(rows, tab, q)
    active_row = choose_active_row(filtered_rows if filtered_rows else rows, selected_record_id)

    return render_template(
        "index.html",
        csv_path=str(csv_path),
        summary_json_path=str(summary_json_path),
        summary=summary,
        rows=filtered_rows,
        active_row=active_row,
        tab=tab,
        query=q,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

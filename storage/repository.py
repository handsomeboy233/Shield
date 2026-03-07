from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from schemas import (
    EventRecord,
    RuleMatchResult,
    AnomalyResult,
    OSRResult,
    FinalDetectionResult,
)


class DetectionRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def start_run(self, run_id: str, input_type: str, input_path: str) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO runs (run_id, input_type, input_path)
            VALUES (?, ?, ?)
            """,
            (run_id, input_type, input_path),
        )
        conn.commit()
        conn.close()

    def save_detection(
        self,
        run_id: str,
        event: EventRecord,
        rule_result: RuleMatchResult,
        anomaly_result: AnomalyResult,
        osr_result: OSRResult,
        final_result: FinalDetectionResult,
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO detections (
                run_id, record_id, source_type, rule_hit, rule_name, rule_severity,
                anomaly_score, is_anomaly, is_unknown, final_label, stage,
                confidence, risk_score, raw_text, reason_rule, reason_anomaly, reason_osr
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                event.record_id,
                event.source_type,
                int(rule_result.hit),
                rule_result.rule_name,
                rule_result.severity,
                anomaly_result.score,
                int(anomaly_result.is_anomaly),
                int(osr_result.is_unknown),
                final_result.final_label,
                final_result.stage,
                final_result.confidence,
                final_result.risk_score,
                event.raw_text,
                rule_result.reason,
                anomaly_result.reason,
                osr_result.reason,
            ),
        )

        conn.commit()
        conn.close()

    def export_summary_csv(self, export_path: str, rows: list[dict]) -> None:
        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)

        with export_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_id",
                    "record_id",
                    "source_type",
                    "rule_hit",
                    "rule_name",
                    "anomaly_score",
                    "is_anomaly",
                    "is_unknown",
                    "final_label",
                    "stage",
                    "confidence",
                    "risk_score",
                    "raw_text",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
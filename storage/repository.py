from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from schemas import AnomalyResult, EventRecord, FinalDetectionResult, OSRResult, RuleMatchResult


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
                anomaly_score, anomaly_threshold, anomaly_model, anomaly_version,
                is_anomaly, is_unknown, osr_method, osr_confidence,
                final_label, stage, confidence, risk_score, raw_text,
                reason_rule, reason_anomaly, reason_osr
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                event.record_id,
                event.source_type,
                int(rule_result.hit),
                rule_result.rule_name,
                rule_result.severity,
                anomaly_result.score,
                anomaly_result.threshold,
                anomaly_result.model_name,
                anomaly_result.model_version,
                int(anomaly_result.is_anomaly),
                int(osr_result.is_unknown),
                osr_result.method,
                osr_result.confidence,
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
     
        # 没有数据时也写一个空文件
        if not rows:
            with export_file.open("w", newline="", encoding="utf-8") as f:
                f.write("")
            return
     
        # 动态汇总所有字段，避免 pipeline 新增列后这里不同步
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
     
        with export_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",   # 即使将来还有额外字段，也不报错
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
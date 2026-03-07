from __future__ import annotations

from modules.anomaly_adapter import AnomalyEngine
from modules.rule_engine import RuleEngine
from modules.osr_stub import OSREngine
from storage.repository import DetectionRepository
from schemas import FinalDetectionResult


class DetectionPipeline:
    def __init__(self, db_path: str, export_path: str):
        self.db_path = db_path
        self.export_path = export_path

        self.rule_engine = RuleEngine()
        self.anomaly_engine = AnomalyEngine()
        self.osr_engine = OSREngine()
        self.repo = DetectionRepository(db_path=db_path)

    def run(self, input_path: str, input_type: str, run_id: str) -> dict:
        self.repo.start_run(run_id=run_id, input_type=input_type, input_path=input_path)

        payload = self.anomaly_engine.detect_file(
            input_path=input_path,
            input_type=input_type,
        )

        if payload["summary"]["total"] == 0:
            self.repo.export_summary_csv(
                export_path=self.export_path,
                rows=[],
            )
            return {
                "total_records": 0,
                "rule_hits": 0,
                "anomalies": 0,
                "unknowns": 0,
                "benign": 0,
                "warning": f"no valid records loaded from {input_path}",
            }

        total_records = 0
        rule_hits = 0
        anomalies = 0
        unknowns = 0
        benign = 0

        export_rows: list[dict] = []

        for item in payload["records"]:
            event = item["event"]
            anomaly_result = item["anomaly"]

            rule_result = self.rule_engine.match(event)
            osr_result = self.osr_engine.recognize(
                event=event,
                anomaly_result=anomaly_result,
                rule_result=rule_result,
            )

            final_result = self._aggregate(
                event=event,
                rule_result=rule_result,
                anomaly_result=anomaly_result,
                osr_result=osr_result,
            )

            self.repo.save_detection(
                run_id=run_id,
                event=event,
                rule_result=rule_result,
                anomaly_result=anomaly_result,
                osr_result=osr_result,
                final_result=final_result,
            )

            export_rows.append({
                "run_id": run_id,
                "record_id": event.record_id,
                "source_type": event.source_type,
                "rule_hit": int(rule_result.hit),
                "rule_name": rule_result.rule_name or "",
                "anomaly_score": anomaly_result.score,
                "is_anomaly": int(anomaly_result.is_anomaly),
                "is_unknown": int(osr_result.is_unknown),
                "final_label": final_result.final_label,
                "stage": final_result.stage,
                "confidence": final_result.confidence,
                "risk_score": final_result.risk_score,
                "raw_text": event.raw_text,
            })

            total_records += 1
            if rule_result.hit:
                rule_hits += 1
            if anomaly_result.is_anomaly:
                anomalies += 1
            if osr_result.is_unknown:
                unknowns += 1
            if final_result.final_label == "benign":
                benign += 1

        self.repo.export_summary_csv(
            export_path=self.export_path,
            rows=export_rows,
        )

        return {
            "total_records": total_records,
            "rule_hits": rule_hits,
            "anomalies": anomalies,
            "unknowns": unknowns,
            "benign": benign,
        }

    def _aggregate(self, event, rule_result, anomaly_result, osr_result) -> FinalDetectionResult:
        if rule_result.hit:
            return FinalDetectionResult(
                record_id=event.record_id,
                stage="rule",
                final_label=rule_result.rule_name or "known_suspicious",
                confidence=0.95,
                risk_score=max(0.80, anomaly_result.score),
            )

        if anomaly_result.is_anomaly and osr_result.is_unknown:
            return FinalDetectionResult(
                record_id=event.record_id,
                stage="osr",
                final_label="unknown_suspicious",
                confidence=osr_result.confidence,
                risk_score=max(0.85, anomaly_result.score),
            )

        if anomaly_result.is_anomaly:
            return FinalDetectionResult(
                record_id=event.record_id,
                stage="anomaly",
                final_label="suspicious",
                confidence=0.75,
                risk_score=anomaly_result.score,
            )

        return FinalDetectionResult(
            record_id=event.record_id,
            stage="normal",
            final_label="benign",
            confidence=0.90,
            risk_score=anomaly_result.score,
        )
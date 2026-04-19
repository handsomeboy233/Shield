from __future__ import annotations

import json
from pathlib import Path

from modules.anomaly_adapter import AnomalyEngine
from modules.osr_stub import OSREngine
from modules.rule_engine import RuleEngine
from modules.self_learning import SelfLearningEngine
from schemas import FinalDetectionResult
from storage.repository import DetectionRepository


class DetectionPipeline:
    def __init__(self, db_path: str, export_path: str):
        self.db_path = db_path
        self.export_path = export_path
        self.rule_engine = RuleEngine()
        self.anomaly_engine = AnomalyEngine()
        self.osr_engine = OSREngine()
        self.self_learning = SelfLearningEngine()
        self.repo = DetectionRepository(db_path=db_path)

    def run(self, input_path: str, input_type: str, run_id: str) -> dict:
        self.repo.start_run(run_id=run_id, input_type=input_type, input_path=input_path)
        payload = self.anomaly_engine.detect_file(input_path=input_path, input_type=input_type)

        if payload["summary"]["total"] == 0:
            self.repo.export_summary_csv(export_path=self.export_path, rows=[])
            summary = {
                "total_records": 0,
                "rule_hits": 0,
                "anomalies": 0,
                "unknowns": 0,
                "benign": 0,
                "warning": f"no valid records loaded from {input_path}",
                "anomaly_backend": payload["summary"].get("backend", "none"),
                "self_learning": self.self_learning.status(),
            }
            self._write_summary_sidecar(summary)
            return summary

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
            osr_meta = self._decode_osr_reason(osr_result.reason)

            final_result = self._aggregate(
                event=event,
                rule_result=rule_result,
                anomaly_result=anomaly_result,
                osr_result=osr_result,
                osr_meta=osr_meta,
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
                "rule_severity": rule_result.severity or "",
                "rule_reason": rule_result.reason,

                "anomaly_score": anomaly_result.score,
                "anomaly_threshold": anomaly_result.threshold,
                "anomaly_model": anomaly_result.model_name,
                "anomaly_version": anomaly_result.model_version,
                "is_anomaly": int(anomaly_result.is_anomaly),
                "anomaly_reason": anomaly_result.reason,

                "is_unknown": int(osr_result.is_unknown),
                "osr_method": osr_result.method,
                "osr_reason": osr_result.reason,
                "osr_pred_label": osr_meta.get("pred_label", ""),
                "osr_final_label": osr_meta.get("final_label", ""),
                "osr_max_prob": osr_meta.get("max_prob"),
                "osr_prob_threshold": osr_meta.get("prob_threshold"),
                "osr_distance": osr_meta.get("dist_to_pred_centroid"),
                "osr_distance_threshold": osr_meta.get("class_distance_threshold"),
                "osr_reject_reason": osr_meta.get("reject_reason", ""),

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

        self.repo.export_summary_csv(export_path=self.export_path, rows=export_rows)

        summary = {
            "total_records": total_records,
            "rule_hits": rule_hits,
            "anomalies": anomalies,
            "unknowns": unknowns,
            "benign": benign,
            "anomaly_backend": payload["summary"].get("backend"),
            "anomaly_model": payload["summary"].get("model_name"),
            "anomaly_version": payload["summary"].get("model_version"),
            "self_learning": self.self_learning.status(),
        }
        self._write_summary_sidecar(summary)
        return summary

    def _write_summary_sidecar(self, summary: dict) -> None:
        summary_path = Path(self.export_path).with_suffix(".summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _decode_osr_reason(self, reason: str) -> dict:
        if not reason:
            return {}
        try:
            payload = json.loads(reason)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {"raw_reason": reason}

    def _compute_risk_score(self, rule_result, anomaly_result, osr_result, osr_meta: dict) -> float:
        score = float(anomaly_result.score)
        pred_label = osr_meta.get("pred_label", "benign")

        if rule_result.hit:
            return max(0.90, score)
        if osr_result.is_unknown:
            return max(0.85, score)
        if anomaly_result.is_anomaly and pred_label in {"command_exec", "suspicious_path_probe"}:
            return max(0.80, score)
        if anomaly_result.is_anomaly:
            return max(0.70, score)
        return score

    def _aggregate(self, event, rule_result, anomaly_result, osr_result, osr_meta: dict) -> FinalDetectionResult:
        pred_label = osr_meta.get("pred_label", "benign")

        if rule_result.hit:
            return FinalDetectionResult(
                record_id=event.record_id,
                stage="rule",
                final_label=rule_result.rule_name or "known_suspicious",
                confidence=0.95,
                risk_score=self._compute_risk_score(rule_result, anomaly_result, osr_result, osr_meta),
            )

        # OSR 拒识优先于后续聚合
        if osr_result.is_unknown:
            return FinalDetectionResult(
                record_id=event.record_id,
                stage="osr",
                final_label="unknown",
                confidence=osr_result.confidence,
                risk_score=self._compute_risk_score(rule_result, anomaly_result, osr_result, osr_meta),
            )

        # 只在“已被异常模块判为异常”时，才用 OSR 的已知攻击类细化标签
        if anomaly_result.is_anomaly and pred_label in {"command_exec", "suspicious_path_probe"}:
            return FinalDetectionResult(
                record_id=event.record_id,
                stage="osr_known",
                final_label=pred_label,
                confidence=osr_result.confidence,
                risk_score=self._compute_risk_score(rule_result, anomaly_result, osr_result, osr_meta),
            )

        if anomaly_result.is_anomaly:
            return FinalDetectionResult(
                record_id=event.record_id,
                stage="anomaly",
                final_label="suspicious",
                confidence=0.75,
                risk_score=self._compute_risk_score(rule_result, anomaly_result, osr_result, osr_meta),
            )

        return FinalDetectionResult(
            record_id=event.record_id,
            stage="normal",
            final_label="benign",
            confidence=0.90,
            risk_score=self._compute_risk_score(rule_result, anomaly_result, osr_result, osr_meta),
        )
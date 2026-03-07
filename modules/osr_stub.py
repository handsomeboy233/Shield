from __future__ import annotations

from schemas import EventRecord, AnomalyResult, RuleMatchResult, OSRResult


class OSREngine:
    def __init__(self) -> None:
        self.method = "osr_stub"

    def recognize(
        self,
        event: EventRecord,
        anomaly_result: AnomalyResult,
        rule_result: RuleMatchResult,
    ) -> OSRResult:
        if rule_result.hit:
            return OSRResult(
                is_unknown=False,
                confidence=0.95,
                method=self.method,
                reason="rule already matched known suspicious pattern",
            )

        if anomaly_result.score >= 0.90:
            return OSRResult(
                is_unknown=True,
                confidence=min(0.99, anomaly_result.score),
                method=self.method,
                reason="very high anomaly score, marked as unknown",
            )

        if anomaly_result.score >= 0.75:
            return OSRResult(
                is_unknown=True,
                confidence=0.75,
                method=self.method,
                reason="high anomaly score, tentatively marked as unknown",
            )

        return OSRResult(
            is_unknown=False,
            confidence=0.60,
            method=self.method,
            reason="no strong unknown evidence",
        )
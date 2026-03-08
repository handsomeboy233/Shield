from __future__ import annotations

from schemas import AnomalyResult, EventRecord, OSRResult, RuleMatchResult


class OSREngine:
    def __init__(self) -> None:
        self.method = "adaptive_osr_stub"

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

        threshold = max(float(anomaly_result.threshold) + 0.05, 0.50)
        if anomaly_result.model_name.startswith("webhawk"):
            threshold = max(threshold, 0.70)

        strong_cluster_signal = any(token in anomaly_result.reason for token in ["noise_point=-1", "minority_cluster=1"])
        if anomaly_result.is_anomaly and (anomaly_result.score >= threshold or strong_cluster_signal):
            return OSRResult(
                is_unknown=True,
                confidence=min(0.95, max(0.72, anomaly_result.score)),
                method=self.method,
                reason=(
                    f"anomaly score {anomaly_result.score:.2f} >= unknown threshold {threshold:.2f}"
                    if anomaly_result.score >= threshold
                    else "cluster rarity indicates potential unknown pattern"
                ),
            )

        return OSRResult(
            is_unknown=False,
            confidence=0.60,
            method=self.method,
            reason=f"anomaly score {anomaly_result.score:.2f} < unknown threshold {threshold:.2f}",
        )

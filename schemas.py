from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EventRecord:
    record_id: str
    source_type: str
    raw_text: str

    ip: Optional[str] = None
    method: Optional[str] = None
    path: Optional[str] = None
    query: Optional[str] = None
    status_code: Optional[int] = None
    size: Optional[int] = None
    user_agent: Optional[str] = None

    process_name: Optional[str] = None
    pid: Optional[int] = None

    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleMatchResult:
    hit: bool
    rule_name: Optional[str]
    severity: Optional[str]
    reason: str


@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float
    threshold: float
    model_name: str
    model_version: str
    reason: str


@dataclass
class OSRResult:
    is_unknown: bool
    confidence: float
    method: str
    reason: str


@dataclass
class FinalDetectionResult:
    record_id: str
    stage: str
    final_label: str
    confidence: float
    risk_score: float
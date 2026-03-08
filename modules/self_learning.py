from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class SelfLearningEngine:
    """
    Safe placeholder for future online learning.

    The point is to expose a stable interface now:
    - dashboard can show the module exists
    - pipeline can include status in run summaries
    - later you can swap in rule updates / feedback replay without touching callers
    """

    def __init__(self) -> None:
        self.enabled = False
        self.mode = "placeholder"
        self.feedback_path = Path("storage/feedback.csv")
        self.rule_version = "rules-v1"
        self.model_version = "model-v1"

    def status(self) -> Dict[str, Any]:
        feedback_count = 0
        if self.feedback_path.exists():
            try:
                with self.feedback_path.open("r", encoding="utf-8", errors="ignore") as f:
                    feedback_count = max(0, sum(1 for _ in f) - 1)
            except Exception:
                feedback_count = 0

        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "feedback_samples": feedback_count,
            "rule_version": self.rule_version,
            "model_version": self.model_version,
            "last_update": "N/A",
            "note": "feedback loop reserved for later stage; current run uses fixed rules and models",
        }

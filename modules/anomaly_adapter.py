from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import unquote

from schemas import AnomalyResult, EventRecord

try:
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    np = None
    DBSCAN = PCA = NearestNeighbors = StandardScaler = None
    SKLEARN_AVAILABLE = False


APACHE_LOG_RE = re.compile(
    r'(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<time>[^\]]+)\]\s+"(?P<request>[^"]*)"\s+(?P<status>\d{3})\s+(?P<size>\S+)'
)


class AnomalyEngine:
    """
    Adapter layer for anomaly detection.

    - Preferred backend: WebHawk-style StandardScaler + PCA + DBSCAN.
    - Fallback backend: heuristic baseline (no extra deps required).
    """

    def __init__(self) -> None:
        self.fallback_model_name = "heuristic_baseline"
        self.fallback_model_version = "midterm-0.2"
        self.webhawk_model_name = "webhawk_pca_dbscan"
        self.webhawk_model_version = "midterm-0.3"
        self.min_records_for_dbscan = 5

    def detect_file(self, input_path: str, input_type: str) -> Dict[str, Any]:
        records = self.load_records(input_path=input_path, input_type=input_type)
        if not records:
            return {
                "records": [],
                "summary": {
                    "total": 0,
                    "anomalies": 0,
                    "backend": "none",
                    "model_name": self.fallback_model_name,
                    "model_version": self.fallback_model_version,
                },
            }

        if SKLEARN_AVAILABLE and len(records) >= self.min_records_for_dbscan:
            try:
                results = self._detect_file_webhawk(records)
                anomaly_count = sum(1 for item in results if item["anomaly"].is_anomaly)
                return {
                    "records": results,
                    "summary": {
                        "total": len(results),
                        "anomalies": anomaly_count,
                        "backend": "webhawk_core",
                        "model_name": self.webhawk_model_name,
                        "model_version": self.webhawk_model_version,
                    },
                }
            except Exception as exc:
                # Gracefully fall back instead of breaking the whole pipeline.
                results = self._detect_file_heuristic(records, extra_reason=f"fallback_from_webhawk={type(exc).__name__}")
                anomaly_count = sum(1 for item in results if item["anomaly"].is_anomaly)
                return {
                    "records": results,
                    "summary": {
                        "total": len(results),
                        "anomalies": anomaly_count,
                        "backend": "heuristic_fallback",
                        "model_name": self.fallback_model_name,
                        "model_version": self.fallback_model_version,
                    },
                }

        results = self._detect_file_heuristic(records)
        anomaly_count = sum(1 for item in results if item["anomaly"].is_anomaly)
        return {
            "records": results,
            "summary": {
                "total": len(results),
                "anomalies": anomaly_count,
                "backend": "heuristic_fallback",
                "model_name": self.fallback_model_name,
                "model_version": self.fallback_model_version,
            },
        }

    def load_records(self, input_path: str, input_type: str) -> List[EventRecord]:
        if input_type == "apache_log":
            return self._load_apache_log(input_path)
        if input_type == "os_processes":
            return self._load_os_processes(input_path)
        raise ValueError(f"unsupported input_type: {input_type}")

    def detect(self, event: EventRecord) -> AnomalyResult:
        if event.source_type == "apache_log":
            return self._detect_apache_heuristic(event)
        if event.source_type == "os_processes":
            return self._detect_process_heuristic(event)
        return AnomalyResult(
            is_anomaly=False,
            score=0.0,
            threshold=1.0,
            model_name=self.fallback_model_name,
            model_version=self.fallback_model_version,
            reason="unsupported source type",
        )

    def _detect_file_heuristic(self, records: List[EventRecord], extra_reason: str | None = None) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for event in records:
            anomaly = self.detect(event)
            if extra_reason:
                anomaly = AnomalyResult(
                    is_anomaly=anomaly.is_anomaly,
                    score=anomaly.score,
                    threshold=anomaly.threshold,
                    model_name=anomaly.model_name,
                    model_version=anomaly.model_version,
                    reason=f"{anomaly.reason}; {extra_reason}" if anomaly.reason else extra_reason,
                )
            rows.append({"event": event, "anomaly": anomaly})
        return rows

    def _detect_file_webhawk(self, records: List[EventRecord]) -> List[Dict[str, Any]]:
        feature_matrix = [self._build_feature_vector(record) for record in records]
        scaled = StandardScaler().fit_transform(feature_matrix)
        reduced = PCA(n_components=2, random_state=42).fit_transform(scaled)
        eps = self._estimate_eps(reduced)
        min_samples = 2 if len(records) < 30 else 3

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(reduced)
        labels = clustering.labels_
        counts = Counter(labels)

        minority_threshold = max(2, int(math.ceil(len(records) * 0.08)))
        if -1 in counts:
            minority_threshold += counts[-1]
        minority_clusters = {
            label for label, count in counts.items()
            if label != -1 and count <= minority_threshold
        }

        rows: List[Dict[str, Any]] = []
        for idx, event in enumerate(records):
            heuristic = self.detect(event)
            label = int(labels[idx])
            cluster_size = counts.get(label, 0)

            score = heuristic.score
            reasons = [f"backend=webhawk_core", f"eps={eps:.4f}", f"cluster={label}", f"cluster_size={cluster_size}"]
            threshold = 0.68 if event.source_type == "apache_log" else 0.60
            is_anomaly = False

            if label == -1:
                score = max(score, 0.95)
                is_anomaly = True
                reasons.append("noise_point=-1")
            elif label in minority_clusters:
                score = max(score, 0.78)
                is_anomaly = True
                reasons.append("minority_cluster=1")
            else:
                if heuristic.is_anomaly:
                    score = max(score, 0.62 if event.source_type == "apache_log" else 0.52)
                    is_anomaly = score >= threshold
                    reasons.append("heuristic_support=1")
                else:
                    reasons.append("cluster_normal=1")

            if heuristic.reason and heuristic.reason != "no obvious anomaly":
                reasons.append(f"heuristic={heuristic.reason}")

            anomaly = AnomalyResult(
                is_anomaly=is_anomaly,
                score=min(score, 1.0),
                threshold=threshold,
                model_name=self.webhawk_model_name,
                model_version=self.webhawk_model_version,
                reason="; ".join(reasons),
            )
            rows.append({"event": event, "anomaly": anomaly})

        return rows

    def _estimate_eps(self, reduced_matrix) -> float:
        neighbors = NearestNeighbors(n_neighbors=2)
        neighbors.fit(reduced_matrix)
        distances, _ = neighbors.kneighbors(reduced_matrix)
        second_nn = np.sort(distances[:, 1])
        # Light-weight replacement for the original KneeLocator based flow.
        idx = int(max(0, min(len(second_nn) - 1, round(0.9 * (len(second_nn) - 1)))))
        eps = float(second_nn[idx])
        return max(eps, 0.15)

    def _build_feature_vector(self, event: EventRecord) -> List[float]:
        if event.source_type == "apache_log":
            path = event.path or ""
            query = event.query or ""
            raw = event.raw_text or ""
            combined = f"{path}?{query}" if query else path

            special_chars = sum(1 for ch in combined if not ch.isalnum() and ch not in {"/", "-", "_", "."})
            encoded_tokens = raw.count("%")
            depth = len([segment for segment in path.split("/") if segment])
            query_pairs = len([part for part in query.split("&") if part]) if query else 0
            suspicious_hits = sum(
                1 for kw in ["../", "union", "select", "cmd=", "wget", "curl", "/.env", "/phpmyadmin"]
                if kw in (raw.lower()) or kw in (combined.lower())
            )
            return [
                float(len(path)),
                float(len(query)),
                float(depth),
                float(query_pairs),
                float(special_chars),
                float(encoded_tokens),
                float(event.status_code or 0),
                math.log1p(float(event.size or 0)),
                float(suspicious_hits),
                1.0 if query else 0.0,
                1.0 if (event.method or "").upper() not in {"GET", "POST", "HEAD", ""} else 0.0,
            ]

        cpu = float(event.features.get("cpu", 0.0) or 0.0)
        mem = float(event.features.get("mem", 0.0) or 0.0)
        virt = float(event.features.get("virt", 0.0) or 0.0)
        res = float(event.features.get("res", 0.0) or 0.0)
        pname = event.process_name or ""
        return [
            cpu,
            mem,
            math.log1p(virt),
            math.log1p(res),
            float(len(pname)),
            1.0 if any(token in pname.lower() for token in ["nmap", "sqlmap", "hydra", "masscan"]) else 0.0,
        ]

    def _load_apache_log(self, input_path: str) -> List[EventRecord]:
        records: List[EventRecord] = []
        path = Path(input_path)

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue

                match = APACHE_LOG_RE.match(line)
                if not match:
                    records.append(
                        EventRecord(
                            record_id=f"apache_{idx}",
                            source_type="apache_log",
                            raw_text=line,
                        )
                    )
                    continue

                request = match.group("request")
                ip = match.group("ip")
                status = int(match.group("status"))
                size_str = match.group("size")
                size = 0 if size_str == "-" else int(size_str)

                method = None
                target = None
                if request and request != "-":
                    parts = request.split()
                    if len(parts) >= 2:
                        method = parts[0]
                        target = parts[1]

                path_only = None
                query = None
                if target:
                    if "?" in target:
                        path_only, query = target.split("?", 1)
                    else:
                        path_only = target

                    if path_only is not None:
                        path_only = unquote(path_only)
                    if query is not None:
                        query = unquote(query)

                records.append(
                    EventRecord(
                        record_id=f"apache_{idx}",
                        source_type="apache_log",
                        raw_text=line,
                        ip=ip,
                        method=method,
                        path=path_only,
                        query=query,
                        status_code=status,
                        size=size,
                        features={},
                    )
                )

        return records

    def _load_os_processes(self, input_path: str) -> List[EventRecord]:
        records: List[EventRecord] = []
        path = Path(input_path)
        in_table = False

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue

                if line.strip().startswith("PID "):
                    in_table = True
                    continue

                if not in_table:
                    continue

                parts = line.split()
                if len(parts) < 12:
                    continue

                try:
                    pid = int(parts[0])
                except ValueError:
                    continue

                process_name = " ".join(parts[11:]) if len(parts) > 11 else parts[-1]

                try:
                    virt = float(parts[4])
                except ValueError:
                    virt = 0.0
                try:
                    res = float(parts[5])
                except ValueError:
                    res = 0.0
                try:
                    cpu = float(parts[8])
                except ValueError:
                    cpu = 0.0
                try:
                    mem = float(parts[9])
                except ValueError:
                    mem = 0.0

                records.append(
                    EventRecord(
                        record_id=f"proc_{idx}",
                        source_type="os_processes",
                        raw_text=line,
                        pid=pid,
                        process_name=process_name,
                        features={
                            "cpu": cpu,
                            "mem": mem,
                            "virt": virt,
                            "res": res,
                        },
                    )
                )

        return records

    def _detect_apache_heuristic(self, event: EventRecord) -> AnomalyResult:
        score = 0.0
        reasons: List[str] = []

        raw_lower = event.raw_text.lower()
        path_lower = (event.path or "").lower()
        query_lower = (event.query or "").lower()

        suspicious_keywords = [
            "union select",
            "or 1=1",
            "../",
            "/.env",
            "/wp-admin",
            "/phpmyadmin",
            "cmd=",
            "wget ",
            "curl ",
            "/bin/sh",
            "powershell",
        ]

        if event.status_code is not None and event.status_code >= 500:
            score += 0.35
            reasons.append(f"status={event.status_code}")

        if event.status_code in (401, 403, 404):
            score += 0.20
            reasons.append(f"status={event.status_code}")

        for kw in suspicious_keywords:
            if kw in raw_lower or kw in path_lower or kw in query_lower:
                score += 0.45
                reasons.append(f"keyword={kw}")
                break

        query_len = len(event.query or "")
        path_len = len(event.path or "")
        encoded_count = event.raw_text.count("%")
        special_count = sum(event.raw_text.count(ch) for ch in ["'", '"', ";", "|", "`", "$("])

        if query_len >= 80:
            score += 0.20
            reasons.append(f"long_query={query_len}")
        if path_len >= 40:
            score += 0.10
            reasons.append(f"long_path={path_len}")
        if encoded_count >= 6:
            score += 0.20
            reasons.append(f"encoded_tokens={encoded_count}")
        if special_count >= 2:
            score += 0.15
            reasons.append(f"special_tokens={special_count}")
        if event.method and event.method.upper() not in {"GET", "POST", "HEAD"}:
            score += 0.20
            reasons.append(f"method={event.method}")
        if event.size is not None and event.size > 1_000_000:
            score += 0.10
            reasons.append(f"size={event.size}")

        score = min(score, 1.0)
        threshold = 0.60

        return AnomalyResult(
            is_anomaly=score >= threshold,
            score=score,
            threshold=threshold,
            model_name=self.fallback_model_name,
            model_version=self.fallback_model_version,
            reason="; ".join(reasons) if reasons else "no obvious anomaly",
        )

    def _detect_process_heuristic(self, event: EventRecord) -> AnomalyResult:
        score = 0.0
        reasons: List[str] = []

        cpu = float(event.features.get("cpu", 0.0) or 0.0)
        mem = float(event.features.get("mem", 0.0) or 0.0)
        virt = float(event.features.get("virt", 0.0) or 0.0)
        res = float(event.features.get("res", 0.0) or 0.0)

        if cpu >= 20:
            score += 0.35
            reasons.append(f"cpu={cpu}")
        if cpu >= 50:
            score += 0.25

        if mem >= 1:
            score += 0.15
            reasons.append(f"mem={mem}")
        if mem >= 3:
            score += 0.15

        if virt >= 500000:
            score += 0.10
            reasons.append(f"virt={virt}")

        if res >= 200000:
            score += 0.10
            reasons.append(f"res={res}")

        score = min(score, 1.0)
        threshold = 0.45

        return AnomalyResult(
            is_anomaly=score >= threshold,
            score=score,
            threshold=threshold,
            model_name=self.fallback_model_name,
            model_version=self.fallback_model_version,
            reason="; ".join(reasons) if reasons else "no obvious anomaly",
        )

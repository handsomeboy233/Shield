from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any

from schemas import EventRecord, AnomalyResult


APACHE_LOG_RE = re.compile(
    r'(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<time>[^\]]+)\]\s+"(?P<request>[^"]*)"\s+(?P<status>\d{3})\s+(?P<size>\S+)'
)


class AnomalyEngine:
    def __init__(self) -> None:
        self.model_name = "heuristic_baseline"
        self.model_version = "midterm-0.1"

    def detect_file(self, input_path: str, input_type: str) -> Dict[str, Any]:
        records = self.load_records(input_path=input_path, input_type=input_type)

        results = []
        anomaly_count = 0

        for event in records:
            anomaly = self.detect(event)
            if anomaly.is_anomaly:
                anomaly_count += 1

            results.append({
                "event": event,
                "anomaly": anomaly,
            })

        return {
            "records": results,
            "summary": {
                "total": len(results),
                "anomalies": anomaly_count,
                "model_name": self.model_name,
                "model_version": self.model_version,
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
            return self._detect_apache(event)
        if event.source_type == "os_processes":
            return self._detect_process(event)

        return AnomalyResult(
            is_anomaly=False,
            score=0.0,
            threshold=1.0,
            model_name=self.model_name,
            model_version=self.model_version,
            reason="unsupported source type",
        )

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

                # top 输出的进程表最少需要：
                # PID USER PR NI VIRT RES SHR S %CPU %MEM TIME+ COMMAND
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

    def _detect_apache(self, event: EventRecord) -> AnomalyResult:
        score = 0.0
        reasons = []

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

        if event.size is not None and event.size > 1_000_000:
            score += 0.10
            reasons.append(f"size={event.size}")

        score = min(score, 1.0)
        threshold = 0.60

        return AnomalyResult(
            is_anomaly=score >= threshold,
            score=score,
            threshold=threshold,
            model_name=self.model_name,
            model_version=self.model_version,
            reason="; ".join(reasons) if reasons else "no obvious anomaly",
        )

    def _detect_process(self, event: EventRecord) -> AnomalyResult:
        score = 0.0
        reasons = []

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
            model_name=self.model_name,
            model_version=self.model_version,
            reason="; ".join(reasons) if reasons else "no obvious anomaly",
        )
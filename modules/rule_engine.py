from __future__ import annotations

import re

from schemas import EventRecord, RuleMatchResult


class RuleEngine:
    def __init__(self) -> None:
        self.apache_path_keywords = [
            "/wp-admin",
            "/phpmyadmin",
            "/.env",
            "/manager/html",
            "/cgi-bin",
            "/boaform",
        ]
        self.apache_query_keywords = [
            "union select",
            "or 1=1",
            "../",
            "cmd=",
            "wget ",
            "curl ",
            "/bin/sh",
            "powershell",
        ]
        self.process_patterns = [
            re.compile(r"(^|\s)nmap(\s|$)"),
            re.compile(r"(^|\s)netcat(\s|$)"),
            re.compile(r"(^|\s)hydra(\s|$)"),
            re.compile(r"(^|\s)sqlmap(\s|$)"),
            re.compile(r"(^|\s)masscan(\s|$)"),
            re.compile(r"(^|\s)nc(\s|$)"),
            re.compile(r"bash\s+-i"),
            re.compile(r"python\s+-c"),
            re.compile(r"perl\s+-e"),
        ]

    def match(self, event: EventRecord) -> RuleMatchResult:
        if event.source_type == "apache_log":
            return self._match_apache(event)
        if event.source_type == "os_processes":
            return self._match_process(event)
        return RuleMatchResult(
            hit=False,
            rule_name=None,
            severity=None,
            reason="unsupported source type for rule matching",
        )

    def _match_apache(self, event: EventRecord) -> RuleMatchResult:
        raw_lower = event.raw_text.lower()
        path_lower = (event.path or "").lower()
        query_lower = (event.query or "").lower()

        for kw in self.apache_path_keywords:
            if kw in path_lower:
                return RuleMatchResult(
                    hit=True,
                    rule_name="suspicious_path_probe",
                    severity="high",
                    reason=f"path matched keyword: {kw}",
                )

        for kw in self.apache_query_keywords:
            if kw in raw_lower or kw in query_lower:
                return RuleMatchResult(
                    hit=True,
                    rule_name="suspicious_payload_pattern",
                    severity="high",
                    reason=f"query/raw matched keyword: {kw}",
                )

        if event.status_code in (401, 403) and any(token in path_lower for token in ["/admin", "/manager", "/login"]):
            return RuleMatchResult(
                hit=True,
                rule_name="access_denied_probe",
                severity="medium",
                reason=f"http status {event.status_code} on sensitive path",
            )

        return RuleMatchResult(
            hit=False,
            rule_name=None,
            severity=None,
            reason="no apache rule matched",
        )

    def _match_process(self, event: EventRecord) -> RuleMatchResult:
        raw_lower = event.raw_text.lower()
        cpu = float(event.features.get("cpu", 0.0) or 0.0)
        mem = float(event.features.get("mem", 0.0) or 0.0)
        virt = float(event.features.get("virt", 0.0) or 0.0)

        for pattern in self.process_patterns:
            if pattern.search(raw_lower):
                return RuleMatchResult(
                    hit=True,
                    rule_name="suspicious_process_name",
                    severity="high",
                    reason=f"process matched suspicious pattern: {pattern.pattern}",
                )

        if cpu >= 80:
            return RuleMatchResult(
                hit=True,
                rule_name="high_cpu_usage",
                severity="medium",
                reason=f"cpu={cpu}",
            )

        if mem >= 5:
            return RuleMatchResult(
                hit=True,
                rule_name="high_mem_usage",
                severity="medium",
                reason=f"mem={mem}",
            )

        if virt >= 10_000_000:
            return RuleMatchResult(
                hit=True,
                rule_name="high_virtual_memory",
                severity="medium",
                reason=f"virt={virt}",
            )

        return RuleMatchResult(
            hit=False,
            rule_name=None,
            severity=None,
            reason="no process rule matched",
        )

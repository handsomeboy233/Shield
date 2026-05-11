#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd, timeout=None):
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout


class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_json({"ok": True})

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)

        if url.path == "/status":
            self.send_json({"ok": True, "message": "demo control server is running"})
            return

        if url.path == "/clear":
            cmd = [
                "/usr/bin/docker",
                "--host",
                "unix:///var/run/docker.sock",
                "compose",
                "exec",
                "webhawk_ui",
                "bin/rails",
                "runner",
                "Incident.delete_all; puts Incident.count",
            ]
            code, out = run_cmd(cmd)
            self.send_json({"ok": code == 0, "code": code, "output": out})
            return

        if url.path == "/run":
            scenario = qs.get("scenario", ["external_unknown"])[0]
            limit = qs.get("limit", ["30"])[0]

            if scenario == "custom":
                input_path = qs.get("input", [""])[0]
                input_format = qs.get("format", ["apache_log"])[0]
                text_col = qs.get("text_col", ["text"])[0]
                cmd = ["bash", "scripts/run_demo_full.sh", "custom", limit, input_path, input_format, text_col]
            else:
                cmd = ["bash", "scripts/run_demo_full.sh", scenario, limit]

            code, out = run_cmd(cmd)
            self.send_json({"ok": code == 0, "code": code, "output": out})
            return

        self.send_json({"ok": False, "error": "unknown endpoint"}, code=404)


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 9000), Handler)
    print("Demo control server running at http://127.0.0.1:9000")
    server.serve_forever()

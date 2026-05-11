#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def post_json(url, payload, timeout=180):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def clean_json_text(text):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    # 如果模型在 JSON 前后加了说明文字，尝试截出最外层 JSON。
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text


def call_ollama(ollama_url, model, sample_text):
    prompt = f"""
You are a cybersecurity analyst.

Analyze the following HTTP access log or normalized HTTP request.
Return ONLY valid JSON. Do not use markdown. Do not add explanations outside JSON.

Required JSON keys:
- potential_attack_attempt: true or false
- severity: "low", "medium", or "high"
- details: a concise explanation in Chinese
- CVE: an empty string if none
- owasp: an array of strings
- recommendation: a concise response suggestion in Chinese

Sample:
{sample_text}
""".strip()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    raw = post_json(f"{ollama_url.rstrip('/')}/api/generate", payload, timeout=240)
    obj = json.loads(raw)
    response = obj.get("response", "").strip()

    try:
        parsed = json.loads(clean_json_text(response))
    except Exception:
        parsed = {
            "potential_attack_attempt": True,
            "severity": "medium",
            "details": response if response else "模型未返回可解析的结构化结果，但该样本已进入语义增强流程。",
            "CVE": "",
            "owasp": [],
            "recommendation": "建议人工复核该请求，并结合访问路径、请求方法和上下文日志判断是否需要拦截或加入监控规则。",
        }

    parsed.setdefault("potential_attack_attempt", True)
    parsed.setdefault("severity", "medium")
    parsed.setdefault("details", "")
    parsed.setdefault("CVE", "")
    parsed.setdefault("owasp", [])
    parsed.setdefault("recommendation", "")

    if parsed["severity"] not in {"low", "medium", "high"}:
        parsed["severity"] = "medium"

    if not isinstance(parsed["owasp"], list):
        parsed["owasp"] = [str(parsed["owasp"])]

    return parsed


def read_inputs(path, input_type="auto", text_column="text", limit=None):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    if input_type == "auto":
        input_type = "csv" if path.suffix.lower() == ".csv" else "log"

    rows = []

    if input_type == "csv":
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            if text_column not in reader.fieldnames:
                raise ValueError(f"CSV 中没有列 {text_column}，实际列为: {reader.fieldnames}")

            for idx, row in enumerate(reader, start=1):
                text = (row.get(text_column) or "").strip()
                if not text:
                    continue
                rows.append({
                    "line_no": idx,
                    "content": text,
                    "cluster_id": row.get("cluster_id", ""),
                    "record_id": row.get("record_id", ""),
                })
                if limit and len(rows) >= limit:
                    break

    else:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rows.append({
                    "line_no": idx,
                    "content": line,
                    "cluster_id": "",
                    "record_id": "",
                })
                if limit and len(rows) >= limit:
                    break

    return rows


def incident_payload(host, item, advice, submit_all):
    is_attack = bool(advice.get("potential_attack_attempt", True))
    severity = advice.get("severity", "medium")

    if not is_attack and not submit_all:
        return None

    if not is_attack:
        severity = "low"

    owasp = advice.get("owasp", [])
    owasp_text = "\n".join(owasp) if owasp else "No OWASP found"

    cve = advice.get("CVE", "")
    cves_text = cve if cve else "No CVE found"

    prefix = ""
    if item.get("cluster_id") != "":
        prefix += f"[簇编号: {item['cluster_id']}] "
    if item.get("record_id") != "":
        prefix += f"[样本编号: {item['record_id']}] "

    return {
        "incident": {
            "cves": cves_text,
            "severity": severity,
            "status": "Open",
            "verdict": "Unknown" if is_attack else "Benign-like",
            "llm_insights": advice.get("details", ""),
            "log_line": item["line_no"],
            "log_line_content": prefix + item["content"],
            "attack_vector": "Web",
            "host": host,
            "owasp": owasp_text,
            "recommendation": advice.get("recommendation", ""),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入日志文件或 CSV 文件")
    parser.add_argument("--input-type", default="auto", choices=["auto", "log", "csv"])
    parser.add_argument("--text-column", default="text", help="CSV 中要分析的文本列，默认 text")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条，0 表示全部")
    parser.add_argument("--model", default="granite4:latest")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--webapp-url", default="http://127.0.0.1:8080/api/v1/incidents")
    parser.add_argument("--host", default="direct-ollama-demo")
    parser.add_argument("--submit-all", action="store_true", help="提交全部样本；不加则只提交模型判断为攻击的样本")
    parser.add_argument("--sleep", type=float, default=0.0, help="每条之间等待秒数，避免模型压力过大")
    args = parser.parse_args()

    rows = read_inputs(
        args.input,
        input_type=args.input_type,
        text_column=args.text_column,
        limit=args.limit if args.limit > 0 else None,
    )

    print(f"[INFO] loaded samples: {len(rows)}")
    print(f"[INFO] model: {args.model}")
    print(f"[INFO] ollama: {args.ollama_url}")
    print(f"[INFO] webapp: {args.webapp_url}")

    submitted = 0
    skipped = 0

    for i, item in enumerate(rows, start=1):
        print(f"\n[{i}/{len(rows)}] analyzing line {item['line_no']}: {item['content'][:160]}")

        try:
            advice = call_ollama(args.ollama_url, args.model, item["content"])
        except Exception as e:
            print(f"[ERROR] Ollama 调用失败: {e}")
            continue

        payload = incident_payload(args.host, item, advice, args.submit_all)
        if payload is None:
            skipped += 1
            print("[SKIP] 模型判定为低风险，未提交前端")
            continue

        try:
            post_json(args.webapp_url, payload, timeout=60)
            submitted += 1
            print(f"[OK] submitted severity={payload['incident']['severity']}")
        except Exception as e:
            print(f"[ERROR] 提交 Webhawk UI 失败: {e}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("\n[DONE]")
    print("submitted =", submitted)
    print("skipped =", skipped)


if __name__ == "__main__":
    main()

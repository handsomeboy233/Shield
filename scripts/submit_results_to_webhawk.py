#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import time
import urllib.request
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


def extract_json(text):
    text = text.strip().replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def yes(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_rows(path):
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def is_risky(row):
    if yes(row.get("rule_hit", "")):
        return True
    if yes(row.get("is_anomaly", "")):
        return True
    if yes(row.get("is_unknown", "")):
        return True

    label = str(row.get("final_label", "")).strip()
    if label and label != "benign":
        return True

    stage = str(row.get("stage", "")).strip()
    if stage and stage != "normal":
        return True

    return False


def select_rows(rows, scope, limit):
    selected = rows if scope == "all" else [r for r in rows if is_risky(r)]

    if not selected:
        print("[WARN] 没有筛出风险/未知样本，改为取前若干条。")
        selected = rows

    if limit > 0:
        selected = selected[:limit]

    return selected


def call_ollama(ollama_url, model, row):
    raw_text = row.get("raw_text", "")

    prompt = "\n".join([
        "你是一名网络安全分析人员。请根据下面的 HTTP 请求和检测结果进行安全分析。",
        "只返回合法 JSON，不要使用 Markdown，不要输出 JSON 以外的内容。",
        "JSON 字段必须包含：potential_attack_attempt, severity, details, CVE, owasp, recommendation。",
        "severity 只能是 low、medium、high。details 和 recommendation 必须使用中文。",
        "CVE 无法判断时填空字符串。owasp 必须是字符串数组。",
        "",
        "原始请求：",
        raw_text,
        "",
        "检测结果：",
        "规则命中: " + str(row.get("rule_hit", "")),
        "规则名称: " + str(row.get("rule_name", "")),
        "规则原因: " + str(row.get("rule_reason", "")),
        "异常分数: " + str(row.get("anomaly_score", "")),
        "是否异常: " + str(row.get("is_anomaly", "")),
        "是否未知: " + str(row.get("is_unknown", "")),
        "开放集原因: " + str(row.get("osr_reason", "")),
        "最终标签: " + str(row.get("final_label", "")),
        "阶段: " + str(row.get("stage", "")),
        "风险分数: " + str(row.get("risk_score", "")),
    ])

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    raw = post_json(ollama_url.rstrip("/") + "/api/generate", payload, timeout=240)
    obj = json.loads(raw)
    response = obj.get("response", "")

    try:
        ans = json.loads(extract_json(response))
    except Exception:
        ans = {
            "potential_attack_attempt": True,
            "severity": "medium",
            "details": response if response else "模型未返回可解析的结构化结果。",
            "CVE": "",
            "owasp": [],
            "recommendation": "建议结合规则命中、异常分数和开放集判别结果进行人工复核。",
        }

    ans.setdefault("potential_attack_attempt", True)
    ans.setdefault("severity", "medium")
    ans.setdefault("details", "")
    ans.setdefault("CVE", "")
    ans.setdefault("owasp", [])
    ans.setdefault("recommendation", "")

    if ans["severity"] not in {"low", "medium", "high"}:
        ans["severity"] = "medium"

    if not isinstance(ans["owasp"], list):
        ans["owasp"] = [str(ans["owasp"])]

    return ans


def submit_incident(webapp_url, row, ans, host, idx):
    is_unknown = yes(row.get("is_unknown", ""))
    rule_hit = yes(row.get("rule_hit", ""))
    label = row.get("final_label", "")

    if is_unknown:
        verdict = "Unknown"
    elif rule_hit:
        verdict = "Known attack"
    elif label == "benign":
        verdict = "Benign-like"
    else:
        verdict = "Suspicious"

    cve = ans.get("CVE", "")
    owasp = ans.get("owasp", [])

    artifact = (
        "[记录编号: " + str(row.get("record_id", "")) + "] "
        "[阶段: " + str(row.get("stage", "")) + "] "
        "[最终标签: " + str(label) + "] "
        + str(row.get("raw_text", ""))
    )

    payload = {
        "incident": {
            "cves": cve if cve else "No CVE found",
            "severity": ans.get("severity", "medium"),
            "status": "Open",
            "verdict": verdict,
            "llm_insights": ans.get("details", ""),
            "log_line": idx,
            "log_line_content": artifact,
            "attack_vector": "Web",
            "host": host,
            "owasp": "\n".join(owasp) if owasp else "No OWASP found",
            "recommendation": ans.get("recommendation", ""),
        }
    }

    post_json(webapp_url, payload, timeout=60)


def save_enhanced_csv(path, rows):
    if not rows:
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--scope", choices=["risky", "all"], default="risky")
    parser.add_argument("--model", default="granite4:latest")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--webapp-url", default="http://127.0.0.1:8080/api/v1/incidents")
    parser.add_argument("--host", default="full-osr-llm-demo")
    parser.add_argument("--out", default="outputs/full_osr_llm_demo/semantic_enhanced_results.csv")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    if not Path(args.csv).exists():
        raise SystemExit("[ERROR] 找不到检测结果 CSV: " + args.csv)

    rows = load_rows(args.csv)
    selected = select_rows(rows, args.scope, args.limit)

    print("检测结果文件:", args.csv)
    print("总记录数:", len(rows))
    print("进入语义增强记录数:", len(selected))

    enhanced = []

    for idx, row in enumerate(selected, start=1):
        raw_text = row.get("raw_text", "")
        print("\n[" + str(idx) + "/" + str(len(selected)) + "] " + raw_text[:140])

        try:
            ans = call_ollama(args.ollama_url, args.model, row)
        except Exception as e:
            print("[ERROR] Ollama 调用失败:", e)
            continue

        new_row = dict(row)
        new_row["llm_severity"] = ans.get("severity", "")
        new_row["llm_details"] = ans.get("details", "")
        new_row["llm_cve"] = ans.get("CVE", "")
        new_row["llm_owasp"] = "; ".join(ans.get("owasp", []))
        new_row["llm_recommendation"] = ans.get("recommendation", "")
        enhanced.append(new_row)

        try:
            submit_incident(args.webapp_url, new_row, ans, args.host, idx)
            print("[已提交前端] severity=" + str(ans.get("severity", "")))
        except Exception as e:
            print("[ERROR] 提交前端失败:", e)

        if args.sleep > 0:
            time.sleep(args.sleep)

    save_enhanced_csv(args.out, enhanced)

    print("\n========== 完成 ==========")
    print("语义增强 CSV:", args.out)
    print("提交前端数量:", len(enhanced))
    print("==========================")


if __name__ == "__main__":
    main()

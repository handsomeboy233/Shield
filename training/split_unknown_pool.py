import re
import pandas as pd
from pathlib import Path

INPUT_CSV = "outputs/osr/unknown_infer_results.csv"
OUTPUT_DIR = "outputs/osr"

# -----------------------------
# 1) 规则模式
# -----------------------------

# 明显属于分布外正常样本：静态资源 / 文档页 / 说明页 / 爬虫
STATIC_EXT = re.compile(
    r"\.(css|js|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot|map|pdf|txt|xml)(\?|$)",
    re.I,
)

DOC_LIKE = re.compile(
    r"(datasets description|network/.*\.html|ssl\.html|dns\.html|ftp\.html|"
    r"readme|index\.html|about|docs?|manual|tutorial|guide)",
    re.I,
)

BOT_LIKE = re.compile(
    r"(bot|spider|crawler|sogou|bingpreview|ahrefs|semrush|yandex|mj12bot|curl)",
    re.I,
)

COMMON_BENIGN_PAGE = re.compile(
    r"(GET / HTTP/1\.1|GET /index\.html|GET /favicon\.ico|GET /robots\.txt)",
    re.I,
)

# 明显像攻击/探测/漏洞利用，但不一定属于你当前已知类
ATTACK_LIKE = re.compile(
    r"(\.\./|%2e%2e|%252e%252e|"              # 路径穿越
    r"union\s+select|select.+from|sleep\(|benchmark\(|"   # SQLi
    r"cmd=|wget|curl\s+http|bash -c|powershell|"          # 命令执行/下载
    r"/cgi-bin/|/shell|webshell|cmd\.php|"                # shell/cgibin
    r"\.env|phpunit|jmx-console|boaform|invokefunction|"  # 常见漏洞面
    r"eval\(|file_put_contents|base64_decode\(|assert\(|"
    r"upload\.php|vuln\.php|actuator|/wp-|/manager/html|/console/|"
    r"struts|solr|jenkins|nacos|zabbix|weblogic|thinkphp|"
    r"autodiscover|owa|ecp|phpmyadmin|elFinder|vendor/phpunit)",
    re.I,
)

# 高风险路径探测，但不完全等同于当前 known 类
SUSPICIOUS_PATHS = re.compile(
    r"(/admin|/login|/console|/api|/debug|/test|/backup|/old|/tmp|"
    r"/config|/db|/phpinfo|/server-status|/actuator|/metrics|"
    r"/owa|/ecp|/autodiscover|/vendor/|/cgi-bin/)",
    re.I,
)

# -----------------------------
# 2) 单条样本分类逻辑
# -----------------------------
def classify_unknown_row(row):
    text = str(row.get("text", ""))
    reject_reason = str(row.get("reject_reason", ""))
    pred_label = str(row.get("pred_label", ""))

    # 先判 OOD-benign
    if STATIC_EXT.search(text):
        return "ood_benign", "static_resource"

    if DOC_LIKE.search(text):
        return "ood_benign", "doc_like_page"

    if BOT_LIKE.search(text):
        return "ood_benign", "bot_like"

    if COMMON_BENIGN_PAGE.search(text) and "STATUS=200" in text:
        return "ood_benign", "common_benign_page"

    # 再判 attack-like unknown
    if ATTACK_LIKE.search(text):
        return "attack_like_unknown", "attack_keyword"

    if SUSPICIOUS_PATHS.search(text) and ("STATUS=404" in text or "STATUS=403" in text or "STATUS=500" in text):
        return "attack_like_unknown", "suspicious_path_with_error_status"

    # 一些被 OSR 拒识但又预测成攻击类的，也优先归 attack-like
    if pred_label in {"suspicious_path_probe", "command_exec"} and reject_reason != "accepted":
        return "attack_like_unknown", "rejected_but_attack_pred"

    # 剩下的进入人工补充
    return "needs_review", "uncertain"


# -----------------------------
# 3) 主流程
# -----------------------------
def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # 只处理 unknown
    unk = df[df["final_label"] == "unknown"].copy()

    pool_types = []
    auto_reasons = []

    for _, row in unk.iterrows():
        pool_type, reason = classify_unknown_row(row)
        pool_types.append(pool_type)
        auto_reasons.append(reason)

    unk["unknown_pool_type"] = pool_types
    unk["auto_pool_reason"] = auto_reasons

    # 输出总表
    unk.to_csv(out_dir / "unknown_only_with_pool.csv", index=False)

    # 分池输出
    unk[unk["unknown_pool_type"] == "ood_benign"].to_csv(
        out_dir / "ood_benign_pool.csv", index=False
    )
    unk[unk["unknown_pool_type"] == "attack_like_unknown"].to_csv(
        out_dir / "attack_like_unknown_pool.csv", index=False
    )
    needs_review = unk[unk["unknown_pool_type"] == "needs_review"].copy()

    # 给人工补充留列
    needs_review["manual_pool"] = ""
    needs_review["manual_true_label"] = ""
    needs_review["notes"] = ""

    needs_review.to_csv(out_dir / "unknown_needs_review.csv", index=False)

    print("=== auto split done ===")
    print(unk["unknown_pool_type"].value_counts())
    print("\nSaved files:")
    print(out_dir / "unknown_only_with_pool.csv")
    print(out_dir / "ood_benign_pool.csv")
    print(out_dir / "attack_like_unknown_pool.csv")
    print(out_dir / "unknown_needs_review.csv")


if __name__ == "__main__":
    main()
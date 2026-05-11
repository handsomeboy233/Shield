#!/bin/bash
# 文件名: run_llm_webhawk.sh
# 功能: 从第四章未知样本代表样本生成 LLM 语义解释，并输出最终 CSV
# 依赖: Docker 容器中的 Webhawk 和 Ollama 已运行

# ------------------------------
# 用户需要修改的变量
# ------------------------------
DOCKER_NAME="webhawk_ollama"                 # Docker 容器名称
MODEL_CLI="ollama query qwen2.5"            # Docker 内 LLM CLI
INPUT_CSV="outputs/osr/inspection_main/unknown_representatives.csv"
TMP_CSV="outputs/osr/manual_samples_for_llm.csv"
LLM_OUTPUT_CSV="outputs/osr/llm_explanations.csv"
FINAL_CSV="outputs/osr/unknown_with_llm_explanations.csv"
SAMPLES_PER_CLUSTER=3                        # 每个簇选几条样本用于解释
# ------------------------------

mkdir -p outputs/osr

echo "[1/5] 挑选每簇代表样本"
python3 - <<'PY'
import pandas as pd

INPUT_CSV = "outputs/osr/inspection_main/unknown_representatives.csv"
TMP_CSV = "outputs/osr/manual_samples_for_llm.csv"
SAMPLES_PER_CLUSTER = 3

try:
    reps = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    raise SystemExit(f"文件 {INPUT_CSV} 不存在，请确认路径")

selected = reps.groupby('cluster_id').head(SAMPLES_PER_CLUSTER)
selected[['cluster_id','text']].to_csv(TMP_CSV, index=False)
print(f"临时 CSV 已生成: {TMP_CSV}")
print(selected.head())
PY

echo "[2/5] 清空旧的 LLM 输出文件"
> $LLM_OUTPUT_CSV

echo "[3/5] 使用 Docker 调用 LLM 对样本生成语义解释"
# 使用 while 循环逐行发送给 Docker 容器内的 LLM
tail -n +2 $TMP_CSV | while IFS=, read -r cluster_id text
do
    # 转义双引号，防止 CLI 报错
    safe_text=$(echo "$text" | sed 's/"/\\"/g')

    # 执行 LLM CLI
    llm_output=$(docker exec -i $DOCKER_NAME bash -c "$MODEL_CLI \"请对以下请求进行安全性解释并给出处置建议：$safe_text\"")

    # 输出到 CSV
    echo "$cluster_id,\"$text\",\"$llm_output\"" >> $LLM_OUTPUT_CSV
done

echo "[4/5] 合并原始簇信息和 LLM 输出"
python3 - <<'PY'
import pandas as pd

CLUSTER_CSV = "outputs/osr/inspection_main/unknown_representatives.csv"
LLM_CSV = "outputs/osr/llm_explanations.csv"
FINAL_CSV = "outputs/osr/unknown_with_llm_explanations.csv"

clustered = pd.read_csv(CLUSTER_CSV)
llm = pd.read_csv(LLM_CSV)

merged = pd.merge(clustered, llm, on=['cluster_id','text'], how='left')
merged.to_csv(FINAL_CSV, index=False, encoding='utf-8')
print(f"最终展示 CSV 已生成: {FINAL_CSV}")
print(merged.head())
PY

echo "[5/5] 完成！请用 $FINAL_CSV 在界面或演示中展示代表样本及 LLM 语义解释。"
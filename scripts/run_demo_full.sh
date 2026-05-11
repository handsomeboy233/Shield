#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-external_unknown}"
LIMIT="${2:-30}"
CUSTOM_INPUT="${3:-}"
CUSTOM_FORMAT="${4:-apache_log}"
CUSTOM_TEXT_COL="${5:-text}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-granite4:latest}"
SCOPE="${SCOPE:-risky}"
INPUT_LIMIT="${INPUT_LIMIT:-120}"
CLEAR="${CLEAR:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p outputs/full_osr_llm_demo

if [[ "$CLEAR" == "1" ]]; then
  echo "[0/4] 清空网页旧记录"
  /usr/bin/docker --host unix:///var/run/docker.sock compose exec webhawk_ui \
    bin/rails runner "Incident.delete_all; puts Incident.count" || true
fi

case "$SCENARIO" in
  real2021)
    INPUT_FILE="./HTTP_LOGS/access.log.2021-10-22"
    INPUT_FORMAT="apache_log"
    HOST_NAME="real2021-full-osr-llm"
    ;;
  real2022)
    INPUT_FILE="./HTTP_LOGS/access.log.2022-12-22"
    INPUT_FORMAT="apache_log"
    HOST_NAME="real2022-full-osr-llm"
    ;;
  external_unknown)
    INPUT_FILE="datasets/osr/processed/csic2010_external_unknown_attack_canonical_text_label.csv"
    INPUT_FORMAT="csv"
    HOST_NAME="external-unknown-full-osr-llm"
    ;;
  unknown_reps)
    INPUT_FILE="outputs/osr/inspection_main/unknown_representatives.csv"
    INPUT_FORMAT="csv"
    HOST_NAME="unknown-representatives-full-osr-llm"
    ;;
  custom)
    INPUT_FILE="$CUSTOM_INPUT"
    INPUT_FORMAT="$CUSTOM_FORMAT"
    HOST_NAME="custom-full-osr-llm"
    ;;
  *)
    echo "[ERROR] 未知场景: $SCENARIO"
    echo "可选: real2021, real2022, external_unknown, unknown_reps, custom"
    exit 1
    ;;
esac

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "[ERROR] 输入文件不存在: $INPUT_FILE"
  exit 1
fi

RUN_INPUT="$INPUT_FILE"

if [[ "$INPUT_FORMAT" == "csv" ]]; then
  echo "[1/4] 将规范化 CSV 转为 Apache 日志"
  RUN_INPUT="outputs/full_osr_llm_demo/${SCENARIO}_input.log"
  "$PYTHON_BIN" scripts/convert_canonical_csv.py \
    --input "$INPUT_FILE" \
    --output "$RUN_INPUT" \
    --text-column "$CUSTOM_TEXT_COL" \
    --limit "$INPUT_LIMIT"
else
  echo "[1/4] 使用 Apache 日志输入: $RUN_INPUT"
fi

EXPORT_CSV="outputs/full_osr_llm_demo/${SCENARIO}_final_results.csv"
DB_FILE="outputs/full_osr_llm_demo/${SCENARIO}_ids.db"

echo "[2/4] 运行本文主检测流程：规则筛选、异常分析、开放集判别"
"$PYTHON_BIN" main.py \
  --input "$RUN_INPUT" \
  --input-type apache_log \
  --export "$EXPORT_CSV" \
  --db "$DB_FILE" \
  --run-name "$SCENARIO" \
  --verbose

echo "[3/4] 调用 Ollama 进行语义增强，并提交到 Webhawk 前端"
"$PYTHON_BIN" scripts/submit_results_to_webhawk.py \
  --csv "$EXPORT_CSV" \
  --limit "$LIMIT" \
  --scope "$SCOPE" \
  --model "$MODEL" \
  --host "$HOST_NAME" \
  --out "outputs/full_osr_llm_demo/${SCENARIO}_semantic_enhanced.csv"

echo "[4/4] 完整流程完成"
echo "检测结果: $EXPORT_CSV"
echo "语义增强结果: outputs/full_osr_llm_demo/${SCENARIO}_semantic_enhanced.csv"

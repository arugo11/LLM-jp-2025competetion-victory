#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFERENCE_DIR="${REPO_ROOT}/inference"
MATH_EVAL_DIR="${REPO_ROOT}/math-eval"
OUTPUT_DIR="${REPO_ROOT}/output"
INFERENCE_OUTPUT_DIR="${INFERENCE_DIR}/output"
INPUT_PATH="${INPUT_PATH:-${REPO_ROOT}/input/final.jsonl}"
PREPARED_FINAL_PATH="${PREPARED_FINAL_PATH:-${OUTPUT_DIR}/final_with_id.jsonl}"
MODEL_REPO="${MODEL_REPO:-HayatoHongoEveryonesAI/open-instruct-grpo-fast}"
MODEL_REVISION="${MODEL_REVISION:-grpo_fast__3__1770120411}"
MODEL_PATH="${MODEL_PATH:-${INFERENCE_DIR}/models/HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-expand-checkpoint-1900-12B-1725}"
NUM_SAMPLES="${NUM_SAMPLES:-40}"
EVAL_K_VALUES="${EVAL_K_VALUES:-1,20,40}"
RUN_LABEL="${RUN_LABEL:-cons40}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.7}"
WAIT_COUNT="${WAIT_COUNT:-0}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}"
OUTPUT_STEM="${OUTPUT_STEM:-$(basename "${MODEL_PATH}")W0-${RUN_LABEL}}"
PREDICTION_PATH="${INFERENCE_OUTPUT_DIR}/output-${OUTPUT_STEM}-reproduce-final.jsonl"
ALL_SAMPLES_PATH="${INFERENCE_OUTPUT_DIR}/output-${OUTPUT_STEM}-reproduce-final_all_samples.jsonl"
ACCURACY_PATH="${MATH_EVAL_DIR}/accuracy/acc-${OUTPUT_STEM}-final.jsonl"

mkdir -p "${OUTPUT_DIR}" "${INFERENCE_OUTPUT_DIR}" "${MATH_EVAL_DIR}/accuracy"

cd "${INFERENCE_DIR}"

if [ -z "${INFERENCE_PYTHON:-}" ]; then
    if [ -x "${INFERENCE_DIR}/.venv/bin/python" ]; then
        INFERENCE_PYTHON="${INFERENCE_DIR}/.venv/bin/python"
    elif [ -x "${INFERENCE_DIR}/.venv-vllm-cu118-match/bin/python" ]; then
        INFERENCE_PYTHON="${INFERENCE_DIR}/.venv-vllm-cu118-match/bin/python"
    else
        uv sync
        INFERENCE_PYTHON="${INFERENCE_DIR}/.venv/bin/python"
    fi
fi

if [ ! -x "${INFERENCE_PYTHON}" ]; then
    echo "Inference python not found at ${INFERENCE_PYTHON}" >&2
    echo "Set INFERENCE_PYTHON explicitly or run uv sync in ${INFERENCE_DIR}" >&2
    exit 1
fi

if [ ! -f "${INPUT_PATH}" ]; then
    echo "Input file not found at ${INPUT_PATH}" >&2
    exit 1
fi

MODEL_READY=0
if [ -f "${MODEL_PATH}/model.safetensors.index.json" ]; then
    if "${INFERENCE_PYTHON}" - "${MODEL_PATH}/model.safetensors.index.json" "${MODEL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

index_path = Path(sys.argv[1])
model_dir = Path(sys.argv[2])
index = json.loads(index_path.read_text())
required = sorted(set(index["weight_map"].values()))
missing = [name for name in required if not (model_dir / name).is_file()]
sys.exit(0 if not missing else 1)
PY
    then
        MODEL_READY=1
    fi
fi

if [ "${MODEL_READY}" -ne 1 ]; then
    "${INFERENCE_PYTHON}" download_model.py \
        --model_name "${MODEL_REPO}" \
        --revision "${MODEL_REVISION}" \
        --local_dir "${MODEL_PATH}"
fi

"${INFERENCE_PYTHON}" prepare_eval_jsonl.py \
    --input_path "${INPUT_PATH}" \
    --output_path "${PREPARED_FINAL_PATH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${INFERENCE_PYTHON}" self-consistency.py \
    --model_path "${MODEL_PATH}" \
    --input_path "${PREPARED_FINAL_PATH}" \
    --output_path "${PREDICTION_PATH}" \
    --max_tokens "${MAX_TOKENS}" \
    --num_samples "${NUM_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --wait_count "${WAIT_COUNT}"

cd "${MATH_EVAL_DIR}"
uv sync
uv run python src/math_eval/eval_consistency.py \
    "${ALL_SAMPLES_PATH}" \
    "${PREPARED_FINAL_PATH}" \
    -o "${ACCURACY_PATH}" \
    -k "${EVAL_K_VALUES}"

printf 'Prepared input: %s\n' "${PREPARED_FINAL_PATH}"
printf 'Inference python: %s\n' "${INFERENCE_PYTHON}"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES_VALUE}"
printf 'Prediction: %s\n' "${PREDICTION_PATH}"
printf 'All samples: %s\n' "${ALL_SAMPLES_PATH}"
printf 'Accuracy: %s\n' "${ACCURACY_PATH}"

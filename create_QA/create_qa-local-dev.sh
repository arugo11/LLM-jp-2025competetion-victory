#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N create_qa_devjsonl
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=08:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

# cd create_QA
# qsub ./create_qa-local-dev.sh
cd $PBS_O_WORKDIR # 実行したディレクトリに移動

JOBID=${PBS_JOBID%%.*}
mkdir -p ./.log
LOGFILE=./.log/create_qa-devjsonl-$JOBID.out
ERRFILE=./.log/create_qa-devjsonl-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

# Avoid port collisions for vLLM. Local sandbox is fixed to 6000.
BASE_PORT=$((10000 + (JOBID % 20000)))
TIR_LLM_PORT=$((BASE_PORT + 1000))
SANDBOX_PORT=6000
echo "TIR_LLM_PORT=${TIR_LLM_PORT}"
echo "SANDBOX_PORT=${SANDBOX_PORT}"

set -euxo pipefail

mkdir -p output

# uvのキャッシュディレクトリを設定 (キャッシュはvscode内で見たいため、プロジェクト下とします。)
export UV_CACHE_DIR="$HOME/LLM-jp-2025competetion-victory/.cache/uv"
mkdir -p "$UV_CACHE_DIR"

# コードを書き替えるたびにビルドする必要があるそうです。
singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       dist/create_qa.sif create_qa.def

# ローカルdev.jsonlを入力して推論
INPUT_JSONL="$(pwd)/input/dev.jsonl"
OUTPUT_JSONL="$(pwd)/output/dev_generated.jsonl"
MODEL_PATH="/app/models/openai/gpt-oss-120b"
TIR_MODEL_NAME="gpt-oss-120b"
OUTPUT_REPO_ID="${OUTPUT_REPO_ID:-}"

if [[ ! -f "$INPUT_JSONL" ]]; then
  echo "Input JSONL not found: $INPUT_JSONL" >&2
  exit 1
fi

REPO_ARGS=()
if [[ -n "$OUTPUT_REPO_ID" ]]; then
  REPO_ARGS=(--repo_id "$OUTPUT_REPO_ID")
fi

singularity exec --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env HF_TOKEN="$HF_TOKEN" \
    --bind "$(pwd)/models:/app/models" \
    --bind "$(pwd)/output:/app/output" \
    --bind "$(pwd)/input:/app/input" \
    dist/create_qa.sif \
    python3 create_answers.py \
    --input_jsonl "/app/input/dev.jsonl" \
    --output_jsonl "/app/output/dev_generated.jsonl" \
    "${REPO_ARGS[@]}" \
    --hf_token "$HF_TOKEN" \
    --sandbox-block-network \
    --model_path "$MODEL_PATH" \
    --max_tokens 4096 \
    --tir-model-name "$TIR_MODEL_NAME" \
    --tir-endpoint-type responses \
    --tir-llm-host "127.0.0.1" \
    --tir-llm-port "$TIR_LLM_PORT" \
    --tir-sandbox-host "127.0.0.1" \
    --tir-sandbox-port "$SANDBOX_PORT" \
    --tir-max-retries 5 \
    --tir-code-timeout 30 \
    --tir-max-output-chars 4000 \
    --tir-temperature 0.2 \
    --vllm-tensor-parallel-size 1 \
    --vllm-extra-args ""

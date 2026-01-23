#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N inference-1gpu
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=4:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

: ${MODEL_USER:="HayatoHongoEveryonesAI"}
: ${MODEL_REPO:="llm-jp-4-8b-instruct"}
: ${MODEL_BASE_PATH:="models/"}

MODEL_NAME="${MODEL_USER}/${MODEL_REPO}"
MODEL_PATH="$MODEL_BASE_PATH/$MODEL_NAME"

echo "Model name: $MODEL_NAME"
echo "Model path: $MODEL_PATH"

# Singularityイメージ保存用ディレクトリの作成
mkdir -p ./dist
# 推論結果の出力ディレクトリの作成
mkdir -p ./output

source /etc/profile.d/modules.sh
module load cuda/12.8

cd $PBS_O_WORKDIR #実行したディレクトリに移動
echo "Current working directory is $(pwd)"

JOBID=${PBS_JOBID%%.*}
# ログの保存
# 保存先は実行ディレクトリの./.logとする
mkdir -p ./.log
LOGFILE=./.log/baseline-1gpu-$JOBID.out
ERRFILE=./.log/baseline-1gpu-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

# uvのキャッシュディレクトリを設定
export UV_CACHE_DIR="$HOME/.cache/uv"
mkdir -p "$UV_CACHE_DIR"

# モデルが存在しなければダウンロードして配置
echo "Check if model exists at $(pwd)/$MODEL_PATH"
if [ ! -d "$(pwd)/$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH. Downloading..."
    uv run python download_model.py --MODEL_NAME "$MODEL_NAME"
fi

# GPU環境の設定
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0
export VLLM_GPU_IDS=0
ulimit -n 65536 || true

# Pythonサンドボックス環境の起動
uv run python -m nemo_skills.code_execution.local_sandbox.local_sandbox_server \
    --port 6000 > sandbox.log 2>&1 &

# vLLMサーバーの起動
uv run python -m nemo_skills.inference.server.serve_vllm \
    --model HayatoHongoEveryonesAI/llm-jp-4-8b-instruct \
    --num_gpus 1 \
    --num_nodes 1 \
    --port 8000 \
    --enforce-eager > vllm.log 2>&1 &

# 推論動作の実行
uv run python tir-sc.py \
    --model_path HayatoHongoEveryonesAI/llm-jp-4-8b-instruct \
    --input_path input/dev.jsonl \
    --output_path dev-baseline.jsonl \
    --log_path inference_log_baseline.jsonl \
    --tir-llm-host 127.0.0.1 \
    --tir-llm-port 8000 \
    --tir-sandbox-host 127.0.0.1 \
    --tir-sandbox-port 6000 \
    --max-new-tokens 512 \
    --temperature 0.0 \
    --retry-temperature 0.4 \
    --repair-attempts 3 \
    --format-retry-attempts 5 \
    --direct-answer-attempts 2 \
    --log-raw-output \
    --enable-wandb \
    --wandb-project "miyako-personal/llm-jp-4-8b-instruct-tir-eval"
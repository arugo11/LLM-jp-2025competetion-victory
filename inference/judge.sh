#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N judge-1gpu
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=2:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

# qsub -v MODEL_REPO=open-instruct-grpo-fast,NAME=GRPO_MODEL judge.sh

: ${MODEL_USER:="HayatoHongoEveryonesAI"}
: ${MODEL_REPO:="llm-jp-4-8b-instruct-sft-long-v5"}
: ${MODEL_BASE_PATH:="models/"}
: ${WAIT:=0}
: ${NAME:="_"}
: ${T:=0.0}
: ${INPUT_FILE:="output/output-open-instruct-grpo-fast_WAIT0-math_all_samples.jsonl"}

MODEL_NAME="${MODEL_USER}/${MODEL_REPO}"
MODEL_PATH="$MODEL_BASE_PATH/$MODEL_NAME"

echo "Model name: $MODEL_NAME"
echo "Model path: $MODEL_PATH"

source /etc/profile.d/modules.sh
module load cuda/12.8

# ~/LLM-jp-2025competetion-victory/inferenceで実行する
cd $PBS_O_WORKDIR #実行したディレクトリに移動
echo "Current working directory is $(pwd)"

JOBID=${PBS_JOBID%%.*}
# ログの保存
# 保存先は実行ディレクトリの./.logとする
mkdir -p ./.log
LOGFILE=./.log/inference-1gpu-$JOBID.out
ERRFILE=./.log/inference-1gpu-$JOBID.err
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
    uv run python download_model.py --model_name "$MODEL_NAME"
fi

source .venv/bin/activate

python judge.py --model_path "$MODEL_PATH" --t "$T" --name "$NAME" --input_file "$INPUT_FILE"
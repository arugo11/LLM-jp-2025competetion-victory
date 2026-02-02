#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HF
#PBS -N inference-1node
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=8:20:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

# bash inference-eval-1node.sh

: ${MODEL_USER:="HayatoHongoEveryonesAI"}
: ${MODEL_REPO:="llm-jp-4-8b-instruct-sft-expand-checkpoint-1900"}
: ${MODEL_BASE_PATH:="models/"}
: ${WAIT:=1}
: ${NAME:="_"}
: ${T:=0.7}

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

# ~/LLM-jp-2025competetion-victory/inferenceで実行する
cd $PBS_O_WORKDIR #実行したディレクトリに移動
echo "Current working directory is $(pwd)"

JOBID=${PBS_JOBID%%.*}
# ログの保存
# 保存先は実行ディレクトリの./.logとする
mkdir -p ./.log
LOGFILE=./.log/inference-1node-$JOBID.out
ERRFILE=./.log/inference-1node-$JOBID.err
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

# Singularityイメージのビルド
echo "Build singularity image"
singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       --build-arg MODEL_NAMES="$MODEL_NAME" \
       dist/${MODEL_REPO}${NAME}-replace_pm.sif self-consistency.def

# 推論の実行
echo "Start inference"
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --net --network none dist/${MODEL_REPO}${NAME}-replace_pm.sif \
    --input_path input/dev_500.jsonl \
    --output_path "$(pwd)/output/output-${MODEL_REPO}${NAME}-cons40-replace_pm.jsonl" \
    --model_path $MODEL_PATH \
    --max_tokens 8192 \
    --num_samples 40 \
    --wait_count $WAIT \
    --temperature $T

# 推論結果の評価
echo "Start evaluation"
cd ..
cd math-eval
uv sync
source .venv/bin/activate
python src/math_eval/eval_consistency.py \
       ../inference/output/output-${MODEL_REPO}${NAME}-cons40-replace_pm_all_samples.jsonl \
       ./targets/dev.jsonl \
       -o ./accuracy/acc-${MODEL_REPO}${NAME}-dev.jsonl \
       -k "1,20,40"

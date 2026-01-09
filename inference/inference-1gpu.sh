#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N inference-1gpu
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=3:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

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

# Singularityイメージのビルド
echo "Build singularity image"
singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       --build-arg MODEL_NAMES="HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long-v5" \
       dist/sft-long-v5.sif self-consistency.def

# 推論の実行
echo "Start inference"
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 --net --network none dist/sft-long-v5.sif \
    --model_path models/HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long-v5 \
    --input_path input/dev.jsonl \
    --output_path "$(pwd)/output/output.jsonl" \
    --max_tokens 4096 \
    --num_samples 10 \
    --temperature 1.0
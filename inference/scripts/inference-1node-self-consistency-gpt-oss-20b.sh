#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HF
#PBS -N inference-1node-gpt-oss-20b
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=12:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

source /etc/profile.d/modules.sh
module load cuda/12.8

# ~/LLM-jp-2025competetion-victory/inferenceで実行する
cd $PBS_O_WORKDIR #実行したディレクトリに移動
echo "Current working directory is $(pwd)"

JOBID=${PBS_JOBID%%.*}
mkdir -p ./.log
LOGFILE=./.log/gpt-oss-20b-1node-$JOBID.out
ERRFILE=./.log/gpt-oss-20b-1node-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

# uvのキャッシュディレクトリを設定
export UV_CACHE_DIR="$HOME/workspace/LLM-jp-2025competetion-victory/.cache/uv"
mkdir -p "$UV_CACHE_DIR"


# GPT-OSS-20Bモデルのビルド
echo "Build llm-jp-20b-gpt-oss"
singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       --build-arg MODEL_NAMES="openai/gpt-oss-20b" \
       dist/gpt-oss-20b.sif self_consistency.def


# GPT-OSS-20Bでの推論
# 生成回数1
echo "Inference gpt-oss-20b num_samples=1"
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --net --network none dist/gpt-oss-20b.sif \
    --model_path models/openai/gpt-oss-20b \
    --input_path input/math500-ja.jsonl \
    --output_path "$(pwd)/output/math-500/gpt-oss-20b/sample-1.jsonl" \
    --num_samples 1
# 生成回数10
echo "Inference gpt-oss-20b num_samples=10"
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --net --network none dist/gpt-oss-20b.sif \
    --model_path models/openai/gpt-oss-20b \
    --input_path input/math500-ja.jsonl \
    --output_path "$(pwd)/output/math-500/gpt-oss-20b/sample-10.jsonl" \
    --num_samples 10
# 生成回数20
echo "Inference gpt-oss-20b num_samples=20"
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --net --network none dist/gpt-oss-20b.sif \
    --model_path models/openai/gpt-oss-20b \
    --input_path input/math500-ja.jsonl \
    --output_path "$(pwd)/output/math-500/gpt-oss-20b/sample-20.jsonl" \
    --num_samples 20
# 生成回数40
echo "Inference gpt-oss-20b num_samples=40"
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --net --network none dist/gpt-oss-20b.sif \
    --model_path models/openai/gpt-oss-20b \
    --input_path input/math500-ja.jsonl \
    --output_path "$(pwd)/output/math-500/gpt-oss-20b/sample-40.jsonl" \
    --num_samples 40
# 生成回数80
echo "Inference gpt-oss-20b num_samples=80"
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --net --network none dist/gpt-oss-20b.sif \
    --model_path models/openai/gpt-oss-20b \
    --input_path input/math500-ja.jsonl \
    --output_path "$(pwd)/output/math-500/gpt-oss-20b/sample-80.jsonl" \
    --num_samples 80
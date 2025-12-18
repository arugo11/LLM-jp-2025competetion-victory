#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N inference-1gpu
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=168:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

# ~/LLM-jp-2025competetion-victory/inferenceで実行する
cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
mkdir -p ./.log
LOGFILE=./.log/inference-$JOBID.out
ERRFILE=./.log/inference-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

mkdir -p output

# uvのキャッシュディレクトリを設定 (キャッシュはvscode内で見たいため、プロジェクト下とします。)
export UV_CACHE_DIR="$HOME/LLM-jp-2025competetion-victory/.cache/uv"
mkdir -p "$UV_CACHE_DIR"

# コードを書き替えるたびにビルドする必要があるそうです。
singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       --build-arg MODEL_NAMES="team-victory/llm-jp-4-8b-instruct" \
       dist/self_consistency.sif self_consistency.def

# 推論を実行します。
echo "Starting inference with self-consistency num-samples=1"
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 --net --network none \
    dist/self_consistency.sif \
    --model_path "$(pwd)/models/team-victory/llm-jp-4-8b-instruct" \
    --input_path "$(pwd)/input/dev.jsonl" \
    --output_path "$(pwd)/output/self-consistency-num-1.jsonl" \
    --max_tokens 4096 \
    --num_samples 1

echo "Starting inference with self-consistency num-samples=10"
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 --net --network none \
    dist/self_consistency.sif \
    --model_path "$(pwd)/models/team-victory/llm-jp-4-8b-instruct" \
    --input_path "$(pwd)/input/dev.jsonl" \
    --output_path "$(pwd)/output/self-consistency-num-10.jsonl" \
    --max_tokens 4096 \
    --num_samples 10

echo "Starting inference with self-consistency num-samples=20"
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 --net --network none \
    dist/self_consistency.sif \
    --model_path "$(pwd)/models/team-victory/llm-jp-4-8b-instruct" \
    --input_path "$(pwd)/input/dev.jsonl" \
    --output_path "$(pwd)/output/self-consistency-num-20.jsonl" \
    --max_tokens 4096 \
    --num_samples 20

echo "Starting inference with self-consistency num-samples=40"
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 --net --network none \
    dist/self_consistency.sif \
    --model_path "$(pwd)/models/team-victory/llm-jp-4-8b-instruct" \
    --input_path "$(pwd)/input/dev.jsonl" \
    --output_path "$(pwd)/output/self-consistency-num-40.jsonl" \
    --max_tokens 4096 \
    --num_samples 40

echo "Starting inference with self-consistency num-samples=80"
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 --net --network none \
    dist/self_consistency.sif \
    --model_path "$(pwd)/models/team-victory/llm-jp-4-8b-instruct" \
    --input_path "$(pwd)/input/dev.jsonl" \
    --output_path "$(pwd)/output/self-consistency-num-80.jsonl" \
    --max_tokens 4096 \
    --num_samples 80

echo "Inference with self-consistency completed."
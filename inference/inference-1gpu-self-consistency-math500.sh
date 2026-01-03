#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N inference-1gpu
#PBS -l select=1:ncpus=192:ngpus=1
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
LOGFILE=./.log/inference-math500-$JOBID.out
ERRFILE=./.log/inference-math500-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

# uvのキャッシュディレクトリを設定 (キャッシュはvscode内で見たいため、プロジェクト下とします。)
export UV_CACHE_DIR="$HOME/workspace/LLM-jp-2025competetion-victory/.cache/uv"
mkdir -p "$UV_CACHE_DIR"

# コードを書き替えるたびにビルドする必要があるそうです。
# llm-jp-8b-instructモデルでのビルド
echo "Build llm-jp-4-8b-instruct"
singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       --build-arg MODEL_NAMES="team-victory/llm-jp-4-8b-instruct" \
       dist/baseline-math500.sif self_consistency_math500.def

# SFTモデルでのビルド
echo "Build llm-jp-4-8b-instruct-sft-test-checkpoint-140"
singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       --build-arg MODEL_NAMES="team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140" \
       dist/sft-test-math500.sif self_consistency_math500.def


# llm-jp-4-instructionでの推論
# 生成回数1
echo "Inference llm-jp-4-8b-instruct num_samples=1"
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 --net --network none dist/baseline-math500.sif \
    --model_path models/team-victory/llm-jp-4-8b-instruct \
    --input_path input/math500-ja-2.jsonl \
    --output_path "$(pwd)/output/math-500/llm-jp-4-instruction/sample-1.jsonl" \
    --num_samples 1

# 生成回数10
echo "Inference llm-jp-4-8b-instruct num_samples=10"
singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0 \
        --net --network none dist/baseline-math500.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct \
        --input_path input/math500-ja-2.jsonl \
        --output_path "$(pwd)/output/math-500/llm-jp-4-instruction/sample-10.jsonl" \
        --num_samples 10
# 生成回数20
echo "Inference llm-jp-4-8b-instruct num_samples=20"
singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0 \
        --net --network none dist/baseline-math500.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct \
        --input_path input/math500-ja-2.jsonl \
        --output_path "$(pwd)/output/math-500/llm-jp-4-instruction/sample-20.jsonl" \
        --num_samples 20

# 生成回数40
echo "Inference llm-jp-4-8b-instruct num_samples=40"
singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0 \
        --net --network none dist/baseline-math500.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct \
        --input_path input/math500-ja-2.jsonl \
        --output_path "$(pwd)/output/math-500/llm-jp-4-instruction/sample-40.jsonl" \
        --num_samples 40

# 生成回数80
echo "Inference llm-jp-4-8b-instruct num_samples=80"
singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0 \
        --net --network none dist/baseline-math500.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct \
        --input_path input/math500-ja-2.jsonl \
        --output_path "$(pwd)/output/math-500/llm-jp-4-instruction/sample-80.jsonl" \
        --num_samples 80

# llm-jp-4-8b-instruct-sft-test-checkpoint-140での推論
# 生成回数1
echo "Inference llm-jp-4-8b-instruct-sft-test-checkpoint-140 num_samples=1"
singularity run --nv --writable-tmpfs \
        --env CUDA_VISIBLE_DEVICES=0 --net --network none dist/sft-test-math500.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140 \
        --input_path input/math500-ja-2.jsonl \
        --output_path "$(pwd)/output/math-500/sft-test1/sample-1.jsonl" \
        --num_samples 1
# 生成回数10
echo "Inference llm-jp-4-8b-instruct-sft-test-checkpoint-140 num_samples=10"
singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0 \
        --net --network none dist/sft-test-math500.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140 \
        --input_path input/math500-ja-2.jsonl \
        --output_path "$(pwd)/output/math-500/sft-test1/sample-10.jsonl" \
        --num_samples 10

# 生成回数20
echo "Inference llm-jp-4-8b-instruct-sft-test-checkpoint-140 num_samples=20"
singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0 \
        --net --network none dist/sft-test-math500.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140 \
        --input_path input/math500-ja-2.jsonl \
        --output_path "$(pwd)/output/math-500/sft-test1/sample-20.jsonl" \
        --num_samples 20

# 生成回数40
echo "Inference llm-jp-4-8b-instruct-sft-test-checkpoint-140 num_samples=40"
singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0 \
        --net --network none dist/sft-test-math500.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140 \
        --input_path input/math500-ja-2.jsonl \
        --output_path "$(pwd)/output/math-500/sft-test1/sample-40.jsonl" \
        --num_samples 40

# 生成回数80
echo "Inference llm-jp-4-8b-instruct-sft-test-checkpoint-140 num_samples=80"
singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0 \
        --net --network none dist/sft-test-math500.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140 \
        --input_path input/math500-ja-2.jsonl \
        --output_path "$(pwd)/output/math-500/sft-test1/sample-80.jsonl" \
        --num_samples 80
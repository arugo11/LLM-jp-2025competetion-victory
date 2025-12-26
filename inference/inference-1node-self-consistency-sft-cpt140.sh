#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HF
#PBS -N inference-1node-sft
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=3:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

# 温度定数の設定
export temperature=1.0

source /etc/profile.d/modules.sh
module load cuda/12.8

# ~/LLM-jp-2025competetion-victory/inferenceで実行する
cd $PBS_O_WORKDIR #実行したディレクトリに移動
echo "Current working directory is $(pwd)"

JOBID=${PBS_JOBID%%.*}
mkdir -p ./.log
LOGFILE=./.log/sft-cpt140-1node$JOBID.out
ERRFILE=./.log/sft-cpt140-1node$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

# uvのキャッシュディレクトリを設定
export UV_CACHE_DIR="$HOME/workspace/LLM-jp-2025competetion-victory/.cache/uv"
mkdir -p "$UV_CACHE_DIR"


# SFTモデルでのビルド
echo "Build llm-jp-4-8b-instruct-sft-test-checkpoint-140"
singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       --build-arg MODEL_NAMES="team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140" \
       dist/sft-test.sif self_consistency.def


# llm-jp-4-8b-instruct-sft-test-checkpoint-140での推論
# 生成回数1
echo "Inference llm-jp-4-8b-instruct-sft-test-checkpoint-140 num_samples=1"
singularity run --nv --writable-tmpfs \
        --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --net --network none dist/sft-test.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140 \
        --input_path input/dev.jsonl \
        --output_path "$(pwd)/output/dev-json/temp1/sft-test1/sample-1.jsonl" \
        --num_samples 1 \
        --temperature $temperature
# 生成回数10
echo "Inference llm-jp-4-8b-instruct-sft-test-checkpoint-140 num_samples=10"
singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        --net --network none dist/sft-test.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140 \
        --input_path input/dev.jsonl \
        --output_path "$(pwd)/output/dev-json/temp1/sft-test1/sample-10.jsonl" \
        --num_samples 10 \
        --temperature $temperature
# 生成回数20
echo "Inference llm-jp-4-8b-instruct-sft-test-checkpoint-140 num_samples=20"
singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        --net --network none dist/sft-test.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140 \
        --input_path input/dev.jsonl \
        --output_path "$(pwd)/output/dev-json/temp1/sft-test1/sample-20.jsonl" \
        --num_samples 20 \
        --temperature $temperature
# 生成回数40
echo "Inference llm-jp-4-8b-instruct-sft-test-checkpoint-140 num_samples=40"
singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        --net --network none dist/sft-test.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140 \
        --input_path input/dev.jsonl \
        --output_path "$(pwd)/output/dev-json/temp1/sft-test1/sample-40.jsonl" \
        --num_samples 40 \
        --temperature $temperature
# 生成回数80
echo "Inference llm-jp-4-8b-instruct-sft-test-checkpoint-140 num_samples=80"
singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        --net --network none dist/sft-test.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140 \
        --input_path input/dev.jsonl \
        --output_path "$(pwd)/output/dev-json/temp1/sft-test1/sample-80.jsonl" \
        --num_samples 80 \
        --temperature $temperature
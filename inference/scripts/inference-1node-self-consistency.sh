#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HF
#PBS -N inference-1gpu
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
LOGFILE=./.log/inference-$JOBID.out
ERRFILE=./.log/inference-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

# uvのキャッシュディレクトリを設定 (キャッシュはvscode内で見たいため、プロジェクト下とします。)
export UV_CACHE_DIR="$HOME/workspace/LLM-jp-2025competetion-victory/.cache/uv"
mkdir -p "$UV_CACHE_DIR"

# SFTモデルでのビルド
echo "Build llm-jp-4-8b-instruct-sft-test-checkpoint-140"
singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       --build-arg MODEL_NAMES="team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140" \
       dist/sft-test-1node.sif self_consistency.def

# 生成回数40
echo "Inference llm-jp-4-8b-instruct-sft-test-checkpoint-140 num_samples=40"
singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        --net --network none dist/sft-test-1node.sif \
        --model_path models/team-victory/llm-jp-4-8b-instruct-sft-test-checkpoint-140 \
        --input_path input/dev.jsonl \
        --output_path "$(pwd)/output/sft-test1-1node-sample-40.jsonl" \
        --max_tokens 4096 \
        --num_samples 40
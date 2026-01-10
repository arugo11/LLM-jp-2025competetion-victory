#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N create_qa-1gpu
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=12:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

# cd inference
# qsub -v HF_TOKEN="your_token_here" ./create_qa-1gpu.sh
cd $PBS_O_WORKDIR #実行したディレクトリに移動

NUM_QUESTIONS=${NUM_QUESTIONS:-10000}
REPO_ID=${REPO_ID:-"team-victory/qa_default"}
MODEL_PATH=${MODEL_PATH:-"models/openai/gpt-oss-20b"}

echo "NUM_QUESTIONS is set to ${NUM_QUESTIONS}"
echo "REPO_ID is set to ${REPO_ID}"

JOBID=${PBS_JOBID%%.*}
mkdir -p ./.log
LOGFILE=./.log/llmjudge_tir-$JOBID.out
ERRFILE=./.log/llmjudge_tir-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

mkdir -p output

# uvのキャッシュディレクトリを設定 (キャッシュはvscode内で見たいため、プロジェクト下とします。)
export UV_CACHE_DIR="$HOME/LLM-jp-2025competetion-victory/.cache/uv"
mkdir -p "$UV_CACHE_DIR"
mkdir -p dist

# コードを書き替えるたびにビルドする必要があるそうです。
singularity build --fakeroot --force \
       dist/llmjudge_tir.sif llmjudge_tir.def

mkdir -p $(pwd)/sandbox_tmp
export NEMO_SKILLS_SANDBOX_TMPDIR="/app/sandbox_tmp"

# 推論を実行します。
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0\
    --env HF_TOKEN=$HF_TOKEN \
    --bind "$(pwd)/models:/app/models" \
    --bind "$(pwd)/output:/app/output" \
    dist/llmjudge_tir.sif \
    --model_path "$MODEL_PATH" \
    --max_tokens 4096 \
    --repo_id $REPO_ID \
    --hf_token $HF_TOKEN \
    --output_jsonl output/test_qa.jsonl \
    --num_questions $NUM_QUESTIONS 
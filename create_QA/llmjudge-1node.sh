#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N create_qa-fullnode
#PBS -l select=1:ncpus=40:ngpus=8
#PBS -l walltime=6:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

# 注意:
# rt_HG や rt_AF (A100) の場合、通常 ngpus=4 または 8 です。
# V100ノードの場合: select=1:ncpus=40:ngpus=4
# A100ノードの場合: select=1:ncpus=72:ngpus=4 (または8)
# 環境に合わせて ncpus, ngpus を調整してください。

cd $PBS_O_WORKDIR

NUM_QUESTIONS=${NUM_QUESTIONS:-50000}
REPO_ID=${REPO_ID:-"team-victory/qa_fullnode"}
MODEL_PATH=${MODEL_PATH:-"models/openai/gpt-oss-20b"}

# Tensor Parallelismのサイズ。
# 20BモデルならA100(40GB)ならtp=1でOK。V100(32GB)ならtp=2推奨。
# ここではデータ並列(4プロセス)にするため1を設定しますが、OOMが出るなら2や4にしてください。
TP_SIZE=1 

echo "NUM_QUESTIONS: ${NUM_QUESTIONS}"
echo "REPO_ID: ${REPO_ID}"
echo "TP_SIZE: ${TP_SIZE}"

JOBID=${PBS_JOBID%%.*}
mkdir -p ./.log
LOGFILE=./.log/llmjudge-$JOBID.out
ERRFILE=./.log/llmjudge-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

mkdir -p output
export UV_CACHE_DIR="$HOME/LLM-jp-2025competetion-victory/.cache/uv"
mkdir -p "$UV_CACHE_DIR"
mkdir -p dist

# ビルド (変更がある場合のみ)
singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       dist/llmjudge.sif llmjudge.def

# 推論実行
# CUDA_VISIBLE_DEVICESは設定せず、Python側で制御させます
singularity run --nv --writable-tmpfs \
    --env HF_TOKEN=$HF_TOKEN \
    --bind "$(pwd)/models:/app/models" \
    --bind "$(pwd)/output:/app/output" \
    dist/llmjudge.sif \
    --model_path "$MODEL_PATH" \
    --max_tokens 4096 \
    --repo_id $REPO_ID \
    --hf_token $HF_TOKEN \
    --output_jsonl output/fullnode_qa.jsonl \
    --num_questions $NUM_QUESTIONS \
    --tp_size $TP_SIZE
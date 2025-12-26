#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HF
#PBS -N create_qa-fullnode
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=24:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

cd $PBS_O_WORKDIR

# --- 設定値 ---
NUM_QUESTIONS=${NUM_QUESTIONS:-50000}
REPO_ID=${REPO_ID:-"HayatoHongoEveryonesAI/qa_fullnode"}
MODEL_PATH=${MODEL_PATH:-"models/openai/gpt-oss-20b"}
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

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export VLLM_USE_RAY=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 追加: ホスト側のGPU状態確認 ---
echo "=== Host GPU Check ==="
echo "Host CUDA_VISIBLE_DEVICES (Before): ${CUDA_VISIBLE_DEVICES:-'Not Set'}"

# ★★★ ここが最重要修正 ★★★
# ホスト側で勝手にかかっている "1枚制限" をここで解除します
unset CUDA_VISIBLE_DEVICES

echo "Host CUDA_VISIBLE_DEVICES (After unset): ${CUDA_VISIBLE_DEVICES:-'CLEARED'}"
nvidia-smi -L

# ビルド
singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       dist/llmjudge.sif llmjudge.def

# --- 修正: CUDA_VISIBLE_DEVICES を 0-7 で強制上書き ---
singularity run --nv --writable-tmpfs \
    --env HF_TOKEN=$HF_TOKEN \
    --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
    --env MKL_NUM_THREADS=$MKL_NUM_THREADS \
    --env PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF \
    --env VLLM_USE_RAY=$VLLM_USE_RAY \
    --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
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
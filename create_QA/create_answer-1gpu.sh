#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N create_answer-1gpu
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=168:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

# cd create_QA
# qsub -v HF_TOKEN="your_token_here" ./create_answer-1gpu.sh
cd $PBS_O_WORKDIR #実行したディレクトリに移動

JOBID=${PBS_JOBID%%.*}
mkdir -p ./.log
LOGFILE=./.log/create_answer-$JOBID.out
ERRFILE=./.log/create_answer-$JOBID.err
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
       dist/create_qa.sif create_qa.def

# 推論を実行します。
REPO_ID="HayatoHongoEveryonesAI/qa_verify_1node_test8"
MODEL_PATH="models/openai/gpt-oss-120b"
TIR_MODEL_NAME="gpt-oss-120b"
OUTPUT_JSONL="output/qa_verify_1node_test8.jsonl"
echo "INPUT_REPO_ID=${REPO_ID}"
echo "OUTPUT_REPO_ID=${REPO_ID}-TIR"

singularity exec --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0\
    --env HF_TOKEN=$HF_TOKEN \
    --bind "$(pwd)/models:/app/models" \
    --bind "$(pwd)/output:/app/output" \
    dist/create_qa.sif \
    python3 create_answers.py \
    --sandbox-block-network \
    --model_path "$MODEL_PATH" \
    --max_tokens 4096 \
    --repo_id $REPO_ID \
    --hf_token $HF_TOKEN \
    --tir-model-name "$TIR_MODEL_NAME" \
    --output_jsonl "$OUTPUT_JSONL" \
    --tir-endpoint-type responses

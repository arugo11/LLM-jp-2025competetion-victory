#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N create_answer-1gpu-array
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=168:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -J 1-10

# qsub -v HF_TOKEN="your_token_here" /path/to/LLM-jp-2025competetion-victory/create_QA/create_answer-1gpu-array.sh
PROJECT_ROOT="${PROJECT_ROOT:-${PBS_O_WORKDIR:-/home/ach18380vf/LLM-jp-2025competetion-victory}}"
WORKDIR="$PROJECT_ROOT/create_QA"
if [ ! -d "$WORKDIR" ]; then
  echo "ERROR: WORKDIR not found: $WORKDIR"
  echo "Set PROJECT_ROOT or submit from the project root so PBS_O_WORKDIR is correct."
  exit 1
fi
cd "$WORKDIR"

JOBID=${PBS_JOBID%%.*}
ARRAY_RAW_ID=${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-1}}
ARRAY_ID=$((ARRAY_RAW_ID - 1))
mkdir -p ./.log
LOGFILE=./.log/create_answer-array-$JOBID-$ARRAY_ID.out
ERRFILE=./.log/create_answer-array-$JOBID-$ARRAY_ID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"
echo "ARRAY_RAW_ID=${ARRAY_RAW_ID}"
echo "ARRAY_ID=${ARRAY_ID}"

set -euxo pipefail

mkdir -p output

# 125kデータを10分割 (1ジョブあたり12,500件)
TOTAL=${TOTAL:-125000}
PARTS=10
CHUNK=$(( (TOTAL + PARTS - 1) / PARTS ))
START=$(( ARRAY_ID * CHUNK ))
END=$(( START + CHUNK ))
if [ $END -gt $TOTAL ]; then END=$TOTAL; fi
if [ $START -ge $TOTAL ]; then
  echo "No work for ARRAY_ID=$ARRAY_ID (START=$START >= TOTAL=$TOTAL)"
  exit 0
fi

OUTPUT_DIR="output/qa_verify_125k6_part${ARRAY_ID}"
mkdir -p "$OUTPUT_DIR"
INPUT_JSONL_HOST="${OUTPUT_DIR}/input.jsonl"
OUTPUT_JSONL_HOST="${OUTPUT_DIR}/output.jsonl"
INPUT_JSONL_IN="/app/${INPUT_JSONL_HOST}"
OUTPUT_JSONL_IN="/app/${OUTPUT_JSONL_HOST}"
PROGRESS_EVERY=${PROGRESS_EVERY:-500}

echo "DATASET=HayatoHongoEveryonesAI/qa_verify_125k6"
echo "RANGE=${START}:${END}"
echo "INPUT_JSONL=${INPUT_JSONL_HOST}"
echo "OUTPUT_JSONL=${OUTPUT_JSONL_HOST}"

# uvのキャッシュディレクトリを設定 (キャッシュはvscode内で見たいため、プロジェクト下とします。)
export UV_CACHE_DIR="$HOME/LLM-jp-2025competetion-victory/.cache/uv"
mkdir -p "$UV_CACHE_DIR"

# SIF が無い/古いときのみビルド
if [ ! -f dist/create_qa.sif ] || [ create_qa.def -nt dist/create_qa.sif ]; then
  singularity build --fakeroot --force \
         --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
         dist/create_qa.sif create_qa.def
fi

# データセットの該当スライスをJSONLへ保存（1回だけ作成）
if [ ! -s "$INPUT_JSONL_HOST" ]; then
  singularity exec --nv --writable-tmpfs \
      --env CUDA_VISIBLE_DEVICES=0 \
      --env HF_TOKEN=$HF_TOKEN \
      --env START=$START \
      --env END=$END \
      --env INPUT_JSONL=$INPUT_JSONL_IN \
      --bind "$(pwd):/app/work" \
      --bind "$(pwd)/output:/app/output" \
      dist/create_qa.sif \
      python3 /app/work/slice_dataset.py \
      --repo-id "HayatoHongoEveryonesAI/qa_verify_125k6" \
      --start "$START" \
      --end "$END" \
      --out "$INPUT_JSONL_IN"
fi

# 推論を実行します。
MODEL_PATH="models/openai/gpt-oss-120b"
TIR_MODEL_NAME="gpt-oss-120b"

singularity exec --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env HF_TOKEN=$HF_TOKEN \
    --bind "$(pwd):/app/work" \
    --bind "$(pwd)/models:/app/models" \
    --bind "$(pwd)/output:/app/output" \
    dist/create_qa.sif \
    python3 /app/work/create_answers.py \
    --sandbox-block-network \
    --model_path "$MODEL_PATH" \
    --max_tokens 4096 \
    --input_jsonl "$INPUT_JSONL_IN" \
    --output_jsonl "$OUTPUT_JSONL_IN" \
    --tir-model-name "$TIR_MODEL_NAME" \
    --tir-endpoint-type responses \
    --progress-every "$PROGRESS_EVERY"

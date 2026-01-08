#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N merge_push_qa_verify_125k6
#PBS -l select=1:ncpus=16:ngpus=0
#PBS -l walltime=04:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

# qsub -v HF_TOKEN="your_token_here" /path/to/LLM-jp-2025competetion-victory/create_QA/merge_and_push_qa_verify_125k6.sh
PROJECT_ROOT="${PROJECT_ROOT:-${PBS_O_WORKDIR:-/home/ach18380vf/LLM-jp-2025competetion-victory}}"
WORKDIR="$PROJECT_ROOT/create_QA"
if [ ! -d "$WORKDIR" ]; then
  echo "ERROR: WORKDIR not found: $WORKDIR"
  echo "Set PROJECT_ROOT or submit from the project root so PBS_O_WORKDIR is correct."
  exit 1
fi
cd "$WORKDIR"

JOBID=${PBS_JOBID%%.*}
mkdir -p ./.log
LOGFILE=./.log/merge_push-$JOBID.out
ERRFILE=./.log/merge_push-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

MERGED_JSONL="output/qa_verify_125k6_merged.jsonl"
MANIFEST_JSON="output/qa_verify_125k6_merged.manifest.json"
OUT_REPO_ID="${OUT_REPO_ID:-HayatoHongoEveryonesAI/qa_verify_125k6-TIR}"

# merge (keep partial output if validation fails)
python3 merge_jsonl_parts.py \
  --pattern "output/qa_verify_125k6_part{index}/output.jsonl" \
  --parts 10 \
  --expected-total 125000 \
  --strict-json \
  --keep-partial \
  --out "$MERGED_JSONL" \
  --manifest "$MANIFEST_JSON"

# push to HF (only if merge succeeded and full file exists)
if [ -s "$MERGED_JSONL" ]; then
  singularity exec --nv --writable-tmpfs \
      --env HF_TOKEN=$HF_TOKEN \
      --bind "$(pwd):/app/work" \
      --bind "$(pwd)/output:/app/output" \
      dist/create_qa.sif \
      python3 /app/work/push_dataset.py \
      --jsonl "/app/$MERGED_JSONL" \
      --repo-id "$OUT_REPO_ID"
else
  echo "ERROR: merged JSONL not found or empty: $MERGED_JSONL"
  exit 1
fi

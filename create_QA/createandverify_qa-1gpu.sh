#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N create_qa-1gpu
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=24:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

# cd inference
# qsub -v HF_TOKEN="your_token_here" ./create_qa-1gpu.sh
cd $PBS_O_WORKDIR #実行したディレクトリに移動

JOBID=${PBS_JOBID%%.*}
mkdir -p ./.log
LOGFILE=./.log/createandverify_qa-$JOBID.out
ERRFILE=./.log/createandverify_qa-$JOBID.err
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
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       dist/createandverify_qa.sif createandverify_qa.def

# 推論を実行します。
REPO_ID="team-victory/qa_verify_5k_test"

singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0\
    --env HF_TOKEN=$HF_TOKEN \
    --bind "$(pwd)/models:/app/models" \
    --bind "$(pwd)/output:/app/output" \
    dist/createandverify_qa.sif \
    --model_path models/openai/gpt-oss-20b \
    --max_tokens 4096 \
    --repo_id $REPO_ID \
    --hf_token $HF_TOKEN \
    --output_jsonl output/test_qa.jsonl
#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N inference-1gpu
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=168:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

# ~/LLM-jp-2025competetion-victory/inferenceで実行する
cd $PBS_O_WORKDIR #実行したディレクトリに移動

JOBID=${PBS_JOBID%%.*}
mkdir -p ./.log
LOGFILE=./.log/inference-$JOBID.out
ERRFILE=./.log/inference-$JOBID.err
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
       --build-arg MODEL_NAMES="team-victory/llm-jp-4-8b-instruct" \
       dist/submission.sif submission.def

# 推論を実行します。
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 --net --network none \
    dist/submission.sif \
    --model_path models/team-victory/llm-jp-4-8b-instruct \
    --input_path "$(pwd)/input/dev.jsonl" \
    --output_path "$(pwd)/output/output_4_1_instruction.jsonl" \
    --max_tokens 1024
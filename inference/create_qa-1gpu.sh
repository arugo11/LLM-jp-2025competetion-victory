#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N create_qa-1gpu
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=168:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

# cd inference
# qsub ./create_qa-1gpu.sh
# ~/LLM-jp-2025competetion-victory/inferenceで実行する
cd $PBS_O_WORKDIR #実行したディレクトリに移動

JOBID=${PBS_JOBID%%.*}
mkdir -p ./.log
LOGFILE=./.log/create_qa-$JOBID.out
ERRFILE=./.log/create_qa-$JOBID.err
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
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 --net --network none \
    --bind "$(pwd)/models:/app/models" \
    dist/create_qa.sif \
    --model_path models/openai/gpt-oss-20b \
    --input_path "$(pwd)/input/dev.jsonl" \
    --output_path "$(pwd)/output/create_qa_gpt-oss-20b.jsonl" \
    --max_tokens 1024
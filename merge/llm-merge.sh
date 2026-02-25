#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N llm-merge
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=8:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

# Setup logs
cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
mkdir -p ./.log
LOGFILE=./.log/train-$JOBID.out
ERRFILE=./.log/train-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

# --- 環境設定 (PBS用) ---
# PBS_NODEFILE の先頭をマスターノードのアドレスとして設定
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
echo "MASTER_ADDR: $MASTER_ADDR"

# ポート番号
export MASTER_PORT=29500
echo "MASTER_PORT: $MASTER_PORT"

echo "Current Directory: $(pwd)"

# --- モジュールとvenv ---
module load cuda/12.8           # 環境に合わせてバージョンを確認してください
export LD_LIBRARY_PATH=/apps/python/3.12.9/lib:$LD_LIBRARY_PATH
uv sync
source .venv/bin/activate         # venvを有効化

python llm-merge.py
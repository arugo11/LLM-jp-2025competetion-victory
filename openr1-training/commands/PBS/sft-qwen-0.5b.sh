#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HF
#PBS -N sft-0.5b
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=168:00:00
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
source env/bin/activate         # venvを有効化

cd open-r1/src || exit 1

# ulimit -v unlimited
# ulimit -m unlimited

accelerate launch \
    --config_file ../recipes/accelerate_configs/zero3.yaml \
    --num_machines 1 \
    --num_processes 8 \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --rdzv_backend c10d \
    open_r1/sft.py \
    --config ../../configs/Qwen2.5-Distill-0.5b-test/sft/config_test.yaml \
    --dataconfig ../../configs/data_configs/example.yaml

# 実行方法
# openr1-trainingで実行する。cd open-r1/srcが出来るように

# 実行コマンド
# qsub ./commands/PBS/sft-qwen-0.5b.sh
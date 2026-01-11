#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HF
#PBS -N sft-qa_verify
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=168:00:00
#PBS -m n
#PBS -o /dev/null
#PBS -e /dev/null

cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
mkdir -p ./.log
LOGFILE=./.log/train-$JOBID.out
ERRFILE=./.log/train-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Current Directory: $(pwd)"

module load cuda/12.8
export LD_LIBRARY_PATH=/apps/python/3.12.9/lib:$LD_LIBRARY_PATH
source env/bin/activate

cd open-r1/src || exit 1

accelerate launch \
  --config_file ../recipes/accelerate_configs/zero3_1node8.yaml \
  --num_machines 1 \
  --num_processes 8 \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  --rdzv_backend c10d \
  open_r1/sft.py \
  --config ../../configs/llmjp4-8B/sft/config_qa_verify_clean_fullft.yaml \
  --dataconfig ../../configs/data_configs/qa_verify_1node_test8_TIR_clean_raw_generation.yaml

# Run from repo root (openr1-training):
# qsub ./commands/PBS/sft-llmjp4-8b-qa_verify-clean-rawgen-fullft.sh

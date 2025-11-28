#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HC
#PBS -N download_hf_model
#PBS -l select=1:ncpus=96:ngpus=1
#PBS -l walltime=5:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -m n

TIMESTAMP=$(date +%Y%m%d%H%M%S)
JOBID=${PBS_JOBID%%.*}
mkdir -p logs
LOGFILE=logs/download_hf_model-$JOBID.out
ERRFILE=logs/download_hf_model-$JOBID.err
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

source ~/LLM-jp-2025competetion-victory/env/venv/bin/activate

huggingface-cli download team-victory/llm-jp-4-8b-instruct \
  --local-dir ~/LLM-jp-2025competetion-victory/models/team-victory/llm-jp-4-8b-instruct \
  --local-dir-use-symlinks False
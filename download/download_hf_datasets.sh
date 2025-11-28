#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HC
#PBS -N download_hf_datasets
#PBS -l select=1
#PBS -l walltime=5:00:00
#PBS -o logs/download_hf_datasets.out
#PBS -e logs/download_hf_datasets.err
#PBS -m n

TIMESTAMP=$(date +%Y%m%d%H%M%S)
JOBID=${PBS_JOBID%%.*}
mkdir -p logs
LOGFILE=logs/download_hf_datasets-$JOBID.out
ERRFILE=logs/download_hf_datasets-$JOBID.err
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

source ~/LLM-jp-2025competetion-victory/env/venv/bin/activate

python ~/LLM-jp-2025competetion-victory/download/download_hf_datasets.py
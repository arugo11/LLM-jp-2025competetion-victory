#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HC
#PBS -N download_hf_model
#PBS -l select=1:ncpus=96:ngpus=1
#PBS -l walltime=5:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -m n

source env/venv/bin/activate

huggingface-cli download team-victory/llm-jp-4-8b-instruct \
  --local-dir models/team-victory/llm-jp-4-8b-instruct \
  --local-dir-use-symlinks False
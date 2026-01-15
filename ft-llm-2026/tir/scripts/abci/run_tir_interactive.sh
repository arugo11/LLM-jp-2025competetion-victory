#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -P gch51701
#PBS -l walltime=2:00:00

set -eux

# ジョブ実行時には $0 が /var/spool/pbs/mom_priv/jobs/... になるため、絶対パスで指定する
source "$HOME/workspace/LLM-jp-2025competetion-victory/ft-llm-2026/tir/scripts/abci/environment.sh"

singularity exec --nv \
  --env CUDA_VISIBLE_DEVICES=${SINGULARITYENV_CUDA_VISIBLE_DEVICES} \
  --env NVIDIA_VISIBLE_DEVICES=${SINGULARITYENV_NVIDIA_VISIBLE_DEVICES} \
  --env VLLM_GPU_IDS=${SINGULARITYENV_VLLM_GPU_IDS} \
  docker://nvcr.io/nvidia/pytorch:24.08-py3 bash -lc "
set -eux
cd \"\$PROJECT_ROOT\"

pip install --user uv

# uvを認識させるためにPATHを通す
export PATH="$HOME/.local/bin:$PATH"

# Generate or refresh the lock file with the container's Python (3.10 in 24.08 image)
uv lock --python 3.10
uv sync --python 3.10

uv run python scripts/apply_patches.py

# ulimitはABCI環境では設定できず、ジョブが途中でハングするのでコメントアウトで無効化する
# ulimit -n 65536 || true

nohup uv run python -m nemo_skills.code_execution.local_sandbox.local_sandbox_server --port 6000 > sandbox.log 2>&1 &
nohup uv run python -m nemo_skills.inference.server.serve_vllm --model meta-llama/Meta-Llama-3.1-8B-Instruct --num_gpus 1 --num_nodes 1 --port 8000 --enforce-eager > vllm.log 2>&1 &

sleep 60
uv run python tir_demo.py
"

#!/bin/bash
set -eux

module load singularitypro/4.1.7

export SINGULARITY_TMPDIR="${PBS_LOCALDIR:-/tmp}"

PROJECT_ROOT="$HOME/LLM-jp-2025competetion-victory/ft-llm-2026/tir"
export PROJECT_ROOT

# Force numeric GPU IDs both outside and inside the container.
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0
export VLLM_GPU_IDS=0
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
export SINGULARITYENV_NVIDIA_VISIBLE_DEVICES=0
export SINGULARITYENV_VLLM_GPU_IDS=0

export PATH="$HOME/.local/bin:$PATH"

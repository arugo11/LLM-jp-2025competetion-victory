#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N grpo_fast_1gpu
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=2:00:00
#PBS -m n

# Setup logs
cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
mkdir -p ./logs
LOGFILE=./logs/grpo_fast-$JOBID.out
ERRFILE=./logs/grpo_fast-$JOBID.err
exec > $LOGFILE 2> $ERRFILE

echo "JOBID=${JOBID}"
echo "Start time: $(date)"

set -euxo pipefail

# Create Triton autotune cache directory to silence warnings
mkdir -p "$HOME/.triton/autotune"

# Setup environment
source scripts/abci/common/setup.sh
echo "ENV_DIR=${ENV_DIR}"

# ========== 仮想環境構築状況の確認 ==========
echo "========== 仮想環境構築状況の確認 =========="
VENV_DIR="${ENV_DIR}/venv"
VENV_BIN="${VENV_DIR}/bin"
VENV_PYTHON="${VENV_BIN}/python"
VENV_PYTHON3="${VENV_BIN}/python3"
VENV_SITE_PACKAGES="${VENV_DIR}/lib/python3.12/site-packages"

echo "ENV_DIR=${ENV_DIR}"
echo "VENV_DIR=${VENV_DIR}"

# 仮想環境ディレクトリの存在確認
echo "--- 仮想環境ディレクトリの存在確認 ---"
if [ -d "${VENV_DIR}" ]; then
    echo "✅ 仮想環境ディレクトリが存在します: ${VENV_DIR}"
    echo "   ディレクトリサイズ: $(du -sh ${VENV_DIR} 2>/dev/null | cut -f1 || echo 'N/A')"
else
    echo "❌ 仮想環境ディレクトリが存在しません: ${VENV_DIR}"
    echo "   仮想環境の構築が必要です"
    exit 1
fi

# Python実行ファイルの存在確認
echo "--- Python実行ファイルの存在確認 ---"
if [ -f "${VENV_PYTHON}" ]; then
    echo "✅ python実行ファイルが存在します: ${VENV_PYTHON}"
    echo "   ファイルサイズ: $(ls -lh ${VENV_PYTHON} 2>/dev/null | awk '{print $5}' || echo 'N/A')"
    echo "   実行可能: $([ -x "${VENV_PYTHON}" ] && echo 'YES' || echo 'NO')"
else
    echo "❌ python実行ファイルが存在しません: ${VENV_PYTHON}"
fi

if [ -f "${VENV_PYTHON3}" ]; then
    echo "✅ python3実行ファイルが存在します: ${VENV_PYTHON3}"
else
    echo "⚠️  python3実行ファイルが存在しません: ${VENV_PYTHON3}"
fi

# 仮想環境のPythonバージョン確認
echo "--- 仮想環境のPythonバージョン確認 ---"
if [ -f "${VENV_PYTHON}" ]; then
    ${VENV_PYTHON} --version 2>&1 || echo "❌ Pythonバージョンの取得に失敗"
    ${VENV_PYTHON} -c "import sys; print(f'Python executable: {sys.executable}'); print(f'Python version: {sys.version}')" 2>&1 || echo "❌ Python情報の取得に失敗"
else
    echo "❌ Python実行ファイルが見つからないため、バージョン確認をスキップ"
fi

# site-packagesディレクトリの確認
echo "--- site-packagesディレクトリの確認 ---"
if [ -d "${VENV_SITE_PACKAGES}" ]; then
    echo "✅ site-packagesディレクトリが存在します: ${VENV_SITE_PACKAGES}"
    PKG_COUNT=$(ls -1 ${VENV_SITE_PACKAGES} 2>/dev/null | wc -l)
    echo "   インストール済みパッケージ数: ${PKG_COUNT}"
    echo "   ディレクトリサイズ: $(du -sh ${VENV_SITE_PACKAGES} 2>/dev/null | cut -f1 || echo 'N/A')"
else
    echo "❌ site-packagesディレクトリが存在しません: ${VENV_SITE_PACKAGES}"
    echo "   仮想環境が正しく構築されていない可能性があります"
fi

# 主要パッケージのインストール状況確認
echo "--- 主要パッケージのインストール状況確認 ---"
if [ -f "${VENV_PYTHON}" ]; then
    ${VENV_PYTHON} << 'PYEOF'
import sys
import importlib.util
import pkg_resources

# パッケージの存在とバージョンを確認
packages_to_check = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'vllm': 'vLLM',
    'ray': 'Ray',
    'deepspeed': 'DeepSpeed',
    'datasets': 'Datasets',
    'numpy': 'NumPy',
    'accelerate': 'Accelerate',
}

print("パッケージ名          | インストール済み | バージョン        | パス")
print("-" * 80)

for pkg_name, pkg_display in packages_to_check.items():
    try:
        # パッケージの存在確認
        spec = importlib.util.find_spec(pkg_name)
        if spec and spec.origin:
            # バージョン取得を試みる
            try:
                dist = pkg_resources.get_distribution(pkg_name)
                version = dist.version
            except:
                version = "N/A"
            
            # 仮想環境のパスが含まれているか確認
            venv_path = '/LLM-jp-2025competition-victory/env/venv'
            is_in_venv = venv_path in spec.origin
            status = "✅ YES" if is_in_venv else "⚠️  OTHER"
            
            # パスの短縮表示
            origin_short = spec.origin[:50] + "..." if len(spec.origin) > 50 else spec.origin
            print(f"{pkg_display:18s} | {status:15s} | {version:15s} | {origin_short}")
        else:
            print(f"{pkg_display:18s} | ❌ NO          | N/A            | NOT FOUND")
    except Exception as e:
        print(f"{pkg_display:18s} | ❌ ERROR       | N/A            | {str(e)[:50]}")
PYEOF
else
    echo "❌ Python実行ファイルが見つからないため、パッケージ確認をスキップ"
fi

# pip listでインストール済みパッケージの概要確認
echo "--- pip listでのインストール済みパッケージ概要 ---"
if [ -f "${VENV_BIN}/pip" ]; then
    echo "主要パッケージのバージョン:"
    ${VENV_BIN}/pip list 2>/dev/null | grep -E "(torch|transformers|vllm|ray|deepspeed|datasets|numpy|accelerate)" || echo "該当パッケージが見つかりません"
    echo ""
    echo "全パッケージ数: $(${VENV_BIN}/pip list 2>/dev/null | wc -l || echo 'N/A')"
else
    echo "⚠️  pipが見つかりません: ${VENV_BIN}/pip"
fi

# 仮想環境の整合性チェック
echo "--- 仮想環境の整合性チェック ---"
if [ -f "${VENV_PYTHON}" ] && [ -d "${VENV_SITE_PACKAGES}" ]; then
    ${VENV_PYTHON} << PYEOF
import sys
import os

# 仮想環境のパスを直接使用（環境変数から取得できない場合に備えて）
venv_base = "${VENV_DIR}"
if venv_base in sys.executable:
    print(f"✅ Python実行ファイルが仮想環境内にあります: {sys.executable}")
else:
    print(f"⚠️  Python実行ファイルが仮想環境外にあります: {sys.executable}")

# site-packagesがsys.pathに含まれているか確認
venv_site_packages = None
for path in sys.path:
    if 'site-packages' in path and 'venv' in path:
        venv_site_packages = path
        break

if venv_site_packages:
    print(f"✅ site-packagesがsys.pathに含まれています: {venv_site_packages}")
else:
    print(f"⚠️  site-packagesがsys.pathに含まれていません")

# 仮想環境のactivateスクリプトの確認
venv_activate = os.path.join(venv_base, 'bin', 'activate')
if os.path.exists(venv_activate):
    print(f"✅ activateスクリプトが存在します: {venv_activate}")
else:
    print(f"❌ activateスクリプトが存在しません: {venv_activate}")
PYEOF
else
    echo "⚠️  仮想環境の整合性チェックをスキップ（必要なファイルが見つかりません）"
fi

echo "=========================================="

# HuggingFace認証の設定
# ~/.cache/huggingface/tokenからトークンを読み込む
if [ -f "${HOME}/.cache/huggingface/token" ]; then
    # トークンファイルの最初の行を取得（重複している可能性があるため）
    HF_TOKEN_RAW=$(head -n 1 "${HOME}/.cache/huggingface/token" | tr -d '\n\r')
    
    # トークンの検証
    if [[ -z "$HF_TOKEN_RAW" ]]; then
        echo "Error: HF_TOKEN is empty after reading from ~/.cache/huggingface/token"
        exit 1
    fi
    
    if [[ ! "$HF_TOKEN_RAW" == hf_* ]]; then
        echo "Warning: HF_TOKEN format may be incorrect (should start with 'hf_')"
        echo "Token starts with: ${HF_TOKEN_RAW:0:3}"
    fi
    
    export HF_TOKEN="$HF_TOKEN_RAW"
    # huggingface_hubが確実にトークンを使用するように設定
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_RAW"
    echo "HF_TOKEN loaded successfully (length: ${#HF_TOKEN} chars)"
else
    echo "Warning: ~/.cache/huggingface/token not found. HuggingFace authentication may fail."
fi

# ========== HuggingFace設定（学習開始前に設定） ==========
# チェックポイントアップロード用の環境変数（デフォルト値）
UPLOAD_CHECKPOINTS_TO_HUB=${UPLOAD_CHECKPOINTS_TO_HUB:-true}
HF_REPO_ID=${HF_REPO_ID:-"HayatoHongoEveryonesAI/open-instruct-grpo-fast"}
HF_REPO_BASE_REVISION=${HF_REPO_BASE_REVISION:-"checkpoint"}
OUTPUT_DIR=${OUTPUT_DIR:-"output"}

# HuggingFaceのデバッグモード設定（学習中も有効）
if [ "${HF_DEBUG:-false}" = "true" ]; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    export TRANSFORMERS_VERBOSITY=debug
    echo "HuggingFace debug mode enabled"
else
    export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
fi

echo "HF_REPO_ID=${HF_REPO_ID}"
echo "HF_DEBUG=${HF_DEBUG:-false}"
echo "UPLOAD_CHECKPOINTS_TO_HUB=${UPLOAD_CHECKPOINTS_TO_HUB}"
echo "=========================================="

# Set PYTHONPATH to open-instruct root
OPEN_INSTRUCT_ROOT="${HOME}/LLM-jp-2025competetion-victory/open-instruct"
export PYTHONPATH="${OPEN_INSTRUCT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd ${OPEN_INSTRUCT_ROOT}

# Set vLLM environment variables (from grpo_fast.sh)
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
#export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fix Ray GPU device ID issue with single_gpu_mode
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export CUDA_VISIBLE_DEVICES=0
# RayにGPUリソースを認識させる（デバッグ前に設定）
export RAY_OVERRIDE_NUM_GPUS=1

# ========== GPU認識状況のデバッグ ==========
echo "========== GPU認識状況デバッグ =========="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
echo "NUM_NODES=${NUM_NODES}"

# nvidia-smiでGPU確認（全GPUを表示）
echo "--- nvidia-smi出力（全GPUリスト） ---"
nvidia-smi --list-gpus || nvidia-smi -L || echo "nvidia-smi list-gpus failed"
echo "--- nvidia-smi詳細情報（全GPU） ---"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || echo "nvidia-smi query failed"

# PyTorchでGPU確認（CUDA_VISIBLE_DEVICES設定後）
echo "--- PyTorch GPU確認（CUDA_VISIBLE_DEVICES設定後） ---"
python3 << 'EOF'
import torch
import os
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs detected by PyTorch: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    if torch.cuda.device_count() < 8:
        print(f"WARNING: Expected 8 GPUs but only {torch.cuda.device_count()} detected!")
        print("This may cause Placement Group creation to fail.")
else:
    print("ERROR: CUDA is not available!")
EOF

# Ray環境変数確認
echo "--- Ray関連環境変数 ---"
echo "RAY_OVERRIDE_NUM_GPUS=${RAY_OVERRIDE_NUM_GPUS:-Not set}"
echo "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO}"
echo "=========================================="

# Rayワーカーが仮想環境を認識せず、依存関係を再ダウンロード・インストールする問題を防ぐ
# Ray workers unpack code to /tmp/ray/... so /stage/.venv path doesn't exist there
# Unset VIRTUAL_ENV to prevent Ray workers from creating new virtual environments
unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT

# 仮想環境のPythonとパッケージを使用できるようにPATHとPYTHONPATHを明示的に設定
# 仮想環境のbinディレクトリをPATHの先頭に追加
VENV_BIN="${ENV_DIR}/venv/bin"
export PATH="${VENV_BIN}:${PATH}"

# ========== 仮想環境認識状況のデバッグ ==========
echo "========== 仮想環境認識状況デバッグ =========="
echo "ENV_DIR=${ENV_DIR}"
echo "VENV_BIN=${VENV_BIN}"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-Not set (unset済み)}"
echo "VIRTUAL_ENV_PROMPT=${VIRTUAL_ENV_PROMPT:-Not set (unset済み)}"

# PATHの確認
echo "--- PATH確認 ---"
echo "PATH=${PATH}"
echo "PATHに仮想環境のbinが含まれているか: $(echo ${PATH} | grep -q ${VENV_BIN} && echo 'YES' || echo 'NO')"

# Python実行パスの確認
echo "--- Python実行パス確認 ---"
PYTHON_EXEC=$(which python)
PYTHON_EXEC_PYTHON3=$(which python3)
echo "which python: ${PYTHON_EXEC}"
echo "which python3: ${PYTHON_EXEC_PYTHON3}"
echo "Python実行パスが仮想環境のものか: $(echo ${PYTHON_EXEC} | grep -q ${VENV_BIN} && echo 'YES' || echo 'NO')"

# Pythonバージョンとパス確認
echo "--- Python詳細情報 ---"
python << 'PYEOF'
import sys
import os
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}")  # 最初の3つだけ表示
print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not set')}")
print(f"PATH: {os.environ.get('PATH', 'Not set')[:200]}...")  # 最初の200文字だけ表示
PYEOF

# 仮想環境のパッケージ確認
echo "--- 仮想環境のパッケージ確認 ---"
python << 'PYEOF'
import sys
import importlib.util

# 主要パッケージのパスを確認
packages_to_check = ['torch', 'transformers', 'vllm', 'ray', 'deepspeed', 'datasets']
for pkg_name in packages_to_check:
    try:
        spec = importlib.util.find_spec(pkg_name)
        if spec and spec.origin:
            # 仮想環境のパスが含まれているか確認
            venv_path = '/LLM-jp-2025competition-victory/env/venv'
            is_in_venv = venv_path in spec.origin
            status = "✅ VENV" if is_in_venv else "⚠️  SYSTEM/OTHER"
            print(f"{pkg_name:15s}: {status} - {spec.origin}")
        else:
            print(f"{pkg_name:15s}: ❌ NOT FOUND")
    except Exception as e:
        print(f"{pkg_name:15s}: ❌ ERROR - {e}")
PYEOF

# HuggingFaceキャッシュの確認
echo "--- HuggingFaceキャッシュ確認 ---"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
echo "HF_HOME=${HF_HOME}"
if [ -d "${HF_HOME}" ]; then
    echo "HuggingFaceキャッシュディレクトリが存在します"
    echo "キャッシュサイズ: $(du -sh ${HF_HOME} 2>/dev/null | cut -f1 || echo 'N/A')"
else
    echo "⚠️  HuggingFaceキャッシュディレクトリが存在しません"
fi

# Rayワーカーが使用する環境変数の確認
echo "--- Rayワーカー用環境変数確認 ---"
echo "Rayワーカーに渡される環境変数:"
echo "  PATH=${PATH}"
echo "  PYTHONPATH=${PYTHONPATH}"
echo "  VIRTUAL_ENV=${VIRTUAL_ENV:-Not set (unset済み)}"
echo "  HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}"
echo "  TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-Not set}"
echo "  HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-Not set}"

# 仮想環境のsite-packages確認
echo "--- 仮想環境のsite-packages確認 ---"
VENV_SITE_PACKAGES="${ENV_DIR}/venv/lib/python3.12/site-packages"
if [ -d "${VENV_SITE_PACKAGES}" ]; then
    echo "site-packagesディレクトリが存在します: ${VENV_SITE_PACKAGES}"
    echo "パッケージ数: $(ls -1 ${VENV_SITE_PACKAGES} 2>/dev/null | wc -l)"
    echo "主要パッケージの存在確認:"
    for pkg in torch transformers vllm ray deepspeed; do
        if [ -d "${VENV_SITE_PACKAGES}/${pkg}" ] || [ -f "${VENV_SITE_PACKAGES}/${pkg}.py" ]; then
            echo "  ✅ ${pkg}"
        else
            echo "  ❌ ${pkg} (見つかりません)"
        fi
    done
else
    echo "⚠️  site-packagesディレクトリが存在しません: ${VENV_SITE_PACKAGES}"
fi

echo "=========================================="

# PYTHONPATHは既に設定されているが、念のため確認
echo "--- 環境変数確認（VIRTUAL_ENV unset後） ---"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-Not set (unset済み)}"
echo "PATH=${PATH}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "=========================================="

# Rayを事前に起動
echo "========== Rayクラスターの起動 =========="
ray stop --force || true
RAY_NODE_PORT=8888
ray start --head --port=${RAY_NODE_PORT} --dashboard-host=0.0.0.0 --num-gpus=1

# RAY_ADDRESSを設定（既存のクラスターに接続するため）
export RAY_ADDRESS="localhost:${RAY_NODE_PORT}"

# Rayクラスターの状態確認
ray status --address="${RAY_ADDRESS}"

echo "========== Rayクラスター起動完了 =========="

# # HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-v5-2 \
# Pythonスクリプトを実行
python open_instruct/grpo_fast.py \
    --dataset_mixer_list HayatoHongoEveryonesAI/qa_verify_2M_v5 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list HayatoHongoEveryonesAI/qa_verify_2M_v5 0.1 \
    --dataset_mixer_eval_list_splits train \
    --dataset_skip_cache \
    --max_prompt_token_length 1024 \
    --response_length 7168 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 12 \
    --num_samples_per_prompt_rollout 30 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --stop_strings "</answer>" \
    --apply_verifiable_reward true \
    --remap_verifier qa_10k=math-verify \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 1e-6 \
    --total_episodes 360000 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --vllm_num_engines 1 \
    --vllm_gpu_memory_utilization 0.25 \
    --beta 0.00 \
    --load_ref_policy false \
    --seed 3 \
    --local_eval_every 100 \
    --vllm_sync_backend gloo \
    --vllm_enable_prefix_caching \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --save_freq 100 \
    --single_gpu_mode \
    --checkpoint_state_dir output/grpo_fast_checkpoint_state \
    --checkpoint_state_freq 100 \
    --push_to_hub \
    --hf_entity HayatoHongoEveryonesAI \
    --hf_repo_id open-instruct-grpo-fast \
    --system_prompt_override_file scripts/train/debug/cute_debug_system_prompt.txt \
    --active_sampling \
    --filter_zero_std_samples \
    --async_steps 4 \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --no_resampling_pass_rate 0.9 \
    --clip_higher 0.272 \
    --mask_truncated_completions \
    --with_tracking \
    --wandb_entity hongo-hayato-6281k-university-of-tokyo \
    --wandb_project_name open-instruct-grpo-fast \
    --verbose \

echo "End time: $(date)"
echo "Training completed!"

# ========== チェックポイントをHuggingFace Hubにアップロード ==========
if [ "$UPLOAD_CHECKPOINTS_TO_HUB" = "true" ] && [ -n "$HF_REPO_ID" ]; then
    echo "========== チェックポイントをHuggingFace Hubにアップロード開始 =========="
    
    # OUTPUT_DIR配下で*_checkpointsパターンのディレクトリを検索
    # grpo_fast.pyでは output_dir が output/run_name になるため
    # チェックポイントディレクトリは output/{run_name}_checkpoints に保存される
    CHECKPOINT_BASE_DIR=$(find "${OUTPUT_DIR}" -maxdepth 2 -type d -name "*_checkpoints" 2>/dev/null | head -1)
    
    if [ -z "$CHECKPOINT_BASE_DIR" ] || [ ! -d "$CHECKPOINT_BASE_DIR" ]; then
        echo "Warning: Checkpoint directory not found in ${OUTPUT_DIR}"
        echo "Searched for pattern: ${OUTPUT_DIR}/*_checkpoints"
        echo "Skipping checkpoint upload."
    else
        echo "Found checkpoint directory: $CHECKPOINT_BASE_DIR"
        # 保存されているすべてのチェックポイントを検索してアップロード
        for checkpoint_dir in "${CHECKPOINT_BASE_DIR}"/step_*; do
            if [ -d "$checkpoint_dir" ]; then
                # step_500 -> 500 のようにステップ番号を抽出
                step_number=$(basename "$checkpoint_dir" | sed 's/step_//')
                
                if [ -n "$step_number" ]; then
                    # リビジョン名を生成（例: checkpoint_step_500）
                    if [ -n "$HF_REPO_BASE_REVISION" ]; then
                        revision_name="${HF_REPO_BASE_REVISION}_step_${step_number}"
                    else
                        revision_name="checkpoint_step_${step_number}"
                    fi
                    
                    echo "Uploading checkpoint: $checkpoint_dir -> ${HF_REPO_ID}/${revision_name}"
                    # HF_DEBUGが有効な場合は詳細ログを出力（既に学習開始前に設定済み）
                    if [ "${HF_DEBUG:-false}" = "true" ]; then
                        huggingface-cli upload \
                            --repo-id "${HF_REPO_ID}" \
                            --revision "${revision_name}" \
                            "${checkpoint_dir}" \
                            . -v || echo "Failed to upload $checkpoint_dir"
                    else
                        huggingface-cli upload \
                            --repo-id "${HF_REPO_ID}" \
                            --revision "${revision_name}" \
                            "${checkpoint_dir}" \
                            . || echo "Failed to upload $checkpoint_dir"
                    fi
                fi
            fi
        done
        echo "========== チェックポイントアップロード完了 =========="
    fi
else
    if [ "$UPLOAD_CHECKPOINTS_TO_HUB" != "true" ]; then
        echo "Checkpoint upload to Hub is disabled (set UPLOAD_CHECKPOINTS_TO_HUB=true to enable)"
    elif [ -z "$HF_REPO_ID" ]; then
        echo "Checkpoint upload to Hub is disabled (set HF_REPO_ID to enable)"
    fi
fi

echo "Test completed!"

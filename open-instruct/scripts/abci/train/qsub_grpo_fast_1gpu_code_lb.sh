#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N grpo_1gpu_code_lb
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=00:10:00
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

# ========== デバッグ環境用の分離設定 ==========
# 本番環境に影響しないようにポート番号とディレクトリを分離
echo "========== デバッグ環境用の分離設定 =========="

# Rayポートの分離（本番は8888、デバッグは8889）
RAY_NODE_PORT_DEBUG=8889
RAY_ADDRESS_DEBUG="localhost:${RAY_NODE_PORT_DEBUG}"

# tool_serverポートの分離（本番は1212、デバッグは1213）
# ロードバランサー用の設定
NUM_TOOL_SERVERS=4  # 起動するtool_serverの数
TOOL_SERVER_BASE_PORT=1214  # 最初のtool_serverポート番号
LOAD_BALANCER_PORT=1213  # ロードバランサーのポート
TOOL_SERVER_URL_DEBUG="http://localhost:${LOAD_BALANCER_PORT}/execute"

# TMPDIRの分離（デバッグ専用）
ORIGINAL_TMPDIR="${TMPDIR:-/tmp}"
DEBUG_TMPDIR="${ORIGINAL_TMPDIR}/debug_ray_${JOBID}"
export TMPDIR="${DEBUG_TMPDIR}"
mkdir -p "${DEBUG_TMPDIR}"
echo "デバッグ用TMPDIR: ${TMPDIR}"
echo "Rayポート（デバッグ）: ${RAY_NODE_PORT_DEBUG}"
echo "tool_server数: ${NUM_TOOL_SERVERS}"
echo "tool_serverベースポート: ${TOOL_SERVER_BASE_PORT}"
echo "ロードバランサーポート: ${LOAD_BALANCER_PORT}"
echo "tool_server URL（ロードバランサー経由）: ${TOOL_SERVER_URL_DEBUG}"
echo "=========================================="

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

# ========== 複数のtool_server起動 + ロードバランサー ==========
echo "========== tool_server起動（複数インスタンス + ロードバランサー） =========="
TOOL_SERVER_DIR="${OPEN_INSTRUCT_ROOT}/open_instruct/tool_utils"

# 既存のプロセスをクリーンアップ
for ((i=0; i<NUM_TOOL_SERVERS; i++)); do
    PORT=$((TOOL_SERVER_BASE_PORT + i))
    if lsof -ti:${PORT} > /dev/null 2>&1; then
        echo "警告: ポート${PORT}は既に使用中です。プロセスを停止します..."
        lsof -ti:${PORT} | xargs kill -9 || true
    fi
done

# ロードバランサーのポートもクリーンアップ
if lsof -ti:${LOAD_BALANCER_PORT} > /dev/null 2>&1; then
    echo "警告: ポート${LOAD_BALANCER_PORT}は既に使用中です。プロセスを停止します..."
    lsof -ti:${LOAD_BALANCER_PORT} | xargs kill -9 || true
fi

sleep 2

# tool_serverディレクトリに移動
cd "${TOOL_SERVER_DIR}" || {
    echo "Error: tool_server directory not found: ${TOOL_SERVER_DIR}"
    exit 1
}

# ログディレクトリ作成
mkdir -p "${OPEN_INSTRUCT_ROOT}/logs"

# 環境変数設定
export PREIMPORT_PKGS="pandas,numpy,sympy,time,math,networkx"
# ProcessPoolExecutorのワーカー数を各サーバーに分散
TOTAL_CPUS=$(nproc)
POOL_SIZE_PER_SERVER=$((TOTAL_CPUS / NUM_TOOL_SERVERS))
if [ ${POOL_SIZE_PER_SERVER} -lt 1 ]; then
    POOL_SIZE_PER_SERVER=1
fi
export POOL_SIZE=${POOL_SIZE_PER_SERVER}
echo "POOL_SIZE per server=${POOL_SIZE} (Total CPU cores: ${TOTAL_CPUS}, Servers: ${NUM_TOOL_SERVERS})"

# 複数のtool_serverを起動
TOOL_SERVER_PIDS=()
TOOL_SERVER_BASE_URLS=""

for ((i=0; i<NUM_TOOL_SERVERS; i++)); do
    PORT=$((TOOL_SERVER_BASE_PORT + i))
    TOOL_SERVER_LOG="${OPEN_INSTRUCT_ROOT}/logs/tool_server_${PORT}-${JOBID}.log"
    
    echo "tool_serverを起動中... (port: ${PORT})"
    nohup python -m uvicorn tool_server:app --host 0.0.0.0 --port ${PORT} > "${TOOL_SERVER_LOG}" 2>&1 &
    PID=$!
    TOOL_SERVER_PIDS+=(${PID})
    TOOL_SERVER_BASE_URLS="${TOOL_SERVER_BASE_URLS},http://localhost:${PORT}"
    
    echo "tool_server started with PID: ${PID} on port ${PORT}"
done

# 先頭のカンマを削除
TOOL_SERVER_BASE_URLS="${TOOL_SERVER_BASE_URLS:1}"

# tool_serverの起動確認（最大30秒待機）
echo "tool_serverの起動を確認中..."
for ((i=0; i<NUM_TOOL_SERVERS; i++)); do
    PORT=$((TOOL_SERVER_BASE_PORT + i))
    SERVER_READY=false
    for j in {1..30}; do
        if curl -s http://localhost:${PORT}/ > /dev/null 2>&1; then
            echo "✓ tool_server is responding on port ${PORT}"
            SERVER_READY=true
            break
        fi
        if [ $j -eq 30 ]; then
            echo "❌ tool_serverの起動に失敗しました（port: ${PORT}）"
            tail -20 "${OPEN_INSTRUCT_ROOT}/logs/tool_server_${PORT}-${JOBID}.log"
            exit 1
        fi
        sleep 1
    done
done

# ロードバランサーを起動
echo "ロードバランサーを起動中... (port: ${LOAD_BALANCER_PORT})"
export TOOL_SERVER_BASE_URLS="${TOOL_SERVER_BASE_URLS}"
LOAD_BALANCER_LOG="${OPEN_INSTRUCT_ROOT}/logs/loadbalancer_${LOAD_BALANCER_PORT}-${JOBID}.log"

nohup python -m uvicorn simple_loadbalancer:app --host 0.0.0.0 --port ${LOAD_BALANCER_PORT} > "${LOAD_BALANCER_LOG}" 2>&1 &
LOAD_BALANCER_PID=$!
echo "Load balancer started with PID: ${LOAD_BALANCER_PID}"

# ロードバランサーの起動確認
echo "ロードバランサーの起動を確認中..."
LOAD_BALANCER_READY=false
for i in {1..30}; do
    if curl -s http://localhost:${LOAD_BALANCER_PORT}/health > /dev/null 2>&1; then
        echo "✓ Load balancer is responding on port ${LOAD_BALANCER_PORT}"
        LOAD_BALANCER_READY=true
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ ロードバランサーの起動に失敗しました"
        tail -20 "${LOAD_BALANCER_LOG}"
        exit 1
    fi
    sleep 1
done

# テストリクエストで動作確認
echo "ロードバランサーの動作確認中..."
TEST_RESPONSE=$(curl -s -X POST "http://localhost:${LOAD_BALANCER_PORT}/execute" \
    -H "Content-Type: application/json" \
    -d '{"code": "print(42)", "timeout": 3}' \
    || echo "FAILED")

if echo "${TEST_RESPONSE}" | grep -q "output"; then
    echo "✅ ロードバランサーは正常に動作しています"
else
    echo "⚠️  ロードバランサーのテストリクエストに失敗しました"
    echo "レスポンス: ${TEST_RESPONSE}"
fi

# 環境変数を設定（grpo_fast.pyで使用）
export CODE_OUTPUT_API_URL="${TOOL_SERVER_URL_DEBUG}"
echo "CODE_OUTPUT_API_URL=${CODE_OUTPUT_API_URL}"
echo "TOOL_SERVER_BASE_URLS=${TOOL_SERVER_BASE_URLS}"

# 元のディレクトリに戻る
cd "${OPEN_INSTRUCT_ROOT}"

echo "========== tool_server + ロードバランサー起動完了 =========="
echo ""

# Rayを事前に起動
echo "========== Rayクラスターの起動（デバッグ用） =========="

# 1. 自分のRayクラスターのみをクリーンアップ（ポート番号を指定）
echo "デバッグ用Rayクラスターのクリーンアップを実行中..."
# デバッグ用アドレスのRayクラスターのみ停止（他のジョブには影響しない）
ray stop --address="${RAY_ADDRESS_DEBUG}" --force 2>/dev/null || true

# 自分のポートで実行中のRayプロセスのみを確認して終了
RAY_PIDS=$(lsof -ti:${RAY_NODE_PORT_DEBUG} 2>/dev/null || true)
if [ -n "${RAY_PIDS}" ]; then
    echo "ポート${RAY_NODE_PORT_DEBUG}で実行中のRayプロセスを停止: ${RAY_PIDS}"
    echo "${RAY_PIDS}" | xargs kill -9 || true
    sleep 5
fi

# Rayの一時ファイル保存先を確認してからクリーンアップ
echo "========== Ray一時ファイル保存先の確認 =========="
echo "TMPDIR=${TMPDIR:-/tmp}"
python3 << 'EOF'
import os
import glob

tmpdir = os.environ.get('TMPDIR', '/tmp')
print(f"TMPDIR: {tmpdir}")
print(f"Expected Ray session directory: {tmpdir}/ray/session_*")

# 実際のディレクトリを確認
ray_dirs = glob.glob(f"{tmpdir}/ray/session_*")
if ray_dirs:
    print(f"Found Ray session directories in {tmpdir}: {ray_dirs}")
else:
    print(f"No Ray session directories found in {tmpdir}/ray/")
EOF
echo "=========================================="

# デバッグ用Rayセッションディレクトリのみをクリーンアップ
if [ -d "${DEBUG_TMPDIR}/ray" ]; then
    echo "デバッグ用Rayセッションディレクトリをクリーンアップ: ${DEBUG_TMPDIR}/ray"
    rm -rf "${DEBUG_TMPDIR}"/ray* 2>/dev/null || true
fi

# 十分な待機時間を確保
echo "Rayクラスターのクリーンアップを待機中..."
sleep 10

# デバッグ用Rayクラスターを起動
RAY_NODE_PORT=${RAY_NODE_PORT_DEBUG}
ray start --head --port=${RAY_NODE_PORT} --dashboard-host=0.0.0.0 --num-gpus=1

# RAY_ADDRESSを設定（デバッグ用）
export RAY_ADDRESS="${RAY_ADDRESS_DEBUG}"

# Ray起動後に実際のセッションディレクトリを確認
echo "========== Ray起動後のセッションディレクトリ確認 =========="
python3 << 'EOF'
import ray
import os
import glob

try:
    if ray.is_initialized():
        # Rayが初期化されている場合、実際のセッションディレクトリを取得
        try:
            # Ray 2.50.0でのセッションディレクトリの取得方法
            import ray._private.utils as ray_utils
            session_dir = ray_utils.get_ray_temp_dir()
            print(f"Ray session directory (via API): {session_dir}")
        except Exception as e:
            print(f"Could not get Ray session directory via API: {e}")
            # フォールバック: TMPDIRから推測
            tmpdir = os.environ.get('TMPDIR', '/tmp')
            print(f"TMPDIR: {tmpdir}")
            print(f"Expected Ray session directory: {tmpdir}/ray/session_*")
            
            # 実際のディレクトリを確認
            ray_dirs = glob.glob(f"{tmpdir}/ray/session_*")
            if ray_dirs:
                print(f"Found Ray session directories: {ray_dirs}")
    else:
        print("Ray is not initialized yet")
except Exception as e:
    print(f"Error checking Ray session directory: {e}")
EOF
echo "=========================================="

# 2. 実行前のRayアクター確認
echo "Rayクラスターの状態を確認中..."
ray status --address="${RAY_ADDRESS}" || echo "Ray cluster not running"
# ray kill --all コマンドはRay 2.50.0では存在しないため削除
# 再度状態確認
ray status --address="${RAY_ADDRESS}"

echo "✅ デバッグ用Rayクラスターが起動しました: ${RAY_ADDRESS}"
ray status --address="${RAY_ADDRESS}"
echo "========== Rayクラスター起動完了 =========="

# # HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-v5-2 \
# Pythonスクリプトを実行
python open_instruct/grpo_fast.py \
    --dataset_mixer_list HayatoHongoEveryonesAI/dev-TIR_v3 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_skip_cache \
    --max_prompt_token_length 1024 \
    --response_length 3072 \
    --pack_length 4096 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 12 \
    --num_samples_per_prompt_rollout 30 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --apply_verifiable_reward true \
    --remap_verifier qa_10k=code-output \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think_code_execution \
    --learning_rate 1e-6 \
    --total_episodes 36000 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --vllm_num_engines 1 \
    --vllm_gpu_memory_utilization 0.3 \
    --beta 0.00 \
    --load_ref_policy false \
    --seed 3 \
    --vllm_sync_backend gloo \
    --vllm_enable_prefix_caching \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --local_eval_every -1 \
    --single_gpu_mode \
    --hf_entity HayatoHongoEveryonesAI \
    --hf_repo_id open-instruct-grpo-fast \
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
    --push_to_hub false \
    --verbose 

echo "End time: $(date)"
echo "Training completed!"

# ========== デバッグ環境のクリーンアップ ==========
echo "========== デバッグ環境のクリーンアップ =========="

# ロードバランサーを停止
if [ -n "${LOAD_BALANCER_PID}" ] && kill -0 ${LOAD_BALANCER_PID} 2>/dev/null; then
    echo "ロードバランサーを停止中: PID ${LOAD_BALANCER_PID}"
    kill ${LOAD_BALANCER_PID} || true
    sleep 2
else
    echo "ロードバランサープロセスが見つかりません（既に停止している可能性があります）"
fi

# ロードバランサーのポートで動作しているプロセスを停止
if lsof -ti:${LOAD_BALANCER_PORT} > /dev/null 2>&1; then
    echo "ポート${LOAD_BALANCER_PORT}で動作中のロードバランサープロセスを停止..."
    lsof -ti:${LOAD_BALANCER_PORT} | xargs kill -9 || true
fi

# 複数のtool_serverを停止
for ((i=0; i<NUM_TOOL_SERVERS; i++)); do
    PORT=$((TOOL_SERVER_BASE_PORT + i))
    PID_INDEX=$i
    if [ ${PID_INDEX} -lt ${#TOOL_SERVER_PIDS[@]} ]; then
        PID=${TOOL_SERVER_PIDS[${PID_INDEX}]}
        if [ -n "${PID}" ] && kill -0 ${PID} 2>/dev/null; then
            echo "tool_server (port ${PORT}) を停止中: PID ${PID}"
            kill ${PID} || true
        fi
    fi
    
    # ポートで動作しているプロセスも停止
    if lsof -ti:${PORT} > /dev/null 2>&1; then
        echo "ポート${PORT}で動作中のtool_serverプロセスを停止..."
        lsof -ti:${PORT} | xargs kill -9 || true
    fi
done

# 自分のRayクラスターのみを停止
echo "デバッグ用Rayクラスターを停止中..."
# デバッグ用アドレスのRayクラスターのみ停止（他のジョブには影響しない）
ray stop --address="${RAY_ADDRESS_DEBUG}" --force 2>/dev/null || true

# デバッグ用TMPDIRをクリーンアップ（オプション: デバッグ時に残したい場合はコメントアウト）
# rm -rf "${DEBUG_TMPDIR}" || true

echo "=========================================="

echo "Test completed!"

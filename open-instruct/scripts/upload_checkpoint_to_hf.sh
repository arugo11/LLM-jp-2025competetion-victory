#!/bin/bash
# チェックポイントをHugging Face Hubにアップロードするシェルスクリプト
# 仮想環境を自動的にアクティブ化してからPythonスクリプトを実行します

set -e

# スクリプトのディレクトリに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 環境設定を読み込む（存在する場合）
if [ -f "scripts/abci/common/setup.sh" ]; then
    echo "📦 Loading environment setup..."
    # ENV_DIRを設定（setup.shがPBS_NODEFILEを必要とする場合があるので、一部のみ読み込む）
    ENV_DIR="${ENV_DIR:-${HOME}/LLM-jp-2025competition-victory/env}"
    export ENV_DIR
    
    # 仮想環境をアクティブ化
    if [ -f "${ENV_DIR}/venv/bin/activate" ]; then
        source "${ENV_DIR}/venv/bin/activate"
        echo "✅ Virtual environment activated: ${ENV_DIR}/venv"
    else
        echo "⚠️  Warning: Virtual environment not found at ${ENV_DIR}/venv"
        echo "   Trying to continue with system Python..."
    fi
else
    echo "⚠️  Warning: setup.sh not found, trying default venv path..."
    ENV_DIR="${HOME}/LLM-jp-2025competition-victory/env"
    if [ -f "${ENV_DIR}/venv/bin/activate" ]; then
        source "${ENV_DIR}/venv/bin/activate"
        echo "✅ Virtual environment activated: ${ENV_DIR}/venv"
    fi
fi

# Pythonスクリプトを実行
echo "🚀 Running upload script..."
python scripts/upload_checkpoint_to_hf.py "$@"


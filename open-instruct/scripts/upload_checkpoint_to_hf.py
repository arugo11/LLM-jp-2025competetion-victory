#!/usr/bin/env python3
"""
ローカルのチェックポイントをHugging Face Hubにアップロードするスクリプト

使用方法:
    # 仮想環境をアクティブ化してから実行（推奨）
    source scripts/abci/common/setup.sh
    python scripts/upload_checkpoint_to_hf.py
    
    または、仮想環境のPythonを直接指定
    ~/LLM-jp-2025competition-victory/env/venv/bin/python scripts/upload_checkpoint_to_hf.py

環境変数:
    HF_TOKEN: Hugging Faceのトークン（~/.cache/huggingface/tokenから自動読み込みも可能）
    ENV_DIR: 仮想環境のディレクトリ（自動検出も可能）
"""

import os
import sys
from pathlib import Path

# 仮想環境のPythonパスを検出して警告を表示
def check_venv():
    """仮想環境の使用状況を確認して警告を表示"""
    script_dir = Path(__file__).parent.resolve()
    open_instruct_root = script_dir.parent
    
    # パターン1: ENV_DIR環境変数から
    env_dir = os.getenv("ENV_DIR")
    if not env_dir:
        # デフォルトのパスを試す
        env_dir = os.path.expanduser("~/LLM-jp-2025competition-victory/env")
    
    venv_python = None
    if env_dir and Path(env_dir).exists():
        venv_python_path = Path(env_dir) / "venv" / "bin" / "python"
        if venv_python_path.exists():
            venv_python = str(venv_python_path)
    
    # 現在のPythonパスを確認
    current_python = sys.executable
    
    # 仮想環境のPythonを使用していない場合、警告を表示
    if venv_python and current_python != venv_python:
        # 仮想環境のPythonパスが現在のパスに含まれているか確認
        if venv_python not in current_python:
            print("⚠️  Warning: Not using venv Python")
            print(f"   Current Python: {current_python}")
            print(f"   Recommended venv Python: {venv_python}")
            print(f"\n   To fix, run one of:")
            print(f"   1. source scripts/abci/common/setup.sh")
            print(f"   2. {venv_python} {sys.argv[0]} {' '.join(sys.argv[1:])}")
            print()
            
            # huggingface_hubがインポートできるか確認
            try:
                import huggingface_hub
            except ImportError:
                print("❌ Error: huggingface_hub module not found!")
                print(f"   Please activate the virtual environment first:")
                print(f"   source scripts/abci/common/setup.sh")
                sys.exit(1)
    elif not venv_python:
        # 仮想環境が見つからない場合
        print("⚠️  Warning: Could not find virtual environment")
        print(f"   Expected location: {env_dir}/venv/bin/python")
        print(f"   Trying to continue with current Python: {current_python}")
        print()

# 仮想環境のチェックを実行
check_venv()

from huggingface_hub import HfApi

# スクリプトがあるディレクトリを基準とした相対パスを設定
script_dir = Path(__file__).parent.resolve()
open_instruct_root = script_dir.parent

# Hugging Faceのトークンを取得
# 1. 環境変数から
hf_token = os.getenv("HF_TOKEN")

# 2. 環境変数がない場合は~/.cache/huggingface/tokenから読み込む
if not hf_token and os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
        hf_token = f.read().strip()

if not hf_token:
    print("❌ Error: HF_TOKEN not found.")
    print("   Set HF_TOKEN environment variable or place token in ~/.cache/huggingface/token")
    sys.exit(1)

# ローカルのチェックポイントディレクトリのパス
# デフォルト: output/grpo_fast_code_checkpoint_state/global_step2200
checkpoint_base_dir = open_instruct_root / "output" / "grpo_fast_code_checkpoint_state"
checkpoint_dir = checkpoint_base_dir / "global_step2200"

# アップロード先のRepository IDを指定
repo_id = "HayatoHongoEveryonesAI/open-instruct-grpo-fast"

# リビジョン名（ブランチ/タグ名）- 既存パターンに合わせる（アンダースコア2つ）
revision = "grpo_fast__3__2200"


def upload_checkpoint(checkpoint_path: Path, repo_id: str, revision: str):
    """チェックポイントをHugging Face Hubにアップロード"""
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint directory not found: {checkpoint_path}")
        print(f"\n確認してください:")
        print(f"  ls -la {checkpoint_path.parent}")
        
        # 利用可能なチェックポイントを表示
        if checkpoint_path.parent.exists():
            available = [d.name for d in checkpoint_path.parent.iterdir() if d.is_dir() and d.name.startswith("global_step")]
            if available:
                print(f"\n利用可能なチェックポイント:")
                for cp in sorted(available):
                    print(f"  - {cp}")
        return False
    
    print(f"✅ Found checkpoint: {checkpoint_path}")
    
    api = HfApi(token=hf_token)
    
    # リポジトリが存在しない場合は作成
    print(f"\n📦 Creating repository '{repo_id}' if it doesn't exist...")
    try:
        api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
        print(f"✅ Repository ready: {repo_id}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create repository: {e}")
    
    # ブランチを作成（存在しない場合）
    try:
        print(f"🔀 Creating branch '{revision}' if it doesn't exist...")
        api.create_branch(repo_id=repo_id, branch=revision, exist_ok=True)
        print(f"✅ Branch ready: {revision}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create branch (may already exist): {e}")
    
    # チェックポイントをアップロード
    print(f"\n📤 Starting upload...")
    print(f"   From: {checkpoint_path}")
    print(f"   To:   {repo_id}")
    print(f"   Revision: {revision}")
    
    try:
        api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            # optimizer statesを除外してモデルファイルのみアップロード
            ignore_patterns=["*optim_states.pt", "*optimizer*.pt", "*scheduler*.pt"],
        )
        print(f"\n✅ Upload completed successfully!")
        print(f"   Repository: https://huggingface.co/{repo_id}/tree/{revision}")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_multiple_checkpoints(steps: list[int], repo_id: str, base_revision: str = "grpo_fast__3__"):
    """複数のチェックポイントを一度にアップロード"""
    
    checkpoint_base_dir = open_instruct_root / "output" / "grpo_fast_code_checkpoint_state"
    api = HfApi(token=hf_token)
    
    # リポジトリを作成
    try:
        api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
        print(f"✅ Repository ready: {repo_id}")
    except Exception as e:
        print(f"⚠️  Warning: {e}")
    
    success_count = 0
    for step in steps:
        checkpoint_dir = checkpoint_base_dir / f"global_step{step}"
        revision = f"{base_revision}{step}"  # base_revisionに既に末尾のアンダースコアが含まれている
        
        print(f"\n{'='*60}")
        print(f"📤 Uploading step {step}...")
        print(f"{'='*60}")
        
        if upload_checkpoint(checkpoint_dir, repo_id, revision):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Upload summary: {success_count}/{len(steps)} checkpoints uploaded successfully")
    print(f"{'='*60}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload checkpoint to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 単一のチェックポイント（2200ステップ）をアップロード
  python scripts/upload_checkpoint_to_hf.py

  # 複数のチェックポイントをアップロード
  python scripts/upload_checkpoint_to_hf.py --steps 2000 2100 2200

  # カスタムリポジトリとリビジョン名を指定
  python scripts/upload_checkpoint_to_hf.py --repo-id MyOrg/my-repo --revision my_checkpoint_2200
        """
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[2200],
        help="Step numbers to upload (default: [2200])"
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        default=repo_id,
        help=f"Repository ID (default: {repo_id})"
    )
    
    parser.add_argument(
        "--revision-prefix",
        type=str,
        default="grpo_fast__3__",
        help="Prefix for revision names (default: grpo_fast__3__)"
    )
    
    parser.add_argument(
        "--checkpoint-base-dir",
        type=str,
        default=None,
        help="Base directory for checkpoints (default: output/grpo_fast_code_checkpoint_state)"
    )
    
    args = parser.parse_args()
    
    # チェックポイントベースディレクトリの設定
    if args.checkpoint_base_dir:
        checkpoint_base_dir = Path(args.checkpoint_base_dir)
    else:
        checkpoint_base_dir = open_instruct_root / "output" / "grpo_fast_code_checkpoint_state"
    
    # 複数のステップをアップロード
    if len(args.steps) > 1:
        upload_multiple_checkpoints(args.steps, args.repo_id, args.revision_prefix)
    else:
        # 単一のステップをアップロード
        step = args.steps[0]
        checkpoint_dir = checkpoint_base_dir / f"global_step{step}"
        revision = f"{args.revision_prefix}{step}"  # revision_prefixに既に末尾のアンダースコアが含まれている
        upload_checkpoint(checkpoint_dir, args.repo_id, revision)


if __name__ == "__main__":
    main()


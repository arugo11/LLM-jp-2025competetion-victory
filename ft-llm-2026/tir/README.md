# TIR on ABCI with Singularity + NeMo-Skills + vLLM

このサブプロジェクトは、ABCI 上で Singularity コンテナを用い、NeMo-Skills と vLLM、ローカル Python サンドボックスを組み合わせた TIR (Tool-Integrated Reasoning) 実験環境を再現するためのものです。`uv` ベースの環境定義とパッチ管理を用意し、ほぼ一発でデモが動く構成になっています。

## 使い方 (最速経路)
1. ログインノードでリポジトリ直下へ移動  
   `cd ~/LLM-jp-2025competetion-victory/ft-llm-2026/tir`
2. インタラクティブ TIR デモを投入  
   `qsub scripts/abci/run_tir_interactive.sh`
   - ジョブ内で `uv lock --python 3.10 && uv sync` し、`scripts/apply_patches.py` が必要なパッチを当てます。
   - ローカルサンドボックス (port 6000) と vLLM OpenAI API (port 8000) を起動し、`tir_demo.py` を実行します。
   - モデル取得には `HF_TOKEN` が環境変数で渡っている必要があります (ファイルには書き込みません)。

## コンテナ内で手動確認したい場合
1. 計算ノードに入る (例: `qsub -I -q rt_HG -l select=1 -l walltime=1:00:00 -P gch51701`)
2. 環境ロード  (`LLM-jp-2025competetion-victory/ft-llm-2026/tir/`のディレクトリで実行)
   `source scripts/abci/environment.sh`
3. コンテナへ入る  
   `singularity exec --nv --env CUDA_VISIBLE_DEVICES=0 --env NVIDIA_VISIBLE_DEVICES=0 --env VLLM_GPU_IDS=0 docker://nvcr.io/nvidia/pytorch:24.08-py3 bash`
4. プロジェクトルートでセットアップ  
   ```
   cd "$PROJECT_ROOT"
   pip install --user uv
   uv lock --python 3.10
   uv sync --python 3.10
   uv run python scripts/apply_patches.py
   # GPU バインドを数値 ID に固定（環境ロード時点で 0 に設定済みだが念のため）
   export CUDA_VISIBLE_DEVICES=0
   export NVIDIA_VISIBLE_DEVICES=0
   export VLLM_GPU_IDS=0
   ```
5. サーバ起動とデモ実行  
   ```
   nohup uv run python -m nemo_skills.code_execution.local_sandbox.local_sandbox_server --port 6000 > sandbox.log 2>&1 &
   nohup uv run python -m nemo_skills.inference.server.serve_vllm --model meta-llama/Meta-Llama-3.1-8B-Instruct --num_gpus 1 --num_nodes 1 --port 8000 --enforce-eager > vllm.log 2>&1 &
   sleep 120 #vLLMが立ち上がるまでの待機時間の目安
   uv run python tir_demo.py
   ```

## パッチ管理について
- `patches/` 配下に差分を保存し、`scripts/apply_patches.py` が `patch` コマンド経由で適用します。
- 内容:
- `sagemaker_sessions.patch`: `/dev/shm` を使わず `tempfile.gettempdir()/sagemaker_sessions` を利用するよう変更。
- `serve_vllm_enforce_eager.patch`: `serve_vllm` に `--enforce-eager` オプションを追加。
- 何度実行しても重複適用で失敗しないようにしています。
- `nemo-skills` は PyPI 未公開のため、GitHub (NVIDIA-NeMo/Skills@df9fbd9…) から取得します。計算ノード/コンテナから GitHub へのアウトバウンド通信が必要です。
- もし以前に `uv sync` を Python 3.11 などで実行して `.venv` が残っている場合は、一度 `.venv` を削除してから `uv lock --python 3.10 && uv sync --python 3.10` をやり直してください。
- vLLM が `CUDA_VISIBLE_DEVICES` に GPU UUID を渡されるとクラッシュするので、`environment.sh` が `CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES/VLLM_GPU_IDS` を 0 に強制します。手動実行時も同様に設定してください。

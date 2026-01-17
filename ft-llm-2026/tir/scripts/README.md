# scripts/README

ABCI 上での実行は基本的に `scripts/abci` 配下のジョブスクリプトを編集して `qsub` します。

## ディレクトリ構成
- `abci/`: PBS ジョブと qsub ラッパー
- `validate_qa_verify.py`: validate_range 実行用 CLI
- `merge_qa_verify.py`: merge_and_push 実行用 CLI
- `apply_patches.py`: NeMo-Skills/vLLM 用パッチ適用(現在運用しない想定)

## ABCI ジョブの使い方
1. `scripts/abci/*.sh` / `*.pbs` を必要に応じて編集
2. `qsub` で投入（例: `scripts/abci/qsub_tir_eval.sh`）

### TIR デモ/評価
- `scripts/abci/run_tir_interactive.sh`
- `scripts/abci/run_tir_eval.sh`
- ラッパー: `scripts/abci/qsub_tir_interactive.sh`, `scripts/abci/qsub_tir_eval.sh`

`environment.sh` が `PROJECT_ROOT` や Singularity の設定を行います。

### validate_range（検証ジョブ）
編集するもの:
- `scripts/abci/run_validate_range.pbs`

よく使う環境変数:
- `DATASET_REPO` (default: `HayatoHongoEveryonesAI/qa_verify_tir_1.3M_v0`)
- `START_ID` / `END_ID` (必須)
- `OUTPUT_PARQUET` / `OUTPUT_DIR`
- `NEMO_SKILLS_SANDBOX_HOST` / `NEMO_SKILLS_SANDBOX_PORT`
- `LOG_EVERY`, `TIMEOUT_SECONDS`, `MAX_OUTPUT_CHARS`, `CODE_MAX_CHARS`

投入例:
```
qsub scripts/abci/run_validate_range.pbs
```
範囲指定の例（ID=0..10000 を処理）:
```
qsub -v START_ID=0,END_ID=10001 \
  ft-llm-2026/tir/scripts/abci/run_validate_range.pbs
```
スキップ挙動:
- 既に `is_valid` が 0/1 の行は **デフォルトでスキップ**します。
- 再実行したい場合は `--force` を指定します。

### merge_and_push（反映ジョブ）
編集するもの:
- `scripts/abci/run_merge_and_push.pbs`

よく使う環境変数:
- `DATASET_REPO`
- `VALIDATION_DIR`
- `COMMIT_MESSAGE` (必須)
- `MAX_SHARD_SIZE`, `REVISION`, `PRIVATE=1` (default)

投入例:
```
qsub scripts/abci/run_merge_and_push.pbs
```

## 推奨フロー（main へ直接反映）
1. `validate_range` を実行して shard を作成
2. `merge_and_push` を実行して **main へ直接反映**（`REVISION` 未指定）

注意:
- `qsub -v` で `COMMIT_MESSAGE` にカンマを含めると失敗します。
  その場合は `qsub -V` を使うか、カンマを避けてください。

## ローカル実行（任意）
- `validate_qa_verify.py` / `merge_qa_verify.py` は CLI から直接実行できます。
- 依存は `requirements.lock.txt` に固定されています。
  - 反映列は `is_valid`, `validation_reason`, `stderr`, `stdout`, `runtime_ms`

### ローカルで main に直接反映（例）
```
cd /home/ach18380vf/LLM-jp-2025competetion-victory/ft-llm-2026/tir
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
source .venv_validate_py312/bin/activate

python scripts/merge_qa_verify.py merge_and_push \
  --dataset-repo HayatoHongoEveryonesAI/qa_verify_tir_1.3M_v0 \
  --validation-dir validation_shards \
  --commit-message "validate train ids=[1,100) via nemo-skills local sandbox, update is_valid and add validation_*"
```

## 注意
- `run_tir_interactive.sh` は PBS 実行時に `$0` が変わるため、`environment.sh` を絶対パスで読み込みます。
- ABCI では認証（`hf auth login`）を事前にログインノードで済ませてください。

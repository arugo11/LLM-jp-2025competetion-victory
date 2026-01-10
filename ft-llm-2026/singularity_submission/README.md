# Singularity Submission

LLM-JP チューニングコンペティション 2026 向け Singularity 提出用推論システム。
vLLM + NeMo-Skills サンドボックスを使用した Tool-Integrated Reasoning (TIR) による数学問題解答を行う。

## ディレクトリ構成

```
singularity_submission/
├── main.py              # エントリーポイント (シンプルな起動のみ)
├── config.py            # 設定・引数解析 (SolverConfig dataclass)
├── solver.py            # ProblemSolver クラス (問題解決のコアロジック)
├── pipeline.py          # パイプライン実行 (LLM/Sandbox初期化、問題処理)
├── executor.py          # LLM通信、サンドボックス実行ユーティリティ
├── prompts.py           # プロンプトテンプレート、メッセージ構築
├── utils.py             # テキスト処理、ロギング、データ構造
├── wandb_tracer.py      # W&B Weave統合（オプション）
├── download_model.py    # モデルダウンロードユーティリティ
├── submission.def       # Singularity定義ファイル
├── pyproject.toml       # Python依存関係
├── Makefile             # ビルド・実行タスク
└── models/              # ダウンロード済みモデル格納先
```


### 各モジュールの役割

| モジュール | 役割 |
|-----------|------|
| `main.py` | エントリーポイント。引数解析→設定変換→パイプライン実行 |
| `config.py` | `SolverConfig` dataclass と CLI引数解析。全設定を一元管理 |
| `solver.py` | `ProblemSolver` クラス。4フェーズの解決フローを実装 |
| `pipeline.py` | パイプライン実行。LLM/Sandbox初期化と問題処理ループ |
| `executor.py` | LLM生成・コード実行のユーティリティ関数 |
| `prompts.py` | プロンプトテンプレートとメッセージ構築関数 |
| `utils.py` | テキスト処理、`SolveResult` dataclass、ロギング |
| `wandb_tracer.py` | オプショナルなW&B Weave統合 |

## 前提条件

- Python 3.12+
- uv (https://docs.astral.sh/uv/)
- Singularity CE 4.1+
- NVIDIA GPU (vLLM実行用)
- NeMo-Skills

## セットアップ

### 1. 依存関係のインストール

```bash
uv sync
```

### 2. モデルのダウンロード

```bash
uv run python download_model.py --model_name team-victory/llm-jp-4-8b-instruct
```

ダウンロード先: `models/team-victory/llm-jp-4-8b-instruct`

### 3. GPU環境の設定
以下, GPU環境下で `LLM-jp-2025competetion-victory/ft-llm-2026/singularity_submission`で作業することを想定しています.
ABCI等でCUDA_VISIBLE_DEVICESがUUID形式の場合、数値に固定する必要がある。

```bash
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0
export VLLM_GPU_IDS=0
ulimit -n 65536 || true
```

## ローカル実行

### サンドボックスとvLLMサーバの起動

別々のターミナルで実行する。

```bash
# ターミナル1: サンドボックス
uv run python -m nemo_skills.code_execution.local_sandbox.local_sandbox_server \
    --port 6000 > sandbox.log 2>&1 &
```

```bash
# ターミナル2: vLLMサーバ
uv run python -m nemo_skills.inference.server.serve_vllm \
    --model models/team-victory/llm-jp-4-8b-instruct \
    --num_gpus 1 \
    --num_nodes 1 \
    --port 8000 \
    --enforce-eager > vllm.log 2>&1 &
```

### 推論の実行

```bash
uv run python main.py \
    --model_path models/team-victory/llm-jp-4-8b-instruct \
    --input_path sample_problems.jsonl \
    --output_path output.jsonl \
    --log_path inference_log.jsonl \
    --tir-llm-host 127.0.0.1 \
    --tir-llm-port 8000 \
    --tir-sandbox-host 127.0.0.1 \
    --tir-sandbox-port 6000 \
    --max-new-tokens 512 \
    --temperature 0.0 \
    --retry-temperature 0.2 \
    --repair-attempts 3 \
    --format-retry-attempts 5 \
    --direct-answer-attempts 2 \
    --log-raw-output \
    --enable-wandb \
    --wandb-project "hongo-hayato-6281k-university-of-tokyo/llm-jp-math-tir"
```

## コマンドライン引数

### 入出力

| 引数 | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `--model_path` | Yes | - | モデルディレクトリのパス |
| `--input_path` | Yes | - | 入力JSONL (問題) |
| `--output_path` | Yes | - | 出力JSONL (解答) |
| `--log_path` | No | `inference_log.jsonl` | 推論ログ |
| `--log-raw-output` | No | False | 生のLLM出力をログに含める |

### LLMサーバ

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--tir-llm-server-type` | `vllm` | サーバタイプ |
| `--tir-llm-host` | `127.0.0.1` | vLLMホスト |
| `--tir-llm-port` | `8000` | vLLMポート |
| `--llm-ready-timeout` | `300.0` | サーバ起動待機タイムアウト(秒) |

### サンドボックス

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--tir-sandbox-type` | `local` | サンドボックスタイプ |
| `--tir-sandbox-host` | `127.0.0.1` | サンドボックスホスト |
| `--tir-sandbox-port` | `6000` | サンドボックスポート |

### 生成パラメータ

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--max-new-tokens` | `512` | 最大生成トークン数 |
| `--min-tokens` | `0` | 最小生成トークン数 |
| `--temperature` | `0.0` | 生成temperature |
| `--retry-temperature` | None | リトライ時のtemperature |
| `--repair-attempts` | `1` | エラー時のリペア試行回数 |
| `--format-retry-attempts` | `2` | タグ欠落時のフォーマット修正リトライ回数 |
| `--direct-answer-attempts` | `2` | 実行失敗時の最終手段(推論のみ)リトライ回数 |

### コード実行

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--code-language` | `ipython` | 実行言語 |
| `--code-timeout` | `10.0` | 実行タイムアウト(秒) |
| `--max-output-chars` | `1000` | 出力最大文字数 |

### W&B Weave (オプション)

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--enable-wandb` | False | Weaveトレースを有効化 |
| `--wandb-project` | `llm-jp-math-tir` | プロジェクト名 (`entity/project` 形式でチーム指定可) |
| `--wandb-disabled` | False | 初期化のみ(送信無効) |

Weaveを使用する場合:

```bash
# 1. weaveパッケージを追加 (gql<4.0.0も必要)
uv add weave 'gql<4.0.0'

# 2. W&Bにログイン (初回のみ)
uv run wandb login

# 3. --enable-wandb フラグを付けて実行
uv run python main.py --enable-wandb ...

# チーム(entity)を指定する場合は entity/project 形式で指定
uv run python main.py --enable-wandb --wandb-project "your-team/your-project" ...
```

注意: `gql>=4.0.0` は `weave` と互換性がないため、`gql<4.0.0` を明示的に指定する必要がある。

ネットワーク接続がない環境では `--enable-wandb` を指定しなければ通常通り動作する。

## 出力フォーマット

LLMは以下のフォーマットで出力する。

```
<python>
from sympy import ...
# 計算処理
print(answer)
</python>
<result>answer</result>
```

### 解答採用の優先順位

1. stdoutの最後の非空行
2. `<result>` ブロックの内容
    - 通常のTIR出力に加え、実行不能時の「<result>のみ」フォールバックもここで採用される

### エラー種別

| エラー | 説明 |
|--------|------|
| `no_python_block` | `<python>` ブロックが見つからない |
| `execution_error` | コード実行エラー (Traceback, SyntaxError等) |
| `empty_stdout` | 実行成功したが出力がない |
| `empty_result` | `<result>` はあるが内容が空 |

## ログファイル

### inference_log.jsonl

問題ごとの推論結果サマリー。

```json
{
  "id": "problem_001",
  "has_python_block": true,
  "exec_success": true,
  "error": "",
  "final_output": "42",
  "temperature": 0.0,
  "repair_used": 0
}
```

## Singularityイメージのビルド

### 手動ビルド

```bash
export UV_CACHE_DIR="$(uv cache dir)"

singularity build --fakeroot --force \
    --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
    --build-arg MODEL_NAMES="team-victory/llm-jp-4-8b-instruct" \
    dist/submission.sif submission.def
```

### Makefileを使用

```bash
make dist/submission.sif
```

## Singularityでの実行

```bash
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 \
    dist/submission.sif \
    --model_path /app/models/team-victory/llm-jp-4-8b-instruct \
    --input_path /path/to/input.jsonl \
    --output_path /path/to/output.jsonl \
    --tir-llm-host 127.0.0.1 \
    --tir-llm-port 8000 \
    --tir-sandbox-host 127.0.0.1 \
    --tir-sandbox-port 6000
```

注意: vLLMサーバとサンドボックスは事前に起動しておく必要がある。

## トラブルシューティング

### vLLM起動エラー: invalid literal for int()

CUDA_VISIBLE_DEVICESがUUID形式になっている。数値に固定する。

```bash
export CUDA_VISIBLE_DEVICES=0
```

### Connection error

`--net --network none` を指定するとコンテナから127.0.0.1へ接続できない。
`--net` オプションを外すか、別ホストのサーバを指定する。

### 別モデルを試したいのに 404 (model does not exist)

`main.py --model_path ...` は vLLM(OpenAI互換)サーバが起動時に読み込んだモデルに対して推論します。
別のモデルを試す場合は、vLLMサーバも同じモデルで起動し直してください（`serve_vllm --model <model_path>`）。

### vLLM起動エラー: Converting from SentencePiece and Tiktoken failed

モデルディレクトリに `tokenizer.json` または `tokenizer.model` が同梱されていないと、vLLM がトークナイザ初期化で落ちることがあります。

対処:

- 最も簡単: ベースモデルの `tokenizer.json` を対象モデルディレクトリへコピーする
- もしくは: vLLM起動時に `--tokenizer <tokenizer_path>` を追加して、トークナイザだけ別パスを使う

`serve_vllm` は未知引数をそのまま vLLM に渡すので、例えば以下が動きます。

```bash
uv run python -m nemo_skills.inference.server.serve_vllm \
    --model models/HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long-v5 \
    --tokenizer models/team-victory/llm-jp-4-8b-instruct \
    --num_gpus 1 \
    --num_nodes 1 \
    --port 8000 \
    --enforce-eager
```

### vLLM起動エラー: Cannot find any model weights

`RuntimeError: Cannot find any model weights with '<model_path>'` は、指定したモデルディレクトリ直下に重みファイル（例: `model-00001-of-00004.safetensors` など）が存在しない状態です。

典型例:

- `model.safetensors.index.json` はあるが、対応する `model-0000*-of-0000*.safetensors` が無い

対処:

- Hugging Face Hub から重みを含めて再ダウンロードする（アクセス権が必要な場合は `HF_TOKEN` を設定）

```bash
uv run python download_model.py \
    --model_name HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long-v5
```

- もしくは、実際に重み（`.safetensors` / `.bin`）が置かれているディレクトリを `--model` に指定する

ダウンロード後、モデルディレクトリ直下に `model-00001-of-00004.safetensors` などが存在することを確認してから vLLM を起動してください。

## モジュール構成

| モジュール | 責務 |
|-----------|------|
| main.py | エントリーポイント、CLI、ProblemSolver、パイプライン |
| prompts.py | プロンプトテンプレート、メッセージ構築関数 |
| utils.py | テキスト処理、SolveResult |
| executor.py | LLM通信、サンドボックス実行、エラー判定 |
| wandb_tracer.py | W&B Weave統合（オプション） |

## ライセンス

プロジェクトのライセンスに従う。

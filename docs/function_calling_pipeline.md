# llm-jp Function Calling 推論手順

本ドキュメントでは、`LLM-jp-2025competetion-victory` リポジトリを使って Function Calling 推論を行うまでの流れを最初から最後までまとめます。環境は Ubuntu + NVIDIA A100 (80GB) を想定していますが、他の環境でも基本は同じです。

---

## 1. 事前準備

1. **UV / Python 環境**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv .venv
   source .venv/bin/activate
   ```

2. **依存インストール**
   ```bash
   uv pip install -e Skills
   uv pip install -e third_party/Math-Verify[antlr4_9_3]
   uv pip install "vllm>=0.5.4"
   ```
   - `antlr4-python3-runtime==4.9.3` を使う点に注意（`math-verify` と `omegaconf` が同じバージョンを仮定しているため）。

3. **OAuth 系トークン (.env)**
   ```dotenv
   HF_TOKEN=****************
   WANDB_API_KEY=****************  # W&B を使う場合
   NS_CONFIG_DIR=/home/user/yudai/projects/LLM-jp-2025competetion-victory/Skills/cluster_configs/local-a100.yaml
   ```
   - W&B を使わない場合は `WANDB_API_KEY` を省略可。

---

## 2. Math-Verify & データ整形

1. **Math-Verify クローン**（まだの場合）
   ```bash
   git clone https://github.com/huggingface/Math-Verify.git third_party/Math-Verify
   ```

2. **データセット整形 & gold CSV**
   ```bash
   uv run --env-file .env python scripts/build_llmjp_fc_dataset.py \
     --input data/dev.jsonl \
     --output Skills/nemo_skills/dataset/llmjp_fc_dev/test.jsonl \
     --math_verify_csv Skills/nemo_skills/dataset/llmjp_fc_dev/gold.csv
   ```
   - `expected_answer` が Math-Verify に最適化された LaTeX 形式で保存されます。

3. **ns prepare_data で再現可能に**
   ```bash
   uv run --env-file .env ns prepare_data llmjp_fc_dev
   ```
   - `Skills/nemo_skills/dataset/llmjp_fc_dev/prepare.py` が上記スクリプトを呼び出します。

---

## 3. クラスタ構成と Docker キャッシュ

1. **`ns setup` を利用する**
   ```bash
   uv run --env-file .env ns setup
   ```
   - 対話形式で保存場所（例: `Skills/cluster_configs`）と config 名を指定します。
   - `local` を選び、マウントに `/home/user/yudai/projects:/workspace/projects` などを登録。
   - `HF_HOME` は必ず上記マウント配下（例: `/workspace/projects/.cache/huggingface`）を指定してください。
   - プロンプト終盤に「Pull/Build するか？」と聞かれたら **Yes** にすると vLLM を含む必要コンテナが事前にビルドされ、以後の `ns eval` が高速化します（Docker キャッシュ対策）。

2. **作成済みの `local-a100.yaml`**
   - `executor: local`
   - `containers.vllm = vllm/vllm-openai:v0.11.0`
   - マウント例:
     - `/home/user/.cache/huggingface -> /models/hf-cache`
     - `/home/user/yudai/projects -> /workspace/projects`
     - `/home/user/yudai/data -> /workspace/data`
     - `/home/user/yudai/projects/LLM-jp-2025competetion-victory/third_party -> /workspace/third_party`
   - `env_vars`: `HF_HOME=/models/hf-cache`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

3. **`NS_CONFIG_DIR` / `NEMO_SKILLS_CONFIGS`**
   - `.env` に `NS_CONFIG_DIR=/home/user/.../Skills/cluster_configs/local-a100.yaml` を入れるか、
   - `NEMO_SKILLS_CONFIGS=/home/user/.../Skills/cluster_configs` をエクスポートしておくと `--cluster` 省略で自動解決されます。

---

## 4. 推論実行

1. **出力先ディレクトリをマウント内に用意**
   ```bash
   mkdir -p /home/user/yudai/projects/llmjp4-fc-dev
   ```
   -> コンテナ側では `/workspace/projects/llmjp4-fc-dev` として見える。

2. **Function Calling 推論コマンド**
   ```bash
   uv run --env-file .env ns eval \
     --benchmarks llmjp_fc_dev \
     --cluster local-a100 \
     --model team-victory/llm-jp-4-8b-instruct \
     --server_type vllm \
     --server_gpus 2 \
     --output_dir /workspace/projects/llmjp4-fc-dev \
     ++parse_reasoning=True \
     ++inference.tokens_to_generate=4096 \
     ++use_client_parsing=False \
     --server_args="--enable-auto-tool-choice \
                    --tool-call-parser hermes \
                    --max-model-len 8192 \
                    --gpu-memory-utilization 0.9" \
     --wandb_project llmjp-fc \
     --wandb_name llmjp4-dev-eval
   ```
   - vLLM は `hf://...` 形式を解釈しないため、**Hugging Face の標準表記 (`namespace/model`) かローカルパス** を渡す点に注意してください。
   - `--server_args` には vLLM の Tool Calling オプションを渡します。環境に応じて `--tensor-parallel-size`、`--dtype bfloat16` などを追記してください。
   - W&B を使わない場合は `--wandb_project/--wandb_name` を削除するか `--disable_wandb` を付与。

3. **実行後の確認**
   - `/workspace/projects/llmjp4-fc-dev/metrics.json` で Math-Verify による `symbolic_correct` を確認。
   - `log_samples_wandb.py` により W&B に生成サンプルが自動送信されます。

---

## 5. トラブルシューティング

| 症状 | 原因/対処 |
| ---- | -------- |
| `ns prepare_data llmjp_fc_dev` が `dev.jsonl` を見つけられない | `prepare.py` は `repo_root/data/dev.jsonl` を参照します。`data/dev.jsonl` が存在するか、パスを修正してください。 |
| `ns eval` が “path … is not mounted” で停止 | `--output_dir` をクラスタ設定でマウント済みのディレクトリに変更するか、`local-a100.yaml` に対象パスを追加します。 |
| `ns --help` で ANTLR エラー | `antlr4-python3-runtime==4.9.3` を再インストールして `math-verify[antlr4_9_3]` を入れ直してください。 |
| vLLM 関連の ImportError | `uv pip install "vllm>=0.5.4"` を忘れていないか確認。 |

---

## 6. 補足

- Math-Verify 用の gold CSV (`Skills/nemo_skills/dataset/llmjp_fc_dev/gold.csv`) は、推論結果 CSV と結合して単体で評価したいときに活用できます。
- `scripts/build_llmjp_fc_dataset.py` を追加データ (train/test) に使う場合は `--input/--output` を切り替えてください。
- すべてのコマンドは `uv run --env-file .env ...` で実行するとトークンやクラスタ設定が統一的に渡せます。

これで Function Calling 推論までのセットアップと実行手順は完了です。必要に応じて適宜パラメータを調整してご利用ください。

---

## 7. 備忘録: 非CoTモデルで出力が空になる問題を解消した手順

### 7.1 症状
- `team-victory/llm-jp-4-8b-instruct` は Chain-of-Thought モデルではないため `<think>…</think>` タグを出力しない。
- それにもかかわらず `++parse_reasoning=True` のまま実行すると、ログに大量の  
  `Thinking end tag </think> not found in generation; setting generation to empty.` が出て、すべての `generation` が空扱いになる。
- 評価結果 (`metrics.json`) では `symbolic_correct=0% / no_answer=100%` となり、`output.jsonl` の `predicted_answer` も `None` で埋まってしまう。

### 7.2 対応方針
1. **Reasoning parsing を無効化**  
   - コマンドラインで `++parse_reasoning=False` を指定して空文字になるのを防ぐ。
2. **データセットに最終回答形式を明示**  
   - `Skills/nemo_skills/dataset/llmjp_fc_dev/test.jsonl` の System Message 末尾に  
     “最終解は必ず `\boxed{}` 形式で 1 行で答える” という追記を自動付与（100件すべて）。
   - これにより非ツールモデルでも確実に最終値が抽出しやすくなる。
3. **推論側でテキストから `predicted_answer` を補完**  
   - `Skills/nemo_skills/inference/eval/bfcl.py` の `BFCLGenerationTask` に、Tool Call がない応答から  
     `\boxed{...}` / LaTeX 環境 / 「答え:」などのキーワード / 最終行の等式 / 数値 を正規表現で拾って後処理するロジックを追加。
   - 算用数字の末尾に付く日本語の単位（例: “4個”, “720円” など）は、ユニット除去の正規表現で丸め落とす。
4. **評価側で事前抽出済み回答を信頼**  
   - `++eval_config.use_predicted_answer_key=True` を付けて `MathEvaluator` が自力で再抽出しないようにする。  
     （さもないとフォーマットが合わず再び `None` になる可能性がある。）

### 7.3 再実行コマンド例
```bash
uv run --env-file .env ns eval \
  --benchmarks llmjp_fc_dev \
  --cluster local-a100 \
  --model team-victory/llm-jp-4-8b-instruct \
  --server_type vllm \
  --server_gpus 1 \
  --output_dir /workspace/projects/llmjp4-fc-dev \
  ++parse_reasoning=False \
  ++inference.tokens_to_generate=3000 \
  ++use_client_parsing=False \
  ++skip_filled=False \
  ++eval_config.use_predicted_answer_key=True \
  --server_args="--enable-auto-tool-choice --tool-call-parser hermes \
                 --max-model-len 4096 --gpu-memory-utilization 0.9" \
  --wandb_project llmjp-fc \
  --wandb_name llmjp4-dev-eval \
  --rerun_done
```

### 7.4 成果
- `/home/user/yudai/projects/llmjp4-fc-dev/eval-results/llmjp_fc_dev/output.jsonl` の 100 行すべてに `predicted_answer` が入るようになった。
- `metrics.json` では `symbolic_correct=47% / no_answer=0%` まで回復（2025-11-09 23:09 JST 時点）。
- 実行ログは `llmjp_fc_dev_no_cot.log` に保存済み。再調査時にはこのログで `</think>` 警告が消えていることも確認できる。

> **Tip:** 今後 CoT タグを出さないモデルを評価する際は、最初から `++parse_reasoning=False` と  
> `++eval_config.use_predicted_answer_key=True` をセットで付けておくと安全です。

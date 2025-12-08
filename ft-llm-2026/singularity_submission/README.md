# Singularity提出用サンプル

このディレクトリは、チューニングコンペティション2026でSingularityイメージを提出するための最小構成例です。
`main.py` が評価用入出力を処理し、Singularity イメージ内では `uv` で固定された Python 依存関係が利用されます。

## 同梱物

- `main.py` : Singularityから実行される推論スクリプト（Python/結果タグの厳格テンプレートを強制）
- `download_model.py` : Hugging Face Hubからモデルを取得するユーティリティ
- `sample_problems.jsonl`, `sample_problems_2.jsonl` : 推論手順を確認するためのサンプル入力
- `submission.def` : Singularity定義ファイル
- `pyproject.toml`, `uv.lock` : Python 依存関係定義

## 前提条件

- Singularity (ABCIでは `singularity-ce version 4.1.5-1.el9` が利用可能)
- [uv](https://docs.astral.sh/uv/)
- NeMo-Skills のローカルサンドボックスおよび vLLM サーバを起動できるGPU環境

## TIRランタイムの起動

`main.py` は NeMo-Skills の `get_sandbox` / `get_code_execution_model` を使用します。
推論を実行する前に、以下のように Python サンドボックスと vLLM サーバを立ち上げてください。

1. GPU ID を UUID ではなく数値に固定します (`tir/README.md` と同様)。
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   export NVIDIA_VISIBLE_DEVICES=0
   export VLLM_GPU_IDS=0
   # 任意: open files の警告を抑えるために ulimit を引き上げ
   ulimit -n 65536 || true
   ```
   - ABCI などで `CUDA_VISIBLE_DEVICES` が `GPU-xxxx` 形式の UUID のままだと、vLLM 起動時に
     `invalid literal for int(): 'GPU-…'` というエラーが発生します。必ず上記のように数値へ固定してください。
   - Singularity を使う場合は、必要に応じて `SINGULARITYENV_CUDA_VISIBLE_DEVICES` なども同じ値に設定します。

```bash
# 例: それぞれ別シェルで実行 (モデル名やGPU割り当ては環境に合わせて変更)
uv run python -m nemo_skills.code_execution.local_sandbox.local_sandbox_server --port 6000 > sandbox.log 2>&1 &
uv run python -m nemo_skills.inference.server.serve_vllm \
    --model llm-jp/llm-jp-3.1-1.8b-instruct4 \
    --num_gpus 1 --num_nodes 1 --port 8000 --enforce-eager > vllm.log 2>&1 &
```

- モデル取得に必要な `HF_TOKEN` などの環境変数は適宜設定してください。
- ポートやホストを変えた場合は、`main.py` の `--tir-llm-host/--tir-llm-port`、`--tir-sandbox-host/--tir-sandbox-port` で接続先を上書きできます。
- より詳しいセットアップ手順は `../tir/README.md` も参照してください。

## 1. モデルの準備

この例では `llm-jp/llm-jp-3.1-1.8b-instruct4` を使用します。
以下を実行すると `models/llm-jp/llm-jp-3.1-1.8b-instruct4` にダウンロードされます。

```bash
uv run python download_model.py --model_name llm-jp/llm-jp-3.1-1.8b-instruct4
```

## 2. Singularityイメージのビルド

1. `uv` のキャッシュディレクトリを環境変数に保持します。
   ```bash
   export UV_CACHE_DIR="$(uv cache dir)"
   ```
2. 含めたいモデルのリストを空白区切りで指定し、Singularityイメージをビルドします。
   この例では前節でダウンロードしたモデルのみを含めています。
   ```bash
   singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       --build-arg MODEL_NAMES="llm-jp/llm-jp-3.1-1.8b-instruct4" \
       dist/submission.sif submission.def
   ```
   - 複数モデルを含めたい場合は `MODEL_NAMES` を空白区切りで列挙してください。
   - `--bind` によってビルド時の Python 依存関係取得が高速化されます。

## 3. ローカルでの動作確認

生成したイメージを使用してサンプル入力で推論を行います。GPUを利用するコマンド例は以下の通りです。

```bash
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 \
    dist/submission.sif \
    --model_path models/llm-jp/llm-jp-3.1-1.8b-instruct4 \
    --tir-model-name "llm-jp/llm-jp-3.1-1.8b-instruct4" \
    --input_path sample_problems.jsonl \
    --output_path "$(pwd)/output.jsonl" \
    --log_path "$(pwd)/inference_log.jsonl" \
    --tir-llm-host 127.0.0.1 --tir-llm-port 8000 \
    --tir-sandbox-host 127.0.0.1 --tir-sandbox-port 6000
```

`--model_path` にはイメージに同梱したモデルディレクトリを指定します。
NeMo-Skills サーバ/サンドボックスを別ノードで動かしている場合は、上記の `--tir-*` 引数で接続先を変更してください。
推論結果は `output.jsonl` に書き出されます。
推論ログはデフォルトでカレントディレクトリの `inference_log.jsonl` に追記されます（`--log_path` で変更可能）。

### 出力フォーマットと実行ポリシー（重要）

- LLMの出力は必ず次の2ブロックのみを許可します。余計な文章・Markdownフェンスは禁止です。
  ```
  <python>
  print(answer)
  </python>
  <result>answer</result>
  ```
- `<python>` / `<result>` が複数返った場合は最初の1つだけを採用し、ログにフラグを残します。
- `<python>` が欠落、または実行時に `SyntaxError` / `Traceback` が出た場合は出力を `no_python_block` / `execution_error` として扱い、詳細は `inference_log.jsonl` に記録します。
- 解答採用の優先順位は「stdout の最後の非空行 > `<result>` ブロック > 生テキスト」です。
- `--log_path` を指定すると JSONL で `has_python_block`, `multiple_python_blocks`, `used_stdout`, `error`, `final_output` などのメタ情報が追記されます。

### トラブルシュート

- vLLM サーバやサンドボックスをホスト側で動かす場合、`--net --network none` を付けるとコンテナから 127.0.0.1 へ接続できず **Connection error** になります。上記のように `--net` を付けずに実行してください。別ホストにサーバを立てている場合は `--tir-llm-host` / `--tir-sandbox-host` にそのホスト名を指定してください。
- `ulimit -n` が権限不足で失敗しても実行自体は続行可能です。
- vLLM に登録されたモデル名と `--tir-model-name` を必ず一致させてください。`serve_vllm --model llm-jp/llm-jp-3.1-1.8b-instruct4` のようにスラッシュ付きで起動した場合、デフォルトでは `model_path` の末尾名（スラッシュ無し）が送られて 404 になります。上記の例のように `--tir-model-name "llm-jp/llm-jp-3.1-1.8b-instruct4"` を明示指定してください。

## 応用

- 推論ロジックを変更する場合は `main.py` を編集してください。
  引数を追加した場合は `singularity run` の呼び出しにも反映させてください。
- Python依存関係を変更する際は `pyproject.toml` を編集し、`uv lock` を実行して
  `uv.lock` を更新します。その後あらためてイメージを再ビルドしてください。
- OS 依存パッケージが必要になった場合は `submission.def` の `%post` セクションに
  `apt-get install` 等を追記します。イメージサイズに注意してください。

このサンプルをベースに、各チームの推論コードやモデルを組み込んだ提出イメージを作成してください。

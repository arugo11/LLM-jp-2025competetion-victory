# このディレクトリについて
このディレクトリでは推論コードについてまとめています。

# 簡単な推論と評価

## バッチジョブによる実行
inference-*.shにバッチジョブ用のスクリプトがあります。以下のコマンドで実行できます。
``` bash
qsub inference-1gpu.sh
```
ターミナルの出力は.logディレクトリにoutファイルとerrファイルの形式で保存されます。


[2025/1/10 inference-1gpu.shの更新内容]`inference-1gpu.sh` のパラメータのデフォルト値が以下のとおり更新されました。
`max_tokens`: 4096 → 16384, `num_samples`: **10 → 40**, `temperature`: **1.0 → 0.7**
また、モデルがすでにダウンロードされている場合は、以下のように **コマンドライン引数でモデルを指定**して実行することも可能です。
```bash
qsub -v MODEL_USER=openai,MODEL_REPO=gpt-oss-20b ./inference-eval-1gpu.sh
```
なお、**シングルノード用の `inference-1node.sh` については特に変更はありません。**


## バッチジョブによる、推論実行と評価
k回推論し、pass@kと、cons@kの評価も同時にします。
結果ファイルは、../math-eval/accuracyに保存されます。
``` bash
qsub -v MODEL_USER=openai,MODEL_REPO=gpt-oss-20b ./inference-eval-1gpu.sh
```

# 詳しい推論処理の手順
この例では、"HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long-v5" を対象に、1GPUでSelf-Consistency による推論処理を行います。
## 前提条件
- Singularity (ABCIでは `singularity-ce version 4.1.5-1.el9` が利用可能)
- [uv](https://docs.astral.sh/uv/)
## 0. GPUサーバーへのアクセス
ログインノードで以下を実行。
```
qsub -I -P gch51701 -q rt_HG -l select=1 -l walltime=1:00:00
```
このコマンドをそのまま実行した場合、1時間で自動終了します。"walltime"で自動終了する時間を変更できます。

## 1. モデルのダウンロード
ABCI上でチームのHuggingFaceアカウントにログイン(=HF_TOKENを登録)している状態で以下を実行します。これにより、"models/HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long-v5"にモデルがダウンロードされます。
``` bash
uv run python download_model.py --model_name HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long-v5
```
コマンド内の"HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long-v5"の部分はHuggingFace上の任意のモデルに変更することができます。その場合、"username/model_name" の形で指定してください。

[注] : "models"ディレクトリがない場合は作成してください。また、異なるモデルを使う際は"その他"の"モデルの変更"を確認してください。

## 2. Singularityイメージのビルド
はじめに"uv"の キャッシュディレクトリを環境変数に追加します。
```
export UV_CACHE_DIR="$(uv cache dir)"
```
以下のコマンドによりSingularityイメージがビルドされます。
``` bash
singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       --build-arg MODEL_NAMES="HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long-v5" \
       dist/sft-long-v5.sif self-consistency.def
```
- MODEL_NAMES: 1 でダウンロードしたモデルを指定
- dist/sft-long-v5.sif: ビルド後のファイルの保存先
    - [注] dist ディレクトリがない場合は作成すること
- self-consistency.def: Singularityの定義ファイル(.def)

MODEL_NAMES に複数のモデルを含むことができるらしい。(要検証)
## 3. 推論処理の実行(1GPU)
2 でビルドした sif ファイルを実際に動作させます。結果は "output/output.jsonl" に保存されます。outputディレクトリがない場合は作成してください。
``` bash
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 --net --network none dist/sft-long-v5.sif \
    --model_path models/HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long-v5 \
    --input_path input/dev.jsonl \
    --output_path "$(pwd)/output/output.jsonl" \
    --num_samples 10 \
    --max_tokens 4096 \
    --temperature 1.0
```
引数の説明
- --model_path: 推論に使用するモデルを指定
- --input_path: 推論対象のファイル(問題のファイル)
- --output_path: 推論結果の出力先
- --max_tokens: 最大系列長
- --num_samples: Self-Consistencyによる推論回数
- --temperature: 推論時の温度パラメータ

# その他
## 1ノードでの推論
"CUDA_VISIBLE_DEVICES"に使用したいGPUのIDをコンマ区切りで指定
``` bash
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --net --network none dist/sft-long-v5.sif \
    --model_path models/HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long-v5 \
    --input_path input/dev.jsonl \
    --output_path "$(pwd)/output/output.jsonl" \
    --max_tokens 4096 \
    --num_samples 10 \
    --temperature 1.0
```
[注] 実行する前に、"0. GPUサーバーへのアクセス"において、rf_HGの部分をrf_HFに変更し、1ノード用の計算ノードにアクセスしてください。

## 検証用データセットの変更
dev.jsonlの形式になっていれば任意のデータセットで検証ができます。具体的には以下のキーを含むデータセットは、コードの変更なしに推論可能です。
- ID: 問題のID
- category: 問題のカテゴリ。dev.jsonlの場合は学年に相当。
- unit: 単元。dev.jsonlの場合は教科書の各章タイトルに相当。
- problem: 実際の問題。モデルに入力される文章。
- solution: 模範解答(数値のみ)

inputディレクトリにmath500-jaデータセットを上記の形式に変換するスクリプト"format_math500.py"があります。
```
python3 format_math500.py --input_file hoge.jsonl --output_file fuga.jsonl
```
- input_file: 変換対象のファイル
- output_file: 変換後の保存先

## モデルの変更
"1. モデルのダウンロード"にて、"--model_name"を変更することで任意のモデルで推論が可能と述べましたが、このチュートリアルで使用したモデルと異なるユーザーのモデルを利用する際には、defファイルの以下の箇所(28行目)で変更が必要です。

```
mkdir -p ${SINGULARITY_ROOTFS}/app/models/HayatoHongoEveryonesAI/
```
この"HayatoHongoEveryonesAI"を利用したいモデルのユーザー名に置き換えるか、新たにmkdirでディレクトリを作成するコマンドを追加してください。

## 特定の revision のモデルをダウンロード
"download_model.py"ではmainブランチに相当するモデルがダウンロードされます。それ以外のブランチ(HF上ではrevisionと呼称)のモデルをダウンロードする場合は、"download_model.py"の25行目をコメントアウトし、revisionを取得したいものにを置き換えてください。

revision名はHFのサイト上において、モデルページから対象のブランチをにアクセスし、そのURLから取得できます。

## データセットのダウンロード
"download_dataset.py"を実行することで、HuggingFace上の任意の公開データセットをjsonファイルで保存します。
```
python3 download_dataset.py --dataset_name username/dataset_name
```

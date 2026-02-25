## create_QA

### 概要

本ディレクトリでは、問題・解答・およびそれらの検証結果を含むデータセットを生成します。

生成されたデータセットは、Hugging Face Hub にアップロードされることを前提としています。

本実装は、

- 問題生成
- 解答生成
- LLM を用いた解答検証

を一括で連続実行する最小構成の実装です。  
検証のみを行うツールではありません。

---

### できること / できないこと

#### できること
- 問題の生成
- 解答の生成
- LLM を用いた解答検証
- 上記をまとめたデータセットの作成
- Hugging Face Hub へのデータセットアップロード

#### できないこと（本ディレクトリ単体では不可）
- 検証のみの単独実行
- 問題生成のみ、または解答生成のみの単独実行

※ 上記については、別ブランチに参考実装があります（後述）。

---

### 主な変更・調整ポイント

生成件数やモデルはコマンドラインから変更できます。  
以下の項目を変更する場合は、記載のファイルを編集してください。

- カテゴリ・ジャンル  
  `category.py`

- プロンプト  
  `llmjudge.py` の前半部分

- 難易度  
  `llmjudge.py` 内の `"difficulty"` パラメータ

---

### 使用手順

#### 0. 環境準備

本ディレクトリへ移動します。

```bash
cd ./create_QA/
````

Python 仮想環境を作成し、依存関係をインストールします。

```bash
uv venv
source .venv/bin/activate
uv sync
```

---

#### 1. モデルのダウンロード

使用するモデルを事前にローカルへダウンロードします。

```bash
python download_model.py --model_name openai/gpt-oss-20b
```

モデル名は用途に応じて変更可能です。

---

#### 2. データセット生成（問題作成・解答作成・検証）

Hugging Face Hub へアップロードするため、
Hugging Face のアクセストークンを環境変数として設定します。

```bash
export HF_TOKEN="your_token_here"
```

以下のジョブを実行すると、

* 問題生成
* 解答生成
* 解答の検証

の 3 ステップが連続して実行され、
結果が 1 つのデータセットとして作成され、HuggingFaceにアップロードされます。

スクリプト名は `llmjudge-1gpu.sh` ですが、
検証のみを行うものではありません。

```bash
# 例：
# 問題数 100 件、リポジトリ名 qa_verify_100_test としてデータセットを作成して、HuggingFaceにアップロードする
qsub -v HF_TOKEN,NUM_QUESTIONS=100,REPO_ID=qa_verify_100_test ./llmjudge-1gpu.sh
```

※ 本手順は qsub（PBS / SGE）環境を前提としています。
※ ローカル実行の場合は `llmjudge-1gpu.sh` を直接実行してください。

※ `REPO_ID` が Hugging Face 上に存在しない場合は、新規に作成されます
（トークンに作成権限が必要です）。

---

### 実行時間の目安

* モデル: gpt-oss-20B
* GPU: H200（VRAM 141GB）× 1
* 約 10,000 件あたり 1 時間

※ 実行時間はモデル・GPU・設定により変動します。

---

### パラメータについて

必要に応じて、以下の環境変数を変更できます。

* `NUM_QUESTIONS`
  生成する問題数

* `REPO_ID`
  Hugging Face 上に作成されるデータセットのリポジトリ名

---

### 環境変数一覧（必須）

事前に設定しておく必要があります。

* `HF_TOKEN`
  Hugging Face のアクセストークン


---

### 備考

本ディレクトリは最小実装です。

以下の用途については、別ブランチに参考実装があります。

* 問題と解答の生成のみ
* 検証のみの実行

参考ブランチ:
[https://github.com/HayatoHongo/LLM-jp-2025competetion-victory/tree/rick-createqa-verifyqa/create_QA](https://github.com/HayatoHongo/LLM-jp-2025competetion-victory/tree/rick-createqa-verifyqa/create_QA)


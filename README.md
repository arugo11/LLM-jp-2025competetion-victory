## このディレクトリについて

<<<<<<< HEAD
# 注意！
`main`を直接編集せず、自分のブランチを作成してください！

## 使い方

各フォルダの詳しい使い方については、**それぞれの `README.md` を参照してください。**

## ディレクトリ構成

### `ft-llm-2026/`

チュートリアルで使用するルートフォルダです。

---

### `inference/`

本リポジトリの**提出用の中核となるフォルダ**です。

* 推論（Inference）の評価を行います
* 現在は**多数決推論システム**を採用しています
* 提出用の **Singularity イメージ（`.sif` ファイル）** もここに配置されています

---

### `math-eval/`

推論結果の**正解率（Accuracy）を計算**するためのフォルダです。

* `inference/` で出力された正解データをもとに評価を行います

---

### `openr1-training/`

**SFT（Supervised Fine-Tuning）** を行うためのフォルダです。

* フォルダ名に `R1` とありますが、**GRPO は使用していません**
* 混同しないよう注意してください
=======
AllenAIのopen-instructを使用しGRPO学習を行います。

## 環境構築コマンド
`cd /home/your_account_name/LLM-jp-2025competetion-victory/open-instruct/installers/abci`

`bash run_setup.sh ${HOME}/LLM-jp-2025competetion-victory/env`

## コマンドライン引数

## デバッグランを走らせる場合、single gpu mode

## 本番学習ラン実行
`cd open-instruct`

`qsub scripts/abci/train/qsub_grpo_fast.sh`
>>>>>>> f12eda6 (Convert open-instruct submodule to regular directory)

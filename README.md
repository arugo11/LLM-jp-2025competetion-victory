# LLM-jp-2025competetion-victory
LLM-jpの2025年度の数学むけタスクのコンペティションのリポジトリです。

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

<<<<<<< rlvr-math
### `open-instruct/`

**GRPO学習** を行うためのフォルダです。

* AllenAIの [open-instruct](https://github.com/allenai/open-instruct) をベースに改変
* 詳細は `open-instruct/README.md` を参照してください

---

=======
>>>>>>> main
### `openr1-training/`

**SFT（Supervised Fine-Tuning）** を行うためのフォルダです。

* フォルダ名に `R1` とありますが、**GRPO は使用していません**
* 混同しないよう注意してください

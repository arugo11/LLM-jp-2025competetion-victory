# 概要
LLMのマージを行うことが出来ます。

# 実行方法
qsub llm-merge.sh

# 設定の変更
main.pyを編集
- BASE_MODEL_DIR 親ディレクトリ
- MODEL その中のマージするモデルのリスト
- METHOD マージ手法の選択 ('linear', 'ties', 'dare_ties', 'dare_linear')
- OUTPUT_PATH マージ後のモデルの出力ディレクトリ
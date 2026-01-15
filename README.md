## このディレクトリについて

AllenAIのopen-instructを使用しGRPO学習を行います。

## 環境構築コマンド
`cd /home/your_account_name/LLM-jp-2025competetion-victory/open-instruct/installers/abci`

`bash run_setup.sh ${HOME}/LLM-jp-2025competetion-victory/env`

## コマンドライン引数

## デバッグランを走らせる場合、single gpu mode

## 本番学習ラン実行
`cd open-instruct`

`qsub scripts/abci/train/qsub_grpo_fast.sh`

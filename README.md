## このディレクトリについて

AllenAIのopen-instructを使用しGRPO学習を行います。

## 環境構築コマンド
`cd /home/your_account_name/LLM-jp-2025competetion-victory/open-instruct/installers/abci`

`bash run_setup.sh ${HOME}/LLM-jp-2025competetion-victory/env`

## コマンドライン引数

## デバッグランを走らせる場合、single gpu mode （未検証）

変更箇所

`#PBS -l select=1:ngpus=1`

`export CUDA_VISIBLE_DEVICES=0`

`export RAY_OVERRIDE_NUM_GPUS=1`

`ray start --head --port=${RAY_NODE_PORT} --dashboard-host=0.0.0.0 --num-gpus=1`

  `--deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    ...
    --vllm_sync_backend gloo \
    ...
    --single_gpu_mode \`

## 本番学習ラン実行
`cd open-instruct`

`qsub scripts/abci/train/qsub_grpo_fast.sh`

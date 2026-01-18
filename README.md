## このディレクトリについて

AllenAIのopen-instructを使用しGRPO学習を行います。

データ供給とアルゴリズムの連携。
データ供給側が動的に中難易度帯のサンプル群を用意し、目的関数側は回答が高分散になるようなサンプル群と相性のよい、標準偏差で正規化しないアドバンテージの計算により、損得による報酬信号の抑制を抑えています。

モデル長文化のダイナミズムに関して。
vLLMアクターが、最大生成長を超えて出力しようとした場合は強制終了されますが、強制終了されたこと自体に罰則を与えず、報酬サンプル群からそのサンプルをフィルタリングします。なぜなら、その長文回答の仕方は良かったのか悪かったのか判定できないからです。したがって、過度な長文化の抑制を避けています。

目的関数における損得の集約の仕方は、元祖GRPOのグループ内サンプル平均集約ではなく、バッチ内トークン平均集約を用いています。そのため、vLLMが長い系列長を生成すればするほど、その分、正解ならばアドバンテージが強調されます。なぜなら、アドバンテージはトークンごとの項に掛かる重みとして働く一方で、サンプル平均では、サンプル長による正規化を経るため、系列長の差が均され、長文サンプルの更新寄与が相対的に弱まるからです。同様に、長文かつ不正解ならば負のアドバンテージも強まります（Aの定義により負になりうる）。

参考
詳細については、OLMo-3論文4.4.1節を参照してください。

## 環境構築コマンド
```
cd /home/your_account_name/LLM-jp-2025competetion-victory/open-instruct/installers/abci
bash run_setup.sh ${HOME}/LLM-jp-2025competetion-victory/env
```

## コマンドライン引数

ほとんどの引数はgrpo_fast.pyのArgsクラスに定義されています。
バッチサイズ = num_unique_prompts_rollout * num_samples_per_prompt_rollout = 360
```
python open_instruct/grpo_fast.py \
    --dataset_mixer_list HayatoHongoEveryonesAI/qa_verify_2M_v5 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list HayatoHongoEveryonesAI/qa_verify_2M_v5 0.1 \
    --dataset_mixer_eval_list_splits train \
    --dataset_skip_cache \
    --max_prompt_token_length 1024 \
    --response_length 7168 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 12 \
    --num_samples_per_prompt_rollout 30 \
    --model_name_or_path HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-v5-2 \
    --stop_strings "</answer>" \
    --apply_verifiable_reward true \
    --remap_verifier qa_10k=math-verify \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 1e-6 \
    --total_episodes 360000 \
    --deepspeed_stage 3 \
    --num_epochs 1 \
    --num_learners_per_node 2 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --vllm_num_engines 6 \
    --vllm_gpu_memory_utilization 0.25 \
    --beta 0.00 \
    --load_ref_policy false \
    --seed 3 \
    --local_eval_every 100 \
    --vllm_sync_backend nccl \
    --vllm_enable_prefix_caching \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --save_freq 100 \
    --checkpoint_state_dir output/grpo_fast_checkpoint_state \
    --checkpoint_state_freq 100 \
    --push_to_hub \
    --hf_entity HayatoHongoEveryonesAI \
    --hf_repo_id open-instruct-grpo-fast \
    --system_prompt_override_file scripts/train/debug/cute_debug_system_prompt.txt \
    --active_sampling \
    --filter_zero_std_samples \
    --async_steps 4 \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --no_resampling_pass_rate 0.9 \
    --clip_higher 0.272 \
    --mask_truncated_completions \
    --with_tracking \
    --wandb_entity hongo-hayato-6281k-university-of-tokyo \
    --wandb_project_name open-instruct-grpo-fast \
    --verbose
```

## 学習実行
```
cd open-instruct
qsub scripts/abci/train/qsub_grpo_fast.sh
```

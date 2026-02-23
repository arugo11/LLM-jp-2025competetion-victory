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

バッチサイズ = num_unique_prompts_rollout * num_samples_per_prompt_rollout = 256
```
python open_instruct/grpo_fast.py \
    --dataset_mixer_list HayatoHongoEveryonesAI/qa_verify_cot_new_6M_v6 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_skip_cache \
    --max_prompt_token_length 1024 \
    --response_length 7168 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 32 \
    --model_name_or_path HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-expand-checkpoint-1900 \
    --apply_verifiable_reward true \
    --remap_verifier qa_10k=math-verify \
    --temperature 1.018 \
    --ground_truths_key ground_truth \
    --chat_template_name math_problem_with_boxed \
    --learning_rate 1e-6 \
    --total_episodes 441600 \
    --deepspeed_stage 3 \
    --num_epochs 1 \
    --num_learners_per_node 3 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --vllm_num_engines 5 \
    --vllm_gpu_memory_utilization 0.65 \
    --beta 0.00 \
    --load_ref_policy false \
    --seed 3 \
    --vllm_sync_backend nccl \
    --vllm_enable_prefix_caching \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --save_freq 100 \
    --local_eval_every -1 \
    --checkpoint_state_dir output/grpo_fast_12b_checkpoint_state \
    --checkpoint_state_freq 100 \
    --push_to_hub \
    --hf_entity HayatoHongoEveryonesAI \
    --hf_repo_id open-instruct-grpo-fast \
    --active_sampling \
    --filter_zero_std_samples \
    --async_steps 4 \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --clip_higher 0.272 \
    --mask_truncated_completions \
    --with_tracking \
    --wandb_entity hongo-hayato-6281k-university-of-tokyo \
    --wandb_project_name open-instruct-grpo-fast \
    --verbose
```

## 温度スケジュール

ステップ1000以降、温度スィープを実施し、以下のように確定しました。
小数点第1位のオーダーでは、truncation maskによってバッチサイズの維持が困難になったり、生成出力長が両極の長さに割れてしまったりすることで、モデルの学習が崩壊しました。

| ステップ範囲 | 温度 |
|---|---|
| 0〜999 | 1.0 |
| 1000〜1300 | 1.01 |
| 1300〜1400 | 1.02 |
| 1400〜1500 | 1.015 |
| 1500〜1600 | 1.02 |
| 1600〜1700 | 1.015 |
| 1700〜1725 | 1.018 |

## 1️⃣ HF_TOKEN の設定

```bash
# トークンを環境変数に設定
export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# 設定内容を確認
echo $HF_TOKEN
# → hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX が表示される
```


```bash
cd \~/.cache/huggingface/      # トークンが保存されるディレクトリ
ls
# token   ← ここに現在使用中のトークンが書かれています
cat token
# → hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX が表示されれば OK
```

### export で設定したHF_TOKENと、キャッシュに設定されているtokenが一致しない場合

```bash
# Hugging Face CLI からログアウト
hf auth logout

# 再度ログイン（ブラウザで認証コードを入力）
hf auth login
```


---

## 2️⃣ Hugging Face アクセス確認

```bash
# 必要なライブラリをインストール
pip install -U huggingface_hub

# モデル情報を取得できるかテスト
python -c "from huggingface_hub import HfApi; \
print(HfApi().model_info('HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-expand-checkpoint-1900'))"
# 情報が出力されればトークンは正しく認証されています
```


---

## 4️⃣ wandb にログイン

```bash
# wandb のログイン（APIキーをブラウザに貼り付け）
wandb login
```

---

## ✅ 確認チェックリスト

- [ ] `HF_TOKEN` が `echo $HF_TOKEN` で正しく表示される  
- [ ] `~/.cache/huggingface/token` に同一のトークンが保存されている  
- [ ] `python -c …model_info…` が成功し、モデル情報が取得できる  
- [ ] 必要なら `hf auth logout && hf auth login` を実行した  
- [ ] `wandb login` が成功した

## 学習実行

```bash
cd open-instruct
qsub scripts/abci/train/qsub_grpo_fast_12b.sh
```

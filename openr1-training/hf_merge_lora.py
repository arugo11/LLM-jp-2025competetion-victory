import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login

# ---------------------------------------------------------
# 設定項目
# ---------------------------------------------------------
# ここにHugging Faceのトークンを入力してください (Read/Write権限のあるもの)
hf_token = os.getenv("HF_TOKEN")  # 環境変数からHugging Faceのトークンを取得

base_model_id = "HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-test-checkpoint-140"  # ベースモデル
adapter_model_id = "HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long" # LoRAアダプター
output_dir = "./merged_model"                        # 保存先

# 事前にログイン処理を行います（これにより認証エラーを防げます）
login(token=hf_token)

# ---------------------------------------------------------
# 1. ベースモデルの読み込み
# ---------------------------------------------------------
print(f"Loading base model: {base_model_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    token=hf_token,  # トークン指定
)

# トークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    token=hf_token   # トークン指定
)

# ---------------------------------------------------------
# 2. LoRAアダプターの統合
# ---------------------------------------------------------
print(f"Loading LoRA adapter: {adapter_model_id}")
# アダプター自体もPrivateリポジトリにある場合はトークンが必要です
model = PeftModel.from_pretrained(
    base_model,
    adapter_model_id,
    token=hf_token   # トークン指定
)

# ---------------------------------------------------------
# 3. マージ実行 (Merge and Unload)
# ---------------------------------------------------------
print("Merging model...")
model = model.merge_and_unload()

# ---------------------------------------------------------
# 4. ローカルへの保存
# ---------------------------------------------------------
# print(f"Saving merged model to: {output_dir}")
# model.save_pretrained(output_dir, safe_serialization=True)
# tokenizer.save_pretrained(output_dir)

# ---------------------------------------------------------
# 5. (オプション) Hugging Face Hubへアップロード
# ---------------------------------------------------------
# アップロードしたい場合は以下のコメントアウトを外してください
repo_id = "HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long-merged"
print(f"Pushing to Hub: {repo_id}")

model.push_to_hub(repo_id, token=hf_token, safe_serialization=True)
tokenizer.push_to_hub(repo_id, token=hf_token)

print("Done! 🎉")
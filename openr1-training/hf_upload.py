import os
from huggingface_hub import HfApi

# スクリプトがあるディレクトリを基準とした相対パスを設定
hf_token = os.getenv("HF_TOKEN")  # 環境変数からHugging Faceのトークンを取得
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "/groups/gch51701/Team025/llm-jp-4-8b-instruct-sft-long-v5"
local_dir = os.path.join(script_dir, relative_path)

# アップロード先のRepository IDを指定 (ユーザー名/リポジトリ名)
# ※ 必要に応じて変更してください
repo_id = "HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-long-v5"

def upload_checkpoint():
    if not os.path.exists(local_dir):
        print(f"Error: Directory not found: {local_dir}")
        return

    api = HfApi(token=hf_token)

    # リポジトリが存在しない場合は作成 (private=True で作成)
    print(f"Creating repository '{repo_id}' if it doesn't exist...")
    try:
        api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create repository (it might already exist or require different permissions). Error: {e}")

    # フォルダのアップロード
    print(f"Starting upload from '{local_dir}' to '{repo_id}'...")
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            # 必要に応じて除外するファイルパターンを指定できます (例: optim statesなど)
            ignore_patterns=["checkpoint-*"],
            # ignore_patterns=["*.pth", "*.pt", "optimizer.pt"],
        )
        print("Upload completed successfully!")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    upload_checkpoint()

import os
from datasets import load_dataset, concatenate_datasets

# ==========================================
# 設定部分
# ==========================================

# 1. マージしたい6つのデータセット名をリストに入れる
SOURCE_DATASETS = [
    "team-victory/qa_verify_50k_1",
    "team-victory/qa_verify_50k_2",
    "team-victory/qa_verify_50k_3",
    "team-victory/qa_verify_50k_4",
    "team-victory/qa_verify_50k_5",
    "team-victory/qa_verify_50k_6",
]

# 2. 新しく作成するアップロード先のデータセット名 (ユーザー名/リポジトリ名)
NEW_REPO_ID = "team-victory/qa_verify_255k_2"

# 3. トークン設定 (huggingface-cli loginしていれば不要ですが、明示的に書く場合)
# HF_TOKEN = "your_write_token_here" 

# ==========================================
# 処理部分
# ==========================================

def merge_and_upload():
    loaded_datasets = []
    
    print(f"合計 {len(SOURCE_DATASETS)} 個のデータセットを読み込み中...")

    # 各データセットをロード
    for i, dataset_name in enumerate(SOURCE_DATASETS):
        try:
            # 提示されたYAMLにsplit: trainとあるため、trainを指定して読み込みます
            ds = load_dataset(dataset_name, split="train")
            print(f"[{i+1}/{len(SOURCE_DATASETS)}] 完了: {dataset_name} ({len(ds)} rows)")
            loaded_datasets.append(ds)
        except Exception as e:
            print(f"エラーが発生しました ({dataset_name}): {e}")
            return

    # データセットの結合
    print("データセットを結合しています...")
    merged_dataset = concatenate_datasets(loaded_datasets)
    print(f"結合完了。合計行数: {len(merged_dataset)}")

    # IDの再採番 (重要)
    # 個別のデータセットでidが0から始まっている場合、単純結合するとidが重複するため、
    # 通し番号に振り直します。
    print("IDを再採番しています...")
    merged_dataset = merged_dataset.map(
        lambda example, idx: {"id": idx},
        with_indices=True
    )

    # アップロード
    print(f"Hugging Face Hubへアップロード中: {NEW_REPO_ID}")
    merged_dataset.push_to_hub(NEW_REPO_ID, private=False) # private=Trueにすると非公開になります
    
    print("すべての処理が完了しました！")

if __name__ == "__main__":
    merge_and_upload()
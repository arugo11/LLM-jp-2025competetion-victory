import yaml
import os
import subprocess
from typing import List, Dict, Any

# ==========================================
# 設定セクション
# ==========================================

# 1. ローカルにあるモデルの親ディレクトリ（絶対パスまたは相対パス）
# 例: ファインチューニングの出力フォルダなど
BASE_MODEL_DIR = "../inference/models/HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-v5-2"

# 2. 上記ディレクトリ内にあるマージしたいチェックポイントのフォルダ名
CHECKPOINT_DIRS = [
    "checkpoint-400",
    "checkpoint-1200",
    "checkpoint-1600",
    "checkpoint-1960",
]

# 3. マージ手法の選択 ('linear', 'ties', 'dare_ties', 'dare_linear')
METHOD = "dare_ties"

# 4. 出力先ディレクトリ
OUTPUT_PATH = "../inference/models/HayatoHongoEveryonesAI"

# 5. 生成するYAMLファイル名
CONFIG_FILENAME = "merge_config_local.yaml"

# ==========================================
# コンフィグ生成ロジック
# ==========================================

def create_config(method: str, base_dir: str, checkpoints: List[str]) -> Dict[str, Any]:
    """
    ローカルパスを結合してmergekit用の設定を作成します。
    """
    
    # チェックポイントのフルパスリストを作成
    model_paths = [os.path.join(base_dir, cp) for cp in checkpoints]
    
    # mergekitの 'base_model' 設定用
    # 通常、モデル構造(config.json)やトークナイザーを参照するために使われます。
    # 親ディレクトリ(BASE_MODEL_DIR)直下にモデルファイルがない場合(チェックポイントのみの場合)、
    # 安全のため「リストの最初のチェックポイント」をベースとして扱います。
    # もし親ディレクトリ自体が有効なモデルなら base_model_path = base_dir としてください。
    base_model_path = model_paths[0] 

    print(f"Using {base_model_path} as the base model definition.")

    config = {
        "base_model": base_model_path,
        "merge_method": method,
        "dtype": "float16",
    }

    models_config = []

    if method == "linear":
        # Linear: 単純平均
        weight = 1.0 / len(model_paths)
        for path in model_paths:
            models_config.append({
                "model": path,
                "parameters": {"weight": 1.0}
            })
            
    elif method == "ties":
        # TIES: ベースモデルとの差分計算が必要
        
        # 基準としてベースモデル(ここでは最初のチェックポイント)を追加
        models_config.append({
            "model": base_model_path,
            "parameters": {"density": 1.0, "weight": 1.0}
        })
        
        for path in model_paths:
            # ベースモデル自体は重複して追加しない、または追加してもweight管理が必要
            # ここではシンプルにリストにある全てのチェックポイントをマージ対象とします
            if path == base_model_path:
                continue 
                
            models_config.append({
                "model": path,
                "parameters": {
                    "density": 0.5,
                    "weight": 1.0
                }
            })

    elif method in ["dare_ties", "dare_linear"]:
        # DARE
        models_config.append({
            "model": base_model_path,
            "parameters": {"weight": 1.0}
        })
        
        for path in model_paths:
            if path == base_model_path:
                continue

            models_config.append({
                "model": path,
                "parameters": {
                    "weight": 1.0,
                    "density": 0.2  # DARE推奨の低density
                }
            })
            
    else:
        raise ValueError(f"Unknown method: {method}")

    config["models"] = models_config
    return config

# ==========================================
# 実行処理
# ==========================================

def main():
    # パス存在確認
    if not os.path.exists(BASE_MODEL_DIR):
        print(f"Error: Base directory '{BASE_MODEL_DIR}' does not exist.")
        return

    print(f"Generating config using checkpoints in: {BASE_MODEL_DIR}")
    
    config_data = create_config(METHOD, BASE_MODEL_DIR, CHECKPOINT_DIRS)
    
    with open(CONFIG_FILENAME, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, sort_keys=False)
    
    print(f"Config saved to {CONFIG_FILENAME}")
    print("Running mergekit...")

    cmd = [
        "mergekit-yaml",
        CONFIG_FILENAME,
        OUTPUT_PATH,
        "--allow-crimes",
        "--copy-tokenizer",
        "--cuda"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\nMerge completed successfully! Model saved to: {OUTPUT_PATH}")
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred during merging: {e}")

if __name__ == "__main__":
    main()
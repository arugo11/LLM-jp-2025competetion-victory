# Hugging Faceからデータセットをダウンロードするスクリプト
import argparse
from pathlib import Path
import json
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="team-victory/qa_verify_10k_test",
        help="dataset name on Hugging Face Hub",
    )
    args = parser.parse_args()

    # データセットのダウンロード
    # load_datasetのchace_dirで保存先を指定
    model_path = Path("./datasets") / args.dataset_name.replace("/", "-")
    dataset = load_dataset(args.dataset_name, cache_dir=str(model_path), split="test")

    print(f"Dataset downloaded to: {model_path}")

    # json形式で保存する
    output_path = "./input/math500-ja.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset):
            # idを振り直す
            item["id"] = i
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
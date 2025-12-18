# Hugging Faceからデータセットをダウンロードするスクリプト
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="team-victory/qa_verify_10k_test",
        help="dataset name on Hugging Face Hub",
    )
    args = parser.parse_args()

    model_path = snapshot_download(
        repo_id=args.dataset_name,
        repo_type="dataset",
        local_dir=Path("datasets") / args.dataset_name,
        local_dir_use_symlinks=False,
    )

    print(f"Dataset downloaded to: {model_path}")


if __name__ == "__main__":
    main()
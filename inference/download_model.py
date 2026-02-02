import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-expand-checkpoint-1900",
        help="Model name on Hugging Face Hub",
    )
    args = parser.parse_args()

    # 特定のリビジョンをダウンロードしたいときは
    # 引数のrevisionで指定する
    # 例えば、"grpo_fast__3__1766613992"というリビジョンを指定する
    # リビジョンはモデルリポジトリのリンクの"tree"以降を見るとわかる
    model_path = snapshot_download(
        repo_id=args.model_name,
        local_dir=Path("models") / (args.model_name + "-12B-1000"),
        local_dir_use_symlinks=False,
        # revision="grpo_fast__3__1769920398",
    )

    print(f"Model downloaded to: {model_path}")


if __name__ == "__main__":
    main()
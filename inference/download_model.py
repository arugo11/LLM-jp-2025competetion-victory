import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="llm-jp/llm-jp-3.1-1.8b-instruct4",
        help="Model name on Hugging Face Hub",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision to download",
    )
    parser.add_argument(
        "--local_dir",
        type=Path,
        default=None,
        help="Optional local output directory for the downloaded snapshot",
    )
    args = parser.parse_args()

    local_dir = args.local_dir or (Path("models") / args.model_name)

    model_path = snapshot_download(
        repo_id=args.model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision=args.revision,
    )

    print(f"Model downloaded to: {model_path}")


if __name__ == "__main__":
    main()

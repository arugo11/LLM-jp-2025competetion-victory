import argparse
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push JSONL dataset to Hugging Face.")
    parser.add_argument("--jsonl", required=True, type=str, help="Input JSONL path")
    parser.add_argument("--repo-id", required=True, type=str, help="HF repo id to push")
    parser.add_argument(
        "--split",
        default="train",
        type=str,
        help="Split name to use when pushing",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        type=str,
        help="HF token (optional, defaults to env HF_TOKEN if set)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF token not provided (set HF_TOKEN or --hf-token).")

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_file():
        raise SystemExit(f"JSONL not found: {jsonl_path}")

    dataset = load_dataset("json", data_files=jsonl_path.as_posix(), split=args.split)
    dataset_dict = DatasetDict()
    dataset_dict[args.split] = Dataset.from_list(list(dataset))
    dataset_dict.push_to_hub(args.repo_id, token=token)
    print(f"Pushed dataset to {args.repo_id} (split={args.split})")


if __name__ == "__main__":
    main()

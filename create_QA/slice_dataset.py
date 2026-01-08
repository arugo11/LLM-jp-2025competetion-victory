import argparse
import os
from pathlib import Path

from datasets import load_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice a HF dataset and save to JSONL.")
    parser.add_argument("--repo-id", required=True, type=str, help="Hugging Face repo id")
    parser.add_argument("--split", default="train", type=str, help="Dataset split")
    parser.add_argument("--start", required=True, type=int, help="Start index (inclusive)")
    parser.add_argument("--end", required=True, type=int, help="End index (exclusive)")
    parser.add_argument("--out", required=True, type=str, help="Output JSONL path")
    parser.add_argument(
        "--hf-token",
        default=None,
        type=str,
        help="HF token (optional, defaults to env HF_TOKEN if set)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    split = f"{args.split}[{args.start}:{args.end}]"

    ds = load_dataset(args.repo_id, split=split, token=token)
    ds.to_json(out_path.as_posix(), orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(ds)} rows to {out_path}")


if __name__ == "__main__":
    main()

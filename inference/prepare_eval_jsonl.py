import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a JSONL file for local evaluation by adding deterministic IDs."
    )
    parser.add_argument("--input_path", type=Path, required=True, help="Source JSONL path")
    parser.add_argument("--output_path", type=Path, required=True, help="Prepared JSONL path")
    parser.add_argument(
        "--id_prefix",
        type=str,
        default="final",
        help="Prefix used when synthesizing missing IDs",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="Starting index used for synthesized IDs",
    )
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with args.input_path.open("r", encoding="utf-8") as src, args.output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for offset, line in enumerate(src):
            item = json.loads(line)
            item.setdefault("id", f"{args.id_prefix}-{args.start_index + offset:06d}")
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

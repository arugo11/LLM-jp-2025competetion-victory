#!/usr/bin/env python3
"""
Decode a JSONL file (with escaped characters) into a pretty-printed UTF-8 text file.

Usage:
    python scripts/decode_jsonl.py --input path/to/input.jsonl --output path/to/output.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def decode_jsonl(input_path: Path, output_path: Path) -> None:
    """Read a JSONL file and write each record as pretty-printed JSON."""
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src, 1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            json.dump(record, dst, ensure_ascii=False, indent=2)
            dst.write("\n\n")
            if idx % 10 == 0:
                dst.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to the JSON Lines file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination file for decoded output (defaults to <input>.decoded.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input.with_suffix(args.input.suffix + ".decoded.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decode_jsonl(args.input, output_path)
    print(f"Decoded {args.input} -> {output_path}")


if __name__ == "__main__":
    main()

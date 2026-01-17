#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return "\n".join(_safe_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _clip_text(text: str, limit: int) -> str:
    if not text:
        return ""
    return text[:limit]


def _collect_parquet_paths(validation_dir: Path) -> list[Path]:
    if not validation_dir.exists():
        raise FileNotFoundError(f"validation_dir not found: {validation_dir}")
    paths = sorted(validation_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under: {validation_dir}")
    return paths


def _load_updates(paths: list[Path]) -> dict[int, dict[str, Any]]:
    updates: dict[int, dict[str, Any]] = {}
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=2000):
            table = pa.Table.from_batches([batch])
            for row in table.to_pylist():
                row_id = int(row["id"])
                stderr_full = _safe_text(row.get("validation_stderr_full"))
                stdout_full = _safe_text(row.get("validation_stdout_full"))
                updates[row_id] = {
                    "is_valid": int(row.get("is_valid", -1)),
                    "validation_reason": _safe_text(row.get("validation_reason")),
                    "stderr": _clip_text(stderr_full, 2048),
                    "stdout": _clip_text(stdout_full, 2048),
                    "runtime_ms": int(row.get("runtime_ms", 0)),
                }
    return updates


def _batch_column(batch: dict[str, Any], name: str, default: Any) -> list[Any]:
    if name in batch:
        return list(batch[name])
    return [default] * len(batch["id"])


def merge_and_push(args: argparse.Namespace) -> None:
    if not args.validation_dir:
        raise SystemExit("--validation-dir is required.")
    validation_dir = Path(args.validation_dir)

    parquet_paths = _collect_parquet_paths(validation_dir)
    updates = _load_updates(parquet_paths)

    dataset = load_dataset(args.dataset_repo, split="train")
    existing_columns = set(dataset.column_names)

    def _apply_updates(batch: dict[str, Any]) -> dict[str, Any]:
        ids = batch["id"]
        current_is_valid = _batch_column(batch, "is_valid", -1)
        current_reason = _batch_column(batch, "validation_reason", None)
        current_stderr = _batch_column(batch, "stderr", None)
        current_stdout = _batch_column(batch, "stdout", None)
        current_runtime = _batch_column(batch, "runtime_ms", None)

        new_is_valid: list[int] = []
        new_reason: list[str | None] = []
        new_stderr: list[str | None] = []
        new_stdout: list[str | None] = []
        new_runtime: list[int | None] = []

        for idx, row_id in enumerate(ids):
            row_id_int = int(row_id)
            update = updates.get(row_id_int)
            if update:
                new_is_valid.append(int(update["is_valid"]))
                new_reason.append(update["validation_reason"] or "")
                new_stderr.append(update["stderr"] or "")
                new_stdout.append(update["stdout"] or "")
                new_runtime.append(int(update["runtime_ms"]))
            else:
                new_is_valid.append(int(current_is_valid[idx]))
                new_reason.append(
                    current_reason[idx]
                    if "validation_reason" in existing_columns
                    else None
                )
                new_stderr.append(
                    current_stderr[idx]
                    if "stderr" in existing_columns
                    else None
                )
                new_stdout.append(
                    current_stdout[idx]
                    if "stdout" in existing_columns
                    else None
                )
                new_runtime.append(
                    current_runtime[idx]
                    if "runtime_ms" in existing_columns
                    else None
                )

        return {
            "is_valid": new_is_valid,
            "validation_reason": new_reason,
            "stderr": new_stderr,
            "stdout": new_stdout,
            "runtime_ms": new_runtime,
        }

    dataset = dataset.map(_apply_updates, batched=True, batch_size=1000)
    dataset.push_to_hub(
        args.dataset_repo,
        split="train",
        commit_message=args.commit_message,
        max_shard_size=args.max_shard_size,
        private=args.private,
        revision=args.revision,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge validation shards and push to hub."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser(
        "merge_and_push", help="Merge validation shards and push to hub."
    )
    merge_parser.add_argument("--dataset-repo", required=True)
    merge_parser.add_argument("--validation-dir", required=True)
    merge_parser.add_argument("--commit-message", required=True)
    merge_parser.add_argument("--max-shard-size", default="1GB")
    privacy_group = merge_parser.add_mutually_exclusive_group()
    privacy_group.add_argument(
        "--private",
        action="store_true",
        help="Push as private dataset (default).",
    )
    privacy_group.add_argument(
        "--public",
        action="store_true",
        help="Push as public dataset.",
    )
    merge_parser.add_argument("--revision", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "merge_and_push":
        if not (args.private or args.public):
            args.private = True
        merge_and_push(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

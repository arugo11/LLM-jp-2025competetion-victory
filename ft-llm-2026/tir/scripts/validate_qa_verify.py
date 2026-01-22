#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from nemo_skills.code_execution.sandbox import get_sandbox

PYTHON_TAG_OPEN = "<PYTHON>"
PYTHON_TAG_CLOSE = "</PYTHON>"


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


def _normalize_generated_solution(value: Any) -> tuple[str, bool]:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace"), True
    if value is None:
        return "", False
    return str(value), False


def _extract_python_code(
    text: str, code_max_chars: int
) -> tuple[str | None, str | None]:
    # 最後の <PYTHON> タグを探す（最新/最も確実なコードブロックを優先）
    start = text.rfind(PYTHON_TAG_OPEN)
    if start == -1:
        return None, "no_python_tag"

    # その <PYTHON> タグの後に対応する </PYTHON> を探す
    end = text.find(PYTHON_TAG_CLOSE, start + len(PYTHON_TAG_OPEN))
    if end == -1:
        return None, "no_python_tag"

    code = text[start + len(PYTHON_TAG_OPEN) : end].strip()

    # 余分なタグチェック: このペアの後にまだタグがあるか
    remaining = text[end + len(PYTHON_TAG_CLOSE) :]
    if PYTHON_TAG_OPEN in remaining or PYTHON_TAG_CLOSE in remaining:
        return None, "multiple_python_tags"

    if len(code) > code_max_chars:
        return code, "code_too_long"
    return code, None


def _resolve_sandbox_host_port(
    host_arg: str | None, port_arg: int | None
) -> tuple[str, int]:
    host = os.environ.get("NEMO_SKILLS_SANDBOX_HOST") or host_arg or "127.0.0.1"
    port_env = os.environ.get("NEMO_SKILLS_SANDBOX_PORT")
    if port_env:
        port = int(port_env)
    elif port_arg is not None:
        port = port_arg
    else:
        port = 5000
    return host, port


def _get_stderr_from_exec(exec_dict: dict[str, Any]) -> tuple[str | None, str | None]:
    if "stderr" in exec_dict:
        return _safe_text(exec_dict.get("stderr")), None
    if "error" in exec_dict:
        return _safe_text(exec_dict.get("error")), None
    if "errors" in exec_dict:
        return _safe_text(exec_dict.get("errors")), None
    return None, "no_stderr_field"


@dataclass
class RuntimeStats:
    sample_size: int = 10000
    seed: int = 1337

    def __post_init__(self) -> None:
        self._samples: list[float] = []
        self._count: int = 0
        self._sum: float = 0.0
        self._rng = random.Random(self.seed)

    def add(self, value: float) -> None:
        self._count += 1
        self._sum += value
        if len(self._samples) < self.sample_size:
            self._samples.append(value)
            return
        idx = self._rng.randint(0, self._count - 1)
        if idx < self.sample_size:
            self._samples[idx] = value

    @property
    def avg(self) -> float:
        if self._count == 0:
            return 0.0
        return self._sum / self._count

    def p95(self) -> float:
        if not self._samples:
            return 0.0
        ordered = sorted(self._samples)
        idx = int(round((len(ordered) - 1) * 0.95))
        return float(ordered[idx])


@dataclass
class FailureSampler:
    max_items: int = 1000
    seed: int = 2025

    def __post_init__(self) -> None:
        self._samples: list[dict[str, Any]] = []
        self._count: int = 0
        self._rng = random.Random(self.seed)

    def add(self, item: dict[str, Any]) -> None:
        self._count += 1
        if len(self._samples) < self.max_items:
            self._samples.append(item)
            return
        idx = self._rng.randint(0, self._count - 1)
        if idx < self.max_items:
            self._samples[idx] = item

    def items(self) -> list[dict[str, Any]]:
        return list(self._samples)


class ParquetShardWriter:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int = 1000) -> None:
        self._path = path
        self._schema = schema
        self._batch_size = batch_size
        self._buffer: list[dict[str, Any]] = []
        self._writer = pq.ParquetWriter(str(path), schema)

    def write_row(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


async def _execute_in_sandbox(
    sandbox: Any,
    code: str,
    timeout: float,
    max_output_chars: int,
) -> tuple[dict[str, Any] | None, str | None, float]:
    started = time.monotonic()
    try:
        exec_dict, _ = await sandbox.execute_code(
            generated_code=code,
            language="python",
            timeout=timeout,
            max_output_characters=max_output_chars,
        )
        runtime_ms = (time.monotonic() - started) * 1000
        return exec_dict, None, runtime_ms
    except asyncio.TimeoutError as exc:
        runtime_ms = (time.monotonic() - started) * 1000
        return None, f"sandbox_timeout: {exc}", runtime_ms
    except Exception as exc:  # pragma: no cover - runtime errors in sandbox
        runtime_ms = (time.monotonic() - started) * 1000
        return None, f"sandbox_exception: {exc}", runtime_ms


def _iter_streaming_rows(
    dataset_repo: str,
    start_id: int,
    end_id: int,
) -> Iterable[dict[str, Any]]:
    filters = [("id", ">=", start_id), ("id", "<", end_id)]
    try:
        dataset = load_dataset(
            dataset_repo,
            split="train",
            streaming=True,
            filters=filters,
        )
    except TypeError:
        dataset = load_dataset(dataset_repo, split="train", streaming=True)
    for row in dataset:
        yield row


async def validate_range(args: argparse.Namespace) -> None:
    host, port = _resolve_sandbox_host_port(args.sandbox_host, args.sandbox_port)
    print(
        "validate_range config: "
        f"dataset_repo={args.dataset_repo} start_id={args.start_id} end_id={args.end_id} "
        f"timeout={args.timeout_seconds} max_output_chars={args.max_output_chars} "
        f"sandbox={host}:{port} code_max_chars={args.code_max_chars} "
        f"force={args.force}"
    )

    output_path = Path(args.output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("is_valid", pa.int64()),
            ("validation_reason", pa.string()),
            ("validation_stderr_full", pa.string()),
            ("validation_stdout_full", pa.string()),
            ("runtime_ms", pa.int64()),
        ],
    )
    writer = ParquetShardWriter(output_path, schema=schema, batch_size=1000)

    sandbox = get_sandbox(sandbox_type="local", host=host, port=port)

    processed = 0
    valid_count = 0
    invalid_count = 0
    reasons = Counter()
    runtime_stats = RuntimeStats()
    failure_samples = FailureSampler(max_items=1000)
    seen_ids: set[int] = set()
    expected_count = max(0, args.end_id - args.start_id)

    try:
        for row in _iter_streaming_rows(args.dataset_repo, args.start_id, args.end_id):
            try:
                row_id = int(row["id"])
            except Exception as exc:
                raise RuntimeError(f"Missing or invalid id in row: {exc}") from exc
            if row_id < args.start_id:
                continue
            if row_id >= args.end_id:
                continue
            if not args.force:
                try:
                    is_valid_value = int(row.get("is_valid", -1))
                except Exception:
                    is_valid_value = -1
                if is_valid_value != -1:
                    continue
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            processed += 1
            generated_solution, decode_error = _normalize_generated_solution(
                row.get("generated_solution"),
            )
            if decode_error:
                code = None
                reason = "decode_error"
            else:
                code, reason = _extract_python_code(
                    generated_solution, args.code_max_chars
                )

            if reason:
                is_valid = 0
                stdout_full = ""
                stderr_full = ""
                runtime_ms = 0
            else:
                exec_dict, exec_error, runtime_ms = await _execute_in_sandbox(
                    sandbox,
                    code or "",
                    args.timeout_seconds,
                    args.max_output_chars,
                )
                if exec_error:
                    is_valid = 0
                    reason = exec_error.split(":", 1)[0]
                    stderr_full = exec_error
                    stdout_full = ""
                else:
                    exec_dict = exec_dict or {}
                    process_status = _safe_text(exec_dict.get("process_status"))
                    stdout_full = _safe_text(exec_dict.get("stdout"))
                    stderr_full, missing_reason = _get_stderr_from_exec(exec_dict)
                    if process_status == "timeout":
                        is_valid = 0
                        reason = "sandbox_timeout"
                        stderr_full = stderr_full or ""
                    elif process_status == "error":
                        is_valid = 0
                        reason = "sandbox_error"
                        stderr_full = stderr_full or ""
                    elif missing_reason:
                        is_valid = 0
                        reason = missing_reason
                        stderr_full = ""
                    else:
                        stderr_full = stderr_full or ""
                        if stderr_full == "":
                            is_valid = 1
                            reason = ""
                        else:
                            is_valid = 0
                            reason = "sandbox_stderr"

            runtime_stats.add(runtime_ms)
            if is_valid == 1:
                valid_count += 1
            else:
                invalid_count += 1
                if reason:
                    reasons[reason] += 1

            record = {
                "id": row_id,
                "is_valid": int(is_valid),
                "validation_reason": reason or "",
                "validation_stderr_full": stderr_full,
                "validation_stdout_full": stdout_full,
                "runtime_ms": int(round(runtime_ms)),
            }
            writer.write_row(record)

            if is_valid == 0:
                failure_item = {
                    "id": row_id,
                    "reason": reason or "",
                    "stderr_head": _clip_text(stderr_full, 2048),
                    "stdout_head": _clip_text(stdout_full, 2048),
                    "runtime_ms": int(round(runtime_ms)),
                }
                failure_samples.add(failure_item)

            if processed % args.log_every == 0:
                reasons_top = ", ".join(
                    f"{name}={count}" for name, count in reasons.most_common(5)
                )
                print(
                    "progress: "
                    f"processed={processed} valid={valid_count} invalid={invalid_count} "
                    f"valid_rate={valid_count / processed if processed else 0.0:.4f} "
                    f"runtime_ms_avg={runtime_stats.avg:.2f} runtime_ms_p95={runtime_stats.p95():.2f} "
                    f"reasons_top={reasons_top}"
                )

            if expected_count and len(seen_ids) >= expected_count:
                break
    finally:
        writer.close()
        await sandbox.close()

    print(
        "summary: "
        f"processed={processed} valid={valid_count} invalid={invalid_count} "
        f"runtime_ms_avg={runtime_stats.avg:.2f} runtime_ms_p95={runtime_stats.p95():.2f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate QA verify code with NeMo sandbox."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate_range", help="Validate a range of ids."
    )
    validate_parser.add_argument("--dataset-repo", required=True)
    validate_parser.add_argument("--start-id", type=int, required=True)
    validate_parser.add_argument("--end-id", type=int, required=True)
    validate_parser.add_argument("--output-parquet", required=True)
    validate_parser.add_argument("--timeout-seconds", type=float, default=10.0)
    validate_parser.add_argument("--max-output-chars", type=int, default=4096)
    validate_parser.add_argument("--code-max-chars", type=int, default=20000)
    validate_parser.add_argument("--log-every", type=int, default=1000)
    validate_parser.add_argument("--sandbox-host", default=None)
    validate_parser.add_argument("--sandbox-port", type=int, default=None)
    validate_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run validation even if is_valid is already set.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "validate_range":
        asyncio.run(validate_range(args))
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

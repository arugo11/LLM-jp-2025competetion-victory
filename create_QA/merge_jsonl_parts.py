import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge JSONL parts with validation and manifest output."
    )
    parser.add_argument(
        "--pattern",
        required=True,
        type=str,
        help="Input file pattern with {index} placeholder",
    )
    parser.add_argument("--parts", required=True, type=int, help="Number of parts")
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output merged JSONL path",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=None,
        help="Expected total number of rows (optional)",
    )
    parser.add_argument(
        "--strict-json",
        action="store_true",
        help="Parse each line as JSON to detect corruption",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Write a manifest JSON with checksums and counts",
    )
    parser.add_argument(
        "--keep-partial",
        action="store_true",
        help="Keep a partial merged file on validation errors",
    )
    parser.add_argument(
        "--partial-suffix",
        type=str,
        default=".partial",
        help="Suffix for partial merged output",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing parts (still reports in manifest/errors)",
    )
    return parser.parse_args()


def _expected_count_for_part(index: int, total: int, parts: int) -> int:
    chunk = (total + parts - 1) // parts
    start = index * chunk
    end = min(start + chunk, total)
    return max(0, end - start)


def _load_json_line(line: str, path: Path, line_no: int) -> None:
    if not line.strip():
        raise ValueError(f"Blank line at {path}:{line_no}")
    try:
        json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def main() -> None:
    args = _parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    part_paths: list[Path] = []
    missing: list[str] = []
    for i in range(args.parts):
        path = Path(args.pattern.format(index=i))
        part_paths.append(path)
        if not path.is_file():
            missing.append(path.as_posix())

    if out_path in part_paths:
        print("ERROR: output path overlaps with input parts.", file=sys.stderr)
        sys.exit(1)

    part_info: list[dict[str, Any]] = []
    total_lines = 0
    merged_sha = hashlib.sha256()
    errors: list[str] = []

    if missing:
        for m in missing:
            errors.append(f"Missing part file: {m}")
        if not args.allow_missing:
            errors.append("Missing parts detected (allow with --allow-missing)")

    temp_out = out_path.with_name(out_path.name + ".tmp")
    with temp_out.open("wb") as out_f:
        for index, path in enumerate(part_paths):
            if not path.is_file():
                part_info.append(
                    {
                        "index": index,
                        "path": path.as_posix(),
                        "lines": 0,
                        "sha256": None,
                        "expected_lines": (
                            _expected_count_for_part(
                                index, args.expected_total, args.parts
                            )
                            if args.expected_total is not None
                            else None
                        ),
                        "missing": True,
                    }
                )
                continue
            part_sha = hashlib.sha256()
            part_lines = 0
            with path.open("rb") as in_f:
                for line_no, raw in enumerate(in_f, start=1):
                    part_sha.update(raw)
                    merged_sha.update(raw)
                    if args.strict_json:
                        _load_json_line(raw.decode("utf-8"), path, line_no)
                    out_f.write(raw)
                    part_lines += 1
            total_lines += part_lines

            expected_count = None
            if args.expected_total is not None:
                expected_count = _expected_count_for_part(
                    index, args.expected_total, args.parts
                )
                if part_lines != expected_count:
                    errors.append(
                        f"Part {index} expected {expected_count} lines, got {part_lines}"
                    )

            part_info.append(
                {
                    "index": index,
                    "path": path.as_posix(),
                    "lines": part_lines,
                    "sha256": part_sha.hexdigest(),
                    "expected_lines": expected_count,
                    "missing": False,
                }
            )

    if args.expected_total is not None and total_lines != args.expected_total:
        errors.append(
            f"Total expected {args.expected_total} lines, got {total_lines}"
        )

    has_errors = bool(errors)

    final_out = out_path
    if has_errors and args.keep_partial:
        final_out = out_path.with_name(out_path.name + args.partial_suffix)

    if has_errors and not args.keep_partial:
        temp_out.unlink(missing_ok=True)
    else:
        final_out.parent.mkdir(parents=True, exist_ok=True)
        temp_out.replace(final_out)

    manifest_path = args.manifest
    if manifest_path:
        manifest = {
            "pattern": args.pattern,
            "parts": args.parts,
            "expected_total": args.expected_total,
            "actual_total": total_lines,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed" if has_errors else "ok",
            "errors": errors,
            "missing_parts": missing,
            "parts_detail": part_info,
            "merged": {
                "path": final_out.as_posix(),
                "lines": total_lines,
                "sha256": merged_sha.hexdigest(),
            },
        }
        Path(manifest_path).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Merged {total_lines} lines into {final_out}")
    if args.manifest:
        print(f"Manifest written to {args.manifest}")

    if has_errors:
        print("ERROR: validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

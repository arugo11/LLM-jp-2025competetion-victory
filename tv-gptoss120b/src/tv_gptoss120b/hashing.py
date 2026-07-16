from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    entries = [
        {"path": str(item.relative_to(path)), "sha256": sha256_file(item)}
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    if not entries:
        raise ValueError(f"cannot hash an empty directory: {path}")
    return sha256_value(entries)

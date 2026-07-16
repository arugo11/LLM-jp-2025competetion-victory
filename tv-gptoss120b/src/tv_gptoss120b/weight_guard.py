from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import sha256_file, sha256_value


def _tensor_files(root: Path) -> dict[str, Path]:
    index = root / "model.safetensors.index.json"
    if index.exists():
        payload = json.loads(index.read_text(encoding="utf-8"))
        return {key: root / filename for key, filename in payload["weight_map"].items()}
    single = next(
        (
            candidate
            for candidate in (root / "model.safetensors", root / "adapter_model.safetensors")
            if candidate.exists()
        ),
        None,
    )
    if single is None:
        raise FileNotFoundError(f"no model or adapter safetensors found under {root}")
    from safetensors import safe_open

    with safe_open(single, framework="pt", device="cpu") as handle:
        return {key: single for key in list(handle.keys())}


def compare_weight_update(before: Path, after: Path, sample_keys: int = 80, points_per_key: int = 16) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    before_files = _tensor_files(before)
    after_files = _tensor_files(after)
    common = sorted(before_files.keys() & after_files.keys(), key=lambda key: sha256_value(key))
    if not common:
        raise ValueError("no common safetensors keys found")
    selected = common[:sample_keys]
    changed = []
    max_abs_diff = 0.0
    for key in selected:
        with safe_open(before_files[key], framework="pt", device="cpu") as handle:
            left = handle.get_tensor(key).reshape(-1)
        with safe_open(after_files[key], framework="pt", device="cpu") as handle:
            right = handle.get_tensor(key).reshape(-1)
        if left.shape != right.shape:
            raise ValueError(f"tensor shape changed unexpectedly: {key}: {left.shape} != {right.shape}")
        if left.numel() == 0:
            continue
        indexes = torch.linspace(0, left.numel() - 1, steps=min(points_per_key, left.numel())).long()
        diff = (left[indexes].float() - right[indexes].float()).abs()
        key_max = float(diff.max().item())
        max_abs_diff = max(max_abs_diff, key_max)
        if key_max > 0:
            changed.append(key)
    report = {
        "before": str(before.resolve()),
        "after": str(after.resolve()),
        "sampled_keys": len(selected),
        "changed_keys": len(changed),
        "changed_key_names": changed,
        "max_abs_diff": max_abs_diff,
    }
    if not changed:
        raise RuntimeError("weight update gate failed: sampled model weights did not change")
    return report


def compare_adapter_update(before: Path, after: Path) -> dict[str, Any]:
    """Prove that a saved PEFT adapter changed from its exact pre-training initialization."""
    import torch
    from safetensors import safe_open

    before_files = _tensor_files(before)
    after_files = _tensor_files(after)
    if before_files.keys() != after_files.keys():
        missing = sorted(before_files.keys() - after_files.keys())
        extra = sorted(after_files.keys() - before_files.keys())
        raise ValueError(f"adapter tensor key set changed: missing={missing[:5]}, extra={extra[:5]}")
    if not before_files:
        raise ValueError("adapter contains no safetensors keys")

    changed = []
    changed_elements = 0
    total_elements = 0
    max_abs_diff = 0.0
    for key in sorted(before_files):
        with safe_open(before_files[key], framework="pt", device="cpu") as handle:
            left = handle.get_tensor(key)
        with safe_open(after_files[key], framework="pt", device="cpu") as handle:
            right = handle.get_tensor(key)
        if left.shape != right.shape:
            raise ValueError(f"adapter tensor shape changed unexpectedly: {key}: {left.shape} != {right.shape}")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise RuntimeError(f"adapter weight update gate failed: non-finite tensor values in {key}")
        difference = (left.float() - right.float()).abs()
        key_changed = int(torch.count_nonzero(difference).item())
        total_elements += left.numel()
        changed_elements += key_changed
        if difference.numel():
            max_abs_diff = max(max_abs_diff, float(difference.max().item()))
        if key_changed:
            changed.append(key)

    report = {
        "before_tensor_files_sha256": {
            path.name: sha256_file(path) for path in sorted(set(before_files.values()))
        },
        "after_tensor_files_sha256": {
            path.name: sha256_file(path) for path in sorted(set(after_files.values()))
        },
        "tensor_keys": len(before_files),
        "total_elements": total_elements,
        "changed_keys": len(changed),
        "changed_key_names": changed,
        "changed_elements": changed_elements,
        "max_abs_diff": max_abs_diff,
        "comparison_scope": "all_adapter_tensors_all_elements",
    }
    if not changed:
        raise RuntimeError("adapter weight update gate failed: adapter tensors did not change from initialization")
    return report

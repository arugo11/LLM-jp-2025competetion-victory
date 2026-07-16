from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tv_gptoss120b.weight_guard import compare_adapter_update, compare_weight_update


def write_adapter(path: Path, *, changed: bool = False, non_finite: bool = False) -> None:
    path.mkdir()
    lora_a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    lora_b = torch.zeros((4, 3), dtype=torch.float32)
    if changed:
        lora_b[2, 1] = 0.125
    if non_finite:
        lora_b[0, 0] = torch.inf
    save_file(
        {
            "base_model.model.layer.lora_A.weight": lora_a,
            "base_model.model.layer.lora_B.weight": lora_b,
        },
        path / "adapter_model.safetensors",
    )


def test_adapter_weight_guard_checks_all_tensors_and_elements(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    write_adapter(before)
    write_adapter(after, changed=True)

    report = compare_adapter_update(before, after)

    assert report["comparison_scope"] == "all_adapter_tensors_all_elements"
    assert report["tensor_keys"] == 2
    assert report["changed_keys"] == 1
    assert report["changed_elements"] == 1
    assert report["max_abs_diff"] == pytest.approx(0.125)
    assert set(report["before_tensor_files_sha256"]) == {"adapter_model.safetensors"}
    assert set(report["after_tensor_files_sha256"]) == {"adapter_model.safetensors"}


def test_adapter_weight_guard_rejects_silent_non_update(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    write_adapter(before)
    write_adapter(after)

    with pytest.raises(RuntimeError, match="did not change"):
        compare_adapter_update(before, after)


def test_adapter_weight_guard_rejects_non_finite_update(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    write_adapter(before)
    write_adapter(after, non_finite=True)

    with pytest.raises(RuntimeError, match="non-finite"):
        compare_adapter_update(before, after)


def test_full_weight_guard_supports_single_safetensors_file(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    save_file({"layer.weight": torch.zeros(8)}, before / "model.safetensors")
    save_file({"layer.weight": torch.ones(8)}, after / "model.safetensors")

    report = compare_weight_update(before, after, sample_keys=1, points_per_key=8)

    assert report["changed_keys"] == 1
    assert report["max_abs_diff"] == pytest.approx(1.0)

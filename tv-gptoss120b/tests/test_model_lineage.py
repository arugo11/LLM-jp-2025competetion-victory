import json
from pathlib import Path

import pytest

from tv_gptoss120b.config import load_config
from tv_gptoss120b.hashing import sha256_file
from tv_gptoss120b.model_lineage import (
    LINEAGE_FILENAME,
    AdapterWeightUpdateEvidence,
    verify_adapter_update_files,
    verify_merged_model_lineage,
)

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"

UPDATE_EVIDENCE = {
    "before_tensor_files_sha256": {"adapter_model.safetensors": "e" * 64},
    "after_tensor_files_sha256": {"adapter_model.safetensors": "f" * 64},
    "tensor_keys": 2,
    "total_elements": 24,
    "changed_keys": 1,
    "changed_key_names": ["base_model.model.layer.lora_B.weight"],
    "changed_elements": 1,
    "max_abs_diff": 0.125,
    "comparison_scope": "all_adapter_tensors_all_elements",
}


def assigned_config():
    payload = load_config(CONFIG).model_dump()
    payload["identity"]["experiment_id"] = "0421"
    return load_config(CONFIG).__class__.model_validate(payload)


def test_merged_model_lineage_rejects_cross_arm_swap(tmp_path: Path) -> None:
    config = assigned_config()
    base = config.models.instruct
    payload = {
        "schema_version": 1,
        "stage": "full-sft-lora-grpo-merged",
        "experiment_id": "0421",
        "arm": "instruct",
        "base_model_repo_id": base.repo_id,
        "base_model_revision": base.revision,
        "sft_model_sha256": "a" * 64,
        "grpo_adapter_sha256": "b" * 64,
        "grpo_dataset_sha256": "1" * 64,
        "grpo_preprocessing_manifest_sha256": "2" * 64,
        "grpo_weight_update_report_sha256": "d" * 64,
        "grpo_weight_update": UPDATE_EVIDENCE,
        "merge_parity_sha256": "c" * 64,
    }
    (tmp_path / LINEAGE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="lineage/arm mismatch"):
        verify_merged_model_lineage(config, "thinking", tmp_path)


def test_merged_model_lineage_rejects_non_digest_evidence(tmp_path: Path) -> None:
    config = assigned_config()
    base = config.models.thinking
    payload = {
        "schema_version": 1,
        "stage": "full-sft-lora-grpo-merged",
        "experiment_id": "0421",
        "arm": "thinking",
        "base_model_repo_id": base.repo_id,
        "base_model_revision": base.revision,
        "sft_model_sha256": "mutable-alias",
        "grpo_adapter_sha256": "b" * 64,
        "grpo_dataset_sha256": "1" * 64,
        "grpo_preprocessing_manifest_sha256": "2" * 64,
        "grpo_weight_update_report_sha256": "d" * 64,
        "grpo_weight_update": UPDATE_EVIDENCE,
        "merge_parity_sha256": "c" * 64,
    }
    (tmp_path / LINEAGE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        verify_merged_model_lineage(config, "thinking", tmp_path)


def test_merged_model_lineage_rejects_zero_adapter_update(tmp_path: Path) -> None:
    config = assigned_config()
    base = config.models.thinking
    update_evidence = {**UPDATE_EVIDENCE, "changed_elements": 0}
    payload = {
        "schema_version": 1,
        "stage": "full-sft-lora-grpo-merged",
        "experiment_id": "0421",
        "arm": "thinking",
        "base_model_repo_id": base.repo_id,
        "base_model_revision": base.revision,
        "sft_model_sha256": "a" * 64,
        "grpo_adapter_sha256": "b" * 64,
        "grpo_dataset_sha256": "1" * 64,
        "grpo_preprocessing_manifest_sha256": "2" * 64,
        "grpo_weight_update_report_sha256": "d" * 64,
        "grpo_weight_update": update_evidence,
        "merge_parity_sha256": "c" * 64,
    }
    (tmp_path / LINEAGE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="greater than 0"):
        verify_merged_model_lineage(config, "thinking", tmp_path)


def test_adapter_update_evidence_is_bound_to_both_tensor_snapshots(tmp_path: Path) -> None:
    initial = tmp_path / "initial-snapshot"
    initial.mkdir()
    before = initial / "adapter_model.safetensors"
    after = tmp_path / "adapter_model.safetensors"
    before.write_bytes(b"before")
    after.write_bytes(b"after")
    evidence = AdapterWeightUpdateEvidence.model_validate({
        **UPDATE_EVIDENCE,
        "before_tensor_files_sha256": {before.name: sha256_file(before)},
        "after_tensor_files_sha256": {after.name: sha256_file(after)},
    })

    verify_adapter_update_files(tmp_path, evidence)
    after.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_adapter_update_files(tmp_path, evidence)

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import ExperimentConfig
from .hashing import canonical_json, sha256_directory, sha256_file
from .preprocessing import PREPROCESSING_MANIFEST

LINEAGE_FILENAME = "tv-gptoss120b-lineage.json"
GRPO_WEIGHT_UPDATE_FILENAME = "tv-gptoss120b-weight-update.json"


class SftModelLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    stage: Literal["full-sft"] = "full-sft"
    experiment_id: str
    arm: Literal["thinking", "instruct"]
    base_model_repo_id: str
    base_model_revision: str
    tokenized_dataset_sha256: str
    preprocessing_manifest_sha256: str
    training_metrics_sha256: str


class AdapterWeightUpdateEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    before_tensor_files_sha256: dict[str, str]
    after_tensor_files_sha256: dict[str, str]
    tensor_keys: int = Field(gt=0)
    total_elements: int = Field(gt=0)
    changed_keys: int = Field(gt=0)
    changed_key_names: list[str] = Field(min_length=1)
    changed_elements: int = Field(gt=0)
    max_abs_diff: float = Field(gt=0)
    comparison_scope: Literal["all_adapter_tensors_all_elements"]

    @model_validator(mode="after")
    def validate_counts_and_digests(self) -> AdapterWeightUpdateEvidence:
        if self.changed_keys != len(self.changed_key_names):
            raise ValueError("adapter update changed_keys does not match changed_key_names")
        if self.changed_keys > self.tensor_keys or self.changed_elements > self.total_elements:
            raise ValueError("adapter update counts exceed compared tensor scope")
        if self.before_tensor_files_sha256.keys() != self.after_tensor_files_sha256.keys():
            raise ValueError("adapter update tensor file sets differ")
        if not self.before_tensor_files_sha256:
            raise ValueError("adapter update evidence has no tensor files")
        for digest in (*self.before_tensor_files_sha256.values(), *self.after_tensor_files_sha256.values()):
            _require_sha256(digest, "adapter_tensor_file_sha256")
        return self


class MergedModelLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    stage: Literal["full-sft-lora-grpo-merged"] = "full-sft-lora-grpo-merged"
    experiment_id: str
    arm: Literal["thinking", "instruct"]
    base_model_repo_id: str
    base_model_revision: str
    sft_model_sha256: str
    grpo_adapter_sha256: str
    grpo_dataset_sha256: str
    grpo_preprocessing_manifest_sha256: str
    grpo_weight_update_report_sha256: str
    grpo_weight_update: AdapterWeightUpdateEvidence
    merge_parity_sha256: str


def _write(path: Path, payload: BaseModel) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json(payload.model_dump(mode="json")) + b"\n")


def _require_sha256(value: str, field: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"model lineage {field} must be a lowercase SHA-256")


def verify_adapter_update_files(
    adapter_dir: Path,
    evidence: AdapterWeightUpdateEvidence,
) -> None:
    roots = {
        "before": adapter_dir / "initial-snapshot",
        "after": adapter_dir,
    }
    mappings = {
        "before": evidence.before_tensor_files_sha256,
        "after": evidence.after_tensor_files_sha256,
    }
    for phase in ("before", "after"):
        for name, expected_digest in mappings[phase].items():
            if Path(name).name != name:
                raise ValueError(f"adapter update evidence contains an unsafe tensor filename: {name}")
            path = roots[phase] / name
            if not path.is_file():
                raise FileNotFoundError(f"adapter update {phase} tensor file is missing: {path}")
            actual_digest = sha256_file(path)
            if actual_digest != expected_digest:
                raise ValueError(
                    f"adapter update {phase} tensor digest mismatch for {name}: "
                    f"expected={expected_digest}, actual={actual_digest}"
                )


def write_sft_model_lineage(
    config: ExperimentConfig,
    arm: Literal["thinking", "instruct"],
    model_dir: Path,
    tokenized_dataset_dir: Path,
    training_metrics_path: Path,
) -> SftModelLineage:
    base = getattr(config.models, arm)
    lineage = SftModelLineage(
        experiment_id=config.identity.experiment_id,
        arm=arm,
        base_model_repo_id=base.repo_id,
        base_model_revision=base.revision,
        tokenized_dataset_sha256=sha256_directory(tokenized_dataset_dir),
        preprocessing_manifest_sha256=sha256_file(tokenized_dataset_dir / PREPROCESSING_MANIFEST),
        training_metrics_sha256=sha256_file(training_metrics_path),
    )
    _write(model_dir / LINEAGE_FILENAME, lineage)
    return lineage


def verify_sft_model_lineage(
    config: ExperimentConfig,
    arm: Literal["thinking", "instruct"],
    model_dir: Path,
) -> SftModelLineage:
    path = model_dir / LINEAGE_FILENAME
    lineage = SftModelLineage.model_validate_json(path.read_text(encoding="utf-8"))
    base = getattr(config.models, arm)
    expected = (config.identity.experiment_id, arm, base.repo_id, base.revision)
    actual = (
        lineage.experiment_id,
        lineage.arm,
        lineage.base_model_repo_id,
        lineage.base_model_revision,
    )
    if actual != expected:
        raise ValueError(f"SFT model lineage/arm mismatch: expected={expected}, actual={actual}")
    _require_sha256(lineage.tokenized_dataset_sha256, "tokenized_dataset_sha256")
    _require_sha256(lineage.preprocessing_manifest_sha256, "preprocessing_manifest_sha256")
    _require_sha256(lineage.training_metrics_sha256, "training_metrics_sha256")
    return lineage


def write_merged_model_lineage(
    config: ExperimentConfig,
    arm: Literal["thinking", "instruct"],
    merged_dir: Path,
    sft_model_dir: Path,
    adapter_dir: Path,
    grpo_dataset_dir: Path,
    merge_parity_path: Path,
) -> MergedModelLineage:
    verify_sft_model_lineage(config, arm, sft_model_dir)
    base = getattr(config.models, arm)
    update_report_path = adapter_dir / GRPO_WEIGHT_UPDATE_FILENAME
    update_evidence = AdapterWeightUpdateEvidence.model_validate_json(
        update_report_path.read_text(encoding="utf-8")
    )
    verify_adapter_update_files(adapter_dir, update_evidence)
    lineage = MergedModelLineage(
        experiment_id=config.identity.experiment_id,
        arm=arm,
        base_model_repo_id=base.repo_id,
        base_model_revision=base.revision,
        sft_model_sha256=sha256_directory(sft_model_dir),
        grpo_adapter_sha256=sha256_directory(adapter_dir),
        grpo_dataset_sha256=sha256_directory(grpo_dataset_dir),
        grpo_preprocessing_manifest_sha256=sha256_file(grpo_dataset_dir / PREPROCESSING_MANIFEST),
        grpo_weight_update_report_sha256=sha256_file(update_report_path),
        grpo_weight_update=update_evidence,
        merge_parity_sha256=sha256_file(merge_parity_path),
    )
    _write(merged_dir / LINEAGE_FILENAME, lineage)
    return lineage


def verify_merged_model_lineage(
    config: ExperimentConfig,
    arm: Literal["thinking", "instruct"],
    merged_dir: Path,
) -> MergedModelLineage:
    path = merged_dir / LINEAGE_FILENAME
    lineage = MergedModelLineage.model_validate_json(path.read_text(encoding="utf-8"))
    base = getattr(config.models, arm)
    expected = (config.identity.experiment_id, arm, base.repo_id, base.revision)
    actual = (
        lineage.experiment_id,
        lineage.arm,
        lineage.base_model_repo_id,
        lineage.base_model_revision,
    )
    if actual != expected:
        raise ValueError(f"merged model lineage/arm mismatch: expected={expected}, actual={actual}")
    for field in (
        "sft_model_sha256",
        "grpo_adapter_sha256",
        "grpo_dataset_sha256",
        "grpo_preprocessing_manifest_sha256",
        "grpo_weight_update_report_sha256",
        "merge_parity_sha256",
    ):
        _require_sha256(getattr(lineage, field), field)
    return lineage

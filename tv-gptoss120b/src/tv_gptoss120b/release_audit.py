from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .hashing import sha256_directory, sha256_file, sha256_value
from .manifest import StageManifest
from .resource_evidence import derive_resource_usage

REQUIRED_STAGE_MANIFESTS = {
    "generate": "generate",
    "curate": "curate",
    "difficulty": "difficulty",
    "preprocess": "preprocess",
    "thinking-sft": "thinking-sft",
    "instruct-sft": "instruct-sft",
    "thinking-grpo": "thinking-grpo",
    "instruct-grpo": "instruct-grpo",
    "evaluate": "evaluate-aime-matched",
}
REQUIRED_SELF_AUDITS = {"pre_qsub", "pre_aime", "pre_publication"}
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256")


def _verify_manifest_artifacts(config: ExperimentConfig, manifest: StageManifest, name: str) -> None:
    if not manifest.inputs or not manifest.outputs:
        raise ValueError(f"stage manifest must contain real inputs and outputs: {name}")
    root = config.experiment_root().resolve()
    for item in (*manifest.inputs, *manifest.outputs):
        _require_sha256(item.digest, f"stage artifact digest: {name}:{item.name}")
        path = Path(item.uri).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"stage artifact must be preserved under EXP_DIR: {name}:{path}")
        if item.kind.endswith("-file"):
            if not path.is_file() or sha256_file(path) != item.digest:
                raise ValueError(f"stage file artifact changed or is missing: {name}:{path}")
        elif item.kind.endswith("-directory"):
            if not path.is_dir() or sha256_directory(path) != item.digest:
                raise ValueError(f"stage directory artifact changed or is missing: {name}:{path}")
        else:
            raise ValueError(f"stage artifact kind is not content-verifiable: {name}:{item.kind}")


def _verify_runtime_provenance(config: ExperimentConfig, manifest: StageManifest, name: str) -> None:
    runtime_stage = config.runtime_stage_for_manifest(name)
    profile_name = config.runtime.stage_profile[runtime_stage]
    profile = config.runtime.profiles[profile_name]
    provenance = manifest.provenance
    expected = {
        "runtime_stage": runtime_stage,
        "runtime_profile": profile_name,
        "runtime_resource_class": profile.resource_class,
        "runtime_profile_sha256": sha256_value(profile.model_dump(mode="json")),
        "runtime_declared_environment_sha256": sha256_value(profile.environment),
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(f"stage runtime provenance mismatch: {name}:{field}")


def _positive_update(value: Any, label: str) -> None:
    if (
        not isinstance(value, dict)
        or int(value.get("changed_keys", 0)) <= 0
        or not math.isfinite(float(value.get("max_abs_diff", float("nan"))))
        or float(value.get("max_abs_diff", 0.0)) <= 0
    ):
        raise ValueError(f"{label} lacks a proven finite nonzero weight update")


def _verify_preprocessing_contract(name: str, preprocessing: Any) -> None:
    normalized_name = name.removesuffix("-preprocessed")
    expected_kind = "sft" if normalized_name.endswith("-sft") else "grpo"
    expected_arm = normalized_name.removesuffix(f"-{expected_kind}")
    if (
        not isinstance(preprocessing, dict)
        or preprocessing.get("kind") != expected_kind
        or preprocessing.get("arm") != expected_arm
        or preprocessing.get("smoke") is not False
        or preprocessing.get("assistant_only_loss") is not (expected_kind == "sft")
        or preprocessing.get("packing") is not False
        or preprocessing.get("required_resource_class") != "cpu_only"
        or int(preprocessing.get("preprocessing_workers", 0)) <= 0
    ):
        raise ValueError(f"{name} lacks the matched production preprocessing contract")
    for field in (
        "config_sha256",
        "chat_template_sha256",
        "prompt_template_sha256",
        "template_kwargs_sha256",
        "source_sha256",
        "source_records_sha256",
        "dataset_records_sha256",
        "dataset_payload_sha256",
        "label_mask_policy_sha256",
    ):
        _require_sha256(preprocessing.get(field), f"{name} preprocessing {field}")
    if not isinstance(preprocessing.get("model_revision"), str) or SHA40.fullmatch(
        preprocessing["model_revision"]
    ) is None:
        raise ValueError(f"{name} preprocessing model_revision must be a commit SHA")


def _verify_preprocessing_metrics(name: str, metrics: dict[str, Any]) -> None:
    _verify_preprocessing_contract(name, metrics.get("preprocessing"))
    _require_sha256(metrics.get("preprocessing_manifest_sha256"), f"{name} preprocessing manifest")


def _verify_stage_metrics(config: ExperimentConfig, name: str, metrics: dict[str, Any]) -> None:
    if name == "generate":
        if metrics.get("spec_count") != config.data.spec_count or metrics.get("backend") != "vllm":
            raise ValueError("generation stage metrics do not prove the full vLLM run")
        if metrics.get("smoke") is not False:
            raise ValueError("generation release evidence cannot be a smoke run")
    elif name == "curate":
        if metrics.get("quality_gate") != "passed":
            raise ValueError("curation quality gate did not pass")
        _require_sha256(metrics.get("dataset_sha256"), "curated dataset digest")
        _require_sha256(metrics.get("aime_reference_sha256"), "AIME reference digest")
    elif name == "difficulty":
        if metrics.get("hardening_gate") is not True:
            raise ValueError("difficulty hardening gate did not pass")
        if int(metrics.get("spec_count", 0)) < config.validation.valid_pair_min.difficulty:
            raise ValueError("difficulty evidence is below the valid paired holdout gate")
        deltas: dict[str, float] = {}
        for field in ("pooled_delta", "thinking_delta", "instruct_delta"):
            value = metrics.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"difficulty evidence lacks a finite {field}")
            deltas[field] = float(value)
        if deltas["pooled_delta"] < config.difficulty.pooled_delta_min:
            raise ValueError(
                "difficulty pooled delta is below the scientific gate: "
                f"{deltas['pooled_delta']} < {config.difficulty.pooled_delta_min}"
            )
        if (
            deltas["thinking_delta"] <= config.difficulty.arm_delta_min
            or deltas["instruct_delta"] <= config.difficulty.arm_delta_min
        ):
            raise ValueError("difficulty arm deltas must exceed the configured minimum in both arms")
    elif name == "preprocess":
        expected = {
            f"{arm}-{kind}-preprocessed"
            for arm in ("thinking", "instruct")
            for kind in ("sft", "grpo")
        }
        artifacts = metrics.get("artifacts")
        if metrics.get("smoke") is not False or metrics.get("artifact_count") != 4:
            raise ValueError("preprocess release evidence must contain four production artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != expected:
            raise ValueError(f"preprocess artifacts must be exactly {sorted(expected)}")
        for artifact_name, preprocessing in artifacts.items():
            _verify_preprocessing_contract(artifact_name, preprocessing)
    elif name.endswith("-sft"):
        _verify_preprocessing_metrics(name, metrics)
        _positive_update(metrics.get("weight_update"), f"{name} Full SFT")
        if int(metrics.get("global_step", 0)) < int(metrics.get("expected_steps", 1)):
            raise ValueError(f"{name} optimizer steps are below the expected gate")
    elif name.endswith("-grpo"):
        _verify_preprocessing_metrics(name, metrics)
        _positive_update(metrics.get("adapter_weight_update"), f"{name} LoRA adapter")
        if int(metrics.get("global_step", -1)) != config.grpo.max_steps:
            raise ValueError(f"{name} optimizer steps do not match the fixed production plan")
        parity = metrics.get("merge_parity")
        if (
            not isinstance(parity, dict)
            or parity.get("greedy_outputs_match") is not True
            or float(parity.get("max_abs_logit_diff", float("inf"))) > float(parity.get("tolerance", 0.0))
        ):
            raise ValueError(f"{name} merge parity gate did not pass")
    elif name == "evaluate":
        if metrics.get("normalized_sample_count") != 960:
            raise ValueError("matched AIME evidence must contain exactly 960 normalized samples")
        _require_sha256(metrics.get("matched_config_sha256"), "matched AIME config digest")
        _require_sha256(metrics.get("harness_manifest_sha256"), "Swallow harness manifest digest")
        _require_sha256(metrics.get("harness_worktree_sha256"), "Swallow harness worktree digest")
        summary = metrics.get("summary")
        if (
            not isinstance(summary, dict)
            or type(summary.get("scientific_aime_gate")) is not bool
            or set(summary.get("arms", {})) != {"thinking", "instruct"}
        ):
            raise ValueError("matched AIME evidence lacks both arms and the scientific result")


def audit_release_evidence(
    config: ExperimentConfig,
    config_path: Path,
    lineage: dict[str, Any],
) -> dict[str, Any]:
    experiment_id = config.require_assigned_id()
    stage_paths = lineage.get("stage_manifests")
    audit_paths = lineage.get("self_audits")
    usage_paths = lineage.get("resource_usage_records")
    if not isinstance(stage_paths, dict) or set(stage_paths) != set(REQUIRED_STAGE_MANIFESTS):
        raise ValueError(f"stage_manifests must be exactly {sorted(REQUIRED_STAGE_MANIFESTS)}")
    if not isinstance(audit_paths, dict) or set(audit_paths) != REQUIRED_SELF_AUDITS:
        raise ValueError(f"self_audits must be exactly {sorted(REQUIRED_SELF_AUDITS)}")
    if not isinstance(usage_paths, list) or not usage_paths:
        raise ValueError("resource_usage_records must be a non-empty list")

    config_sha = sha256_file(config_path)
    stage_checksums = {}
    git_shas = set()
    for name, expected_stage in REQUIRED_STAGE_MANIFESTS.items():
        path = Path(str(stage_paths[name]))
        manifest = StageManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.experiment_id != experiment_id or manifest.stage != expected_stage:
            raise ValueError(f"stage manifest identity mismatch: {name}")
        if manifest.config_sha256 != config_sha:
            raise ValueError(f"stage manifest config hash mismatch: {name}")
        if SHA40.fullmatch(manifest.git_sha) is None:
            raise ValueError(f"stage manifest git SHA is not immutable: {name}")
        _verify_runtime_provenance(config, manifest, name)
        _verify_manifest_artifacts(config, manifest, name)
        _verify_stage_metrics(config, name, manifest.metrics)
        git_shas.add(manifest.git_sha)
        stage_checksums[name] = sha256_file(path)
    if len(git_shas) != 1:
        raise ValueError(f"all production stages must use one code commit, found {git_shas}")

    audit_checksums = {}
    for name, raw_path in audit_paths.items():
        path = Path(str(raw_path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_fields = {
            "schema_version",
            "audit_name",
            "status",
            "experiment_id",
            "config_sha256",
            "git_sha",
            "checked_at",
            "checks",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise ValueError(f"self audit schema is invalid for {name}")
        checked_at = datetime.fromisoformat(str(payload["checked_at"]))
        checks = payload["checks"]
        if (
            payload["schema_version"] != 1
            or payload["audit_name"] != name
            or payload["status"] != "PASS"
            or payload["experiment_id"] != experiment_id
            or payload["config_sha256"] != config_sha
            or payload["git_sha"] not in git_shas
            or checked_at.tzinfo is None
            or not isinstance(checks, dict)
            or not checks
            or any(value is not True for value in checks.values())
        ):
            raise ValueError(f"self audit did not pass for {name}")
        audit_checksums[name] = sha256_file(path)

    h200_total = 0.0
    cpu_total = 0.0
    stage_totals = {stage: 0.0 for stage in config.resources.stage_node_hours}
    usage_checksums = {}
    job_ids = set()
    derived_usage_records = []
    max_storage_bytes = 0
    max_storage_inodes = 0
    for raw_path in usage_paths:
        path = Path(str(raw_path))
        payload = derive_resource_usage(config, path)
        derived_usage_records.append(payload)
        if payload.get("experiment_id") != experiment_id:
            raise ValueError(f"resource usage experiment mismatch: {path}")
        job_id = str(payload.get("pbs_job_id", ""))
        if not job_id or job_id in job_ids:
            raise ValueError(f"resource usage has missing or duplicate PBS job ID: {job_id}")
        job_ids.add(job_id)
        node_hours = float(payload["node_hours"])
        resource_type = payload["resource_type"]
        if resource_type == "rt_HF":
            stage = payload["budget_stage"]
            if stage not in stage_totals:
                raise ValueError(f"unknown H200 budget stage: {stage}")
            stage_totals[stage] += node_hours
            h200_total += node_hours
        elif resource_type == "rt_HC":
            cpu_total += node_hours
        else:
            raise ValueError(f"unapproved resource type in usage record: {resource_type}")
        max_storage_bytes = max(max_storage_bytes, int(payload["measured_storage_bytes"]))
        max_storage_inodes = max(max_storage_inodes, int(payload["measured_storage_inodes"]))
        usage_checksums[job_id] = payload["evidence_sha256"]
    reserve_overage = sum(
        max(0.0, total - config.resources.stage_node_hours[stage])
        for stage, total in stage_totals.items()
    )
    if reserve_overage > config.resources.h200_node_hours_reserve:
        raise ValueError(f"H200 reserve budget exceeded: {reserve_overage}")
    if h200_total > config.resources.h200_node_hours_absolute_max:
        raise ValueError(f"H200 absolute budget exceeded: {h200_total}")
    if h200_total > config.resources.h200_node_hours_planned or reserve_overage > 0:
        reserve_records = [item for item in derived_usage_records if item["reserve_used"]]
        invalid_reserve_reason = any(
            item.get("reserve_reason") != "fixed_stage_completion" for item in reserve_records
        )
        if not reserve_records or invalid_reserve_reason:
            raise ValueError("H200 reserve usage lacks fixed_stage_completion evidence")
    if cpu_total > config.resources.cpu_node_hours_max:
        raise ValueError(f"CPU budget exceeded: {cpu_total}")
    if max_storage_bytes > config.resources.storage_bytes_max:
        raise ValueError(f"storage budget exceeded: {max_storage_bytes}")
    return {
        "status": "PASS",
        "experiment_id": experiment_id,
        "config_sha256": config_sha,
        "git_sha": next(iter(git_shas)),
        "stage_manifest_sha256": stage_checksums,
        "self_audit_sha256": audit_checksums,
        "resource_usage_sha256": usage_checksums,
        "h200_node_hours": h200_total,
        "cpu_node_hours": cpu_total,
        "stage_node_hours": stage_totals,
        "reserve_node_hours": reserve_overage,
        "max_storage_bytes": max_storage_bytes,
        "max_storage_inodes": max_storage_inodes,
    }

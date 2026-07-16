from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .config import ExperimentConfig
from .hashing import canonical_json, sha256_file

SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvidenceFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str


class ResourceEvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    experiment_id: str
    pbs_job_id: str
    job_manifest: EvidenceFile
    pbs_script: EvidenceFile
    qstat_final: EvidenceFile
    storage_audit: EvidenceFile


def write_resource_evidence_record(
    config: ExperimentConfig,
    *,
    pbs_job_id: str,
    job_manifest_path: Path,
    pbs_script_path: Path,
    qstat_final_path: Path,
    storage_audit_path: Path,
    output_path: Path,
    require_success: bool = True,
) -> ResourceEvidenceRecord:
    record = ResourceEvidenceRecord(
        schema_version=1,
        experiment_id=config.require_assigned_id(),
        pbs_job_id=pbs_job_id,
        job_manifest=EvidenceFile(path=str(job_manifest_path.resolve()), sha256=sha256_file(job_manifest_path)),
        pbs_script=EvidenceFile(path=str(pbs_script_path.resolve()), sha256=sha256_file(pbs_script_path)),
        qstat_final=EvidenceFile(path=str(qstat_final_path.resolve()), sha256=sha256_file(qstat_final_path)),
        storage_audit=EvidenceFile(path=str(storage_audit_path.resolve()), sha256=sha256_file(storage_audit_path)),
    )
    with output_path.open("xb") as handle:
        handle.write(canonical_json(record.model_dump(mode="json")) + b"\n")
    derive_resource_usage(config, output_path, require_success=require_success)
    return record


def _verified_path(root: Path, evidence: EvidenceFile, label: str) -> Path:
    path = Path(evidence.path).resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError(f"{label} evidence must be a file under EXP_DIR: {path}")
    if not SHA256.fullmatch(evidence.sha256) or sha256_file(path) != evidence.sha256:
        raise ValueError(f"{label} evidence SHA-256 mismatch: {path}")
    return path


def _walltime_seconds(value: Any) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    parts = str(value).split(":")
    if len(parts) != 3:
        raise ValueError(f"qstat walltime must be HH:MM:SS: {value}")
    hours, minutes, seconds = (int(part) for part in parts)
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"qstat walltime is invalid: {value}")
    return hours * 3600 + minutes * 60 + seconds


def _qstat_job(path: Path, job_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("Jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, dict) or set(jobs) != {job_id} or not isinstance(jobs[job_id], dict):
        raise ValueError(f"qstat evidence must contain exactly job {job_id}")
    return jobs[job_id]


def _selected_nodes(resource_list: dict[str, Any]) -> int:
    if "nodect" in resource_list:
        nodes = int(resource_list["nodect"])
    else:
        match = re.match(r"^([0-9]+)(?::|$)", str(resource_list.get("select", "")))
        if match is None:
            raise ValueError("qstat Resource_List lacks nodect/select node count")
        nodes = int(match.group(1))
    if nodes <= 0:
        raise ValueError("qstat selected node count must be positive")
    return nodes


def _storage_metrics(path: Path, expected_root: Path) -> tuple[int, int]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        if key in {
            "audit_mode",
            "storage_audit_target",
            "created_at",
            "target_group",
            "bytes",
            "inode",
            "scan_status",
        }:
            if key in values:
                raise ValueError(f"storage audit contains duplicate field: {key}")
            values[key] = value
    required = {
        "audit_mode",
        "storage_audit_target",
        "created_at",
        "target_group",
        "bytes",
        "inode",
        "scan_status",
    }
    if set(values) != required:
        raise ValueError(f"storage audit fields are incomplete: {sorted(required - values.keys())}")
    if (
        values["audit_mode"] != "deep"
        or values["scan_status"] != "complete"
        or values["target_group"] != "gcg51557"
        or Path(values["storage_audit_target"]).resolve() != expected_root
    ):
        raise ValueError("release requires a complete deep storage audit for the canonical EXP_DIR")
    measured_bytes = int(values["bytes"])
    measured_inodes = int(values["inode"])
    from datetime import datetime

    captured_at = datetime.fromisoformat(values["created_at"])
    if captured_at.tzinfo is None:
        raise ValueError("storage audit created_at must be timezone-aware")
    if measured_bytes < 0 or measured_inodes < 0:
        raise ValueError("storage audit bytes/inodes must be non-negative")
    return measured_bytes, measured_inodes


def derive_storage_metrics(config: ExperimentConfig, storage_audit_path: Path) -> dict[str, Any]:
    root = config.experiment_root().resolve()
    path = storage_audit_path.resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("storage audit evidence must be a file under EXP_DIR")
    measured_bytes, measured_inodes = _storage_metrics(path, root)
    created_at = next(
        line.split("=", maxsplit=1)[1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("created_at=")
    )
    return {"bytes": measured_bytes, "inodes": measured_inodes, "created_at": created_at}


def derive_pre_qsub_storage_metrics(config: ExperimentConfig, storage_audit_path: Path) -> dict[str, Any]:
    """Read a complete bounded or deep audit used for the next qsub decision.

    Release/resource evidence continues to require ``derive_storage_metrics`` and a
    complete PBS-side deep audit.  The initial compute job may use a bounded
    login-node audit; subsequent jobs may reuse a newer complete deep audit instead
    of discarding stronger evidence.
    """
    root = config.experiment_root().resolve()
    path = storage_audit_path.resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("pre-qsub storage audit evidence must be a file under EXP_DIR")
    values: dict[str, str] = {}
    base_fields = {
        "audit_mode",
        "storage_audit_target",
        "created_at",
        "target_group",
        "bytes",
        "inode",
        "scan_status",
    }
    allowed = base_fields | {"file_scan_limit"}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        if key not in allowed:
            continue
        if key in values:
            raise ValueError(f"pre-qsub storage audit contains duplicate field: {key}")
        values[key] = value
    mode = values.get("audit_mode")
    required = allowed if mode == "bounded-pre-qsub" else base_fields
    if mode not in {"bounded-pre-qsub", "deep"} or not required.issubset(values):
        raise ValueError(f"pre-qsub storage audit fields are incomplete: {sorted(required - values.keys())}")
    if (
        values["scan_status"] != "complete"
        or values["target_group"] != config.identity.group
        or Path(values["storage_audit_target"]).resolve() != root
    ):
        raise ValueError("qsub requires a complete bounded or deep audit for the canonical EXP_DIR")
    measured_bytes = int(values["bytes"])
    measured_inodes = int(values["inode"])
    file_scan_limit = int(values["file_scan_limit"]) if mode == "bounded-pre-qsub" else None
    from datetime import datetime

    captured_at = datetime.fromisoformat(values["created_at"])
    if captured_at.tzinfo is None:
        raise ValueError("pre-qsub storage audit created_at must be timezone-aware")
    if measured_bytes < 0 or measured_inodes < 0 or (
        mode == "bounded-pre-qsub" and file_scan_limit != 100_000
    ):
        raise ValueError("pre-qsub storage audit metrics or fixed 100000-file limit are invalid")
    if file_scan_limit is not None and measured_inodes > file_scan_limit:
        raise ValueError("pre-qsub storage audit exceeded its bounded login-node scan limit")
    return {
        "audit_mode": mode,
        "bytes": measured_bytes,
        "inodes": measured_inodes,
        "created_at": values["created_at"],
        "file_scan_limit": file_scan_limit,
    }


def derive_resource_usage(
    config: ExperimentConfig,
    evidence_path: Path,
    *,
    require_success: bool = True,
) -> dict[str, Any]:
    evidence = ResourceEvidenceRecord.model_validate_json(evidence_path.read_text(encoding="utf-8"))
    experiment_id = config.require_assigned_id()
    root = config.experiment_root().resolve()
    if evidence.experiment_id != experiment_id:
        raise ValueError("resource evidence experiment ID mismatch")
    if not evidence_path.resolve().is_relative_to(root):
        raise ValueError("resource evidence record must be stored under EXP_DIR")

    manifest_path = _verified_path(root, evidence.job_manifest, "job manifest")
    pbs_path = _verified_path(root, evidence.pbs_script, "PBS script")
    qstat_path = _verified_path(root, evidence.qstat_final, "qstat")
    storage_path = _verified_path(root, evidence.storage_audit, "storage audit")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    approval_fields = {
        "submit_account",
        "responsible_person",
        "experiment_owner",
        "user_execution_approval_ref",
    }
    team_approval = manifest.get("team_approval_ref")
    opportunistic_reserved_smoke = (
        team_approval is None
        and manifest.get("billing_mode") == "reserved"
        and manifest.get("smoke") is True
        and manifest.get("preemptible") is True
        and manifest.get("automatic_retry") is False
        and manifest.get("nodes") == 1
        and manifest.get("array_size") == 1
        and manifest.get("max_array_concurrency") == 1
        and isinstance(manifest.get("opportunistic_policy_ref"), str)
        and bool(manifest["opportunistic_policy_ref"])
    )
    if (
        manifest.get("experiment_id") != experiment_id
        or Path(str(manifest.get("output_root", ""))).resolve() != root
        or manifest.get("job_name") is None
        or any(not isinstance(manifest.get(field), str) or not manifest[field] for field in approval_fields)
        or not (
            isinstance(team_approval, str) and bool(team_approval)
            or opportunistic_reserved_smoke
        )
    ):
        raise ValueError("job manifest identity/output root/ownership/approval evidence is invalid")
    resource_type = manifest.get("resource_type")
    budget_stage = manifest.get("budget_stage")
    if resource_type not in {"rt_HF", "rt_HC"}:
        raise ValueError(f"unapproved resource type in job manifest: {resource_type}")
    if (resource_type == "rt_HC") != (budget_stage == "cpu"):
        raise ValueError("CPU resource evidence must use only the cpu budget stage")
    if resource_type == "rt_HF" and budget_stage not in config.resources.stage_node_hours:
        raise ValueError("GPU resource evidence has an unknown fixed budget stage")

    job = _qstat_job(qstat_path, evidence.pbs_job_id)
    if job.get("Job_Name") != manifest["job_name"]:
        raise ValueError("qstat job name does not match the immutable job manifest")
    if job.get("job_state") != "F":
        raise ValueError("resource evidence requires a finished PBS job")
    raw_exit_status = job.get("Exit_status")
    if raw_exit_status is None:
        raise ValueError("resource evidence lacks PBS Exit_status")
    exit_status = int(raw_exit_status)
    if exit_status < 0:
        raise ValueError("resource evidence has an invalid PBS Exit_status")
    if require_success and exit_status != 0:
        raise ValueError("release evidence requires a finished successful PBS job")
    resources_used = job.get("resources_used")
    resource_list = job.get("Resource_List")
    if not isinstance(resources_used, dict) or not isinstance(resource_list, dict):
        raise ValueError("qstat evidence lacks resources_used/Resource_List")
    nodes = _selected_nodes(resource_list)
    if nodes != int(manifest.get("nodes", -1)):
        raise ValueError("qstat selected nodes do not match the immutable job manifest")
    requested_seconds = _walltime_seconds(resource_list.get("walltime"))
    if requested_seconds != _walltime_seconds(manifest.get("requested_walltime")):
        raise ValueError("qstat requested walltime does not match the immutable job manifest")
    actual_seconds = _walltime_seconds(resources_used.get("walltime"))
    if actual_seconds <= 0:
        raise ValueError("qstat actual walltime must be positive")
    if actual_seconds > requested_seconds:
        raise ValueError("qstat actual walltime exceeds requested walltime")

    pbs_text = pbs_path.read_text(encoding="utf-8")
    required_directives = {
        f"#PBS -N {manifest['job_name']}",
        f"#PBS -P {config.identity.group}",
        f"#PBS -v RTYPE={resource_type}",
        f"#PBS -l walltime={manifest['requested_walltime']}",
    }
    if not required_directives.issubset(set(pbs_text.splitlines())):
        raise ValueError("PBS script does not match job manifest identity/resource/walltime")
    select_prefix = f"#PBS -l select={nodes}:"
    if not any(line.startswith(select_prefix) for line in pbs_text.splitlines()):
        raise ValueError("PBS script selected nodes do not match qstat/job manifest")

    measured_bytes, measured_inodes = _storage_metrics(storage_path, root)
    return {
        "experiment_id": experiment_id,
        "pbs_job_id": evidence.pbs_job_id,
        "resource_type": resource_type,
        "budget_stage": budget_stage,
        "nodes": nodes,
        "actual_walltime_seconds": actual_seconds,
        "node_hours": nodes * actual_seconds / 3600,
        "exit_status": exit_status,
        "successful": exit_status == 0,
        "measured_storage_bytes": measured_bytes,
        "measured_storage_inodes": measured_inodes,
        "reserve_used": bool(manifest.get("reserve_used", False)),
        "reserve_reason": manifest.get("reserve_reason"),
        "evidence_sha256": {
            "record": sha256_file(evidence_path),
            "job_manifest": evidence.job_manifest.sha256,
            "pbs_script": evidence.pbs_script.sha256,
            "qstat_final": evidence.qstat_final.sha256,
            "storage_audit": evidence.storage_audit.sha256,
        },
    }

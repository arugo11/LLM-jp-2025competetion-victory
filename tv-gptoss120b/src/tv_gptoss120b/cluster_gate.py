from __future__ import annotations

import json
import shlex
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .hashing import sha256_file
from .pbs import derive_projected_resource_usage
from .resource_evidence import derive_storage_metrics


def verify_qsub_gate(
    config: ExperimentConfig,
    *,
    approval_record: Path,
    policy_snapshot: Path,
    job_manifest: Path,
    pbs_script: Path,
    storage_audit: Path,
    plan_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Final local gate. The cluster-resource-preflight skill remains the final external gate."""
    experiment_id = config.require_assigned_id()
    approval = json.loads(approval_record.read_text(encoding="utf-8"))
    policy = json.loads(policy_snapshot.read_text(encoding="utf-8"))
    manifest = json.loads(job_manifest.read_text(encoding="utf-8"))
    projected = derive_projected_resource_usage(config, manifest)
    storage = derive_storage_metrics(config, storage_audit)
    current = now or datetime.now(UTC)
    captured = datetime.fromisoformat(policy["captured_at"])
    storage_captured = datetime.fromisoformat(storage["created_at"])
    if captured.tzinfo is None:
        raise ValueError("policy snapshot timestamp must be timezone-aware")
    failures = []
    policy_age = current - captured.astimezone(UTC)
    storage_age = current - storage_captured.astimezone(UTC)
    if not timedelta(0) <= policy_age <= timedelta(minutes=60):
        failures.append("Slack queue policy snapshot is older than 60 minutes")
    if not timedelta(0) <= storage_age <= timedelta(minutes=60):
        failures.append("deep storage audit is older than 60 minutes")
    if approval.get("plan_sha256") != plan_sha256:
        failures.append("approved plan hash mismatch")
    if approval.get("user_approved") is not True or approval.get("foundation_meeting_approved") is not True:
        failures.append("user and foundation-model meeting approvals are both required")
    if approval.get("experiment_id") != experiment_id:
        failures.append("approval experiment ID mismatch")
    if manifest.get("plan_sha256") != plan_sha256:
        failures.append("job manifest is not bound to the approved plan hash")
    if (
        manifest.get("experiment_id") != experiment_id
        or manifest.get("group") != config.identity.group
        or Path(str(manifest.get("output_root", ""))).resolve() != config.experiment_root().resolve()
    ):
        failures.append("job manifest identity/group/output root mismatch")
    if policy.get("queue_verified") is not True or not policy.get("queue"):
        failures.append("queue/quota snapshot is not verified")
    if policy.get("quota_verified") is not True:
        failures.append("queue/quota snapshot is not verified")
    if manifest.get("queue") != policy.get("queue"):
        failures.append("job manifest queue does not match the verified policy snapshot")
    if projected["h200_after_job"] > config.resources.h200_node_hours_absolute_max:
        failures.append("H200 absolute budget would be exceeded")
    if manifest.get("nodes") != 1:
        failures.append("approved pilot permits exactly one node per job")
    runtime_profile = None
    try:
        runtime_stage = config.runtime_stage_for_manifest(str(manifest["runtime_stage"]))
        runtime_profile = config.runtime_profile_for_stage(runtime_stage)
        if runtime_profile.resource_class == "h200" and (
            manifest.get("resource_type") != "rt_HF" or int(manifest.get("gpus_per_node", 0)) != 8
        ):
            failures.append("H200 runtime profile requires rt_HF with 8 GPUs")
        if runtime_profile.resource_class == "cpu_only" and (
            manifest.get("resource_type") != "rt_HC" or int(manifest.get("gpus_per_node", -1)) != 0
        ):
            failures.append("CPU runtime profile requires rt_HC with zero GPUs")
    except (KeyError, ValueError) as error:
        failures.append(f"runtime stage/profile is invalid: {error}")
    if (
        manifest.get("resource_type") == "rt_HF"
        and projected["requested_node_hours"]
        > config.resources.stage_node_hours[str(manifest["budget_stage"])]
    ):
        failures.append("single GPU request exceeds its fixed-stage budget")
    if projected["reserve_after_job"] > config.resources.h200_node_hours_reserve:
        failures.append("H200 fixed-stage reserve would be exceeded")
    needs_reserve = (
        projected["reserve_after_job"] > 0
        or projected["h200_after_job"] > config.resources.h200_node_hours_planned
    )
    if needs_reserve and (
        manifest.get("reserve_used") is not True
        or manifest.get("reserve_reason") != "fixed_stage_completion"
    ):
        failures.append("reserve lacks the fixed-stage completion declaration")
    if manifest.get("reserve_used") is True and not needs_reserve:
        failures.append("reserve is declared without evidence-derived need")
    if projected["cpu_after_job"] > config.resources.cpu_node_hours_max:
        failures.append("CPU absolute budget would be exceeded")
    if storage["bytes"] > config.resources.storage_bytes_max:
        failures.append("250 GB storage gate would be exceeded")
    canonical_root = config.experiment_root()
    pbs_text = pbs_script.read_text(encoding="utf-8")
    expected_pbs_lines = {
        f"#PBS -N {manifest.get('job_name')}",
        f"#PBS -P {manifest.get('group')}",
        f"#PBS -q {manifest.get('queue')}",
        f"#PBS -v RTYPE={manifest.get('resource_type')}",
        (
            f"#PBS -l select={manifest.get('nodes')}:ncpus={manifest.get('cpus_per_node')}:"
            f"ngpus={manifest.get('gpus_per_node')}"
        ),
        f"#PBS -l walltime={manifest.get('requested_walltime')}",
        *(
            [
                f"export {name}={shlex.quote(value)}"
                for name, value in sorted(runtime_profile.environment.items())
            ]
            if runtime_profile is not None
            else []
        ),
        shlex.join(manifest.get("main_command", [])),
    }
    if str(canonical_root) not in pbs_text or not expected_pbs_lines.issubset(set(pbs_text.splitlines())):
        failures.append("PBS script does not match the canonical experiment manifest")
    if failures:
        raise RuntimeError("qsub gate failed: " + "; ".join(failures))
    return {
        "status": "PASS",
        "experiment_id": experiment_id,
        "queue": policy["queue"],
        "approval_record_sha256": sha256_file(approval_record),
        "policy_snapshot_sha256": sha256_file(policy_snapshot),
        "job_manifest_sha256": sha256_file(job_manifest),
        "pbs_script_sha256": sha256_file(pbs_script),
        "storage_audit_sha256": sha256_file(storage_audit),
        "projected_resource_usage": projected,
        "measured_storage": {"bytes": storage["bytes"], "inodes": storage["inodes"]},
        "requires_external_cluster_preflight": True,
    }

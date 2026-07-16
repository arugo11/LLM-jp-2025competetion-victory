from __future__ import annotations

import json
import shlex
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .hashing import sha256_file
from .pbs import derive_projected_resource_usage
from .resource_evidence import derive_pre_qsub_storage_metrics


def _walltime_seconds(value: object) -> int | None:
    parts = str(value).split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (int(item) for item in parts)
    except ValueError:
        return None
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def _opportunistic_reserved_smoke_allowed(
    approval: dict[str, Any], policy: dict[str, Any], manifest: dict[str, Any]
) -> bool:
    scope = policy.get("opportunistic_reserved_smoke")
    if not isinstance(scope, dict) or scope.get("allowed") is not True:
        return False
    requested = _walltime_seconds(manifest.get("requested_walltime"))
    maximum = _walltime_seconds(scope.get("max_walltime"))
    policy_ref = scope.get("source_ref")
    return bool(
        approval.get("user_approved") is True
        and approval.get("approval_scope") == "opportunistic_reserved_smoke"
        and manifest.get("smoke") is True
        and manifest.get("billing_mode") == "reserved"
        and manifest.get("nodes") == 1
        and manifest.get("array_size") == 1
        and manifest.get("max_array_concurrency") == 1
        and manifest.get("preemptible") is True
        and manifest.get("automatic_retry") is False
        and manifest.get("team_approval_ref") in {None, ""}
        and isinstance(policy_ref, str)
        and manifest.get("opportunistic_policy_ref") == policy_ref
        and requested is not None
        and maximum is not None
        and requested <= maximum
    )


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
    storage = derive_pre_qsub_storage_metrics(config, storage_audit)
    current = now or datetime.now(UTC)
    captured = datetime.fromisoformat(policy["checked_at"])
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
    full_approval = (
        approval.get("user_approved") is True
        and approval.get("foundation_meeting_approved") is True
    )
    opportunistic_smoke = _opportunistic_reserved_smoke_allowed(approval, policy, manifest)
    if not full_approval and not opportunistic_smoke:
        failures.append(
            "user and foundation-model meeting approvals are required outside the narrow "
            "opportunistic reserved smoke scope"
        )
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
    allowed_queues = policy.get("allowed_queues")
    if (
        policy.get("queue_verified") is not True
        or not isinstance(allowed_queues, list)
        or not allowed_queues
    ):
        failures.append("queue/quota snapshot is not verified")
    if policy.get("quota_verified") is not True:
        failures.append("queue/quota snapshot is not verified")
    if manifest.get("queue") not in (allowed_queues if isinstance(allowed_queues, list) else []):
        failures.append("job manifest queue is not in the verified policy snapshot")
    resource_shape = policy.get("resource_shapes", {}).get(manifest.get("resource_type"), {})
    if not isinstance(resource_shape, dict) or (
        manifest.get("cpus_per_node") != resource_shape.get("cpus_per_node")
        or manifest.get("gpus_per_node") != resource_shape.get("gpus_per_node")
    ):
        failures.append("job manifest CPU/GPU shape does not match current scheduler facts")
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
    predicted_bytes = manifest.get("predicted_new_bytes_upper_bound")
    if not isinstance(predicted_bytes, int) or isinstance(predicted_bytes, bool) or predicted_bytes < 0:
        failures.append("predicted_new_bytes_upper_bound must be a non-negative integer")
        predicted_bytes = 0
    if storage["bytes"] + predicted_bytes > config.resources.storage_bytes_max:
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
        "queue": manifest["queue"],
        "approval_record_sha256": sha256_file(approval_record),
        "policy_snapshot_sha256": sha256_file(policy_snapshot),
        "job_manifest_sha256": sha256_file(job_manifest),
        "pbs_script_sha256": sha256_file(pbs_script),
        "storage_audit_sha256": sha256_file(storage_audit),
        "projected_resource_usage": projected,
        "measured_storage": {"bytes": storage["bytes"], "inodes": storage["inodes"]},
        "projected_storage_bytes": storage["bytes"] + predicted_bytes,
        "approval_mode": "opportunistic_reserved_smoke" if opportunistic_smoke else "full",
        "requires_external_cluster_preflight": True,
    }

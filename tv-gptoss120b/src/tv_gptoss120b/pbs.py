from __future__ import annotations

import json
import shlex
from pathlib import Path

from .config import ExperimentConfig
from .resource_evidence import derive_resource_usage

FORBIDDEN_CALLER_RESOURCE_FIELDS = {
    "cumulative_cpu_node_hours_after_job",
    "cumulative_stage_node_hours_after_job",
    "cumulative_h200_node_hours_after_job",
    "cumulative_reserve_node_hours_after_job",
}


def derive_projected_resource_usage(config: ExperimentConfig, manifest: dict) -> dict:
    forbidden = FORBIDDEN_CALLER_RESOURCE_FIELDS & manifest.keys()
    if forbidden:
        raise ValueError(
            "caller-authored cumulative resource fields are forbidden: "
            f"{sorted(forbidden)}"
        )
    required = {
        "resource_type",
        "nodes",
        "requested_walltime",
        "budget_stage",
        "runtime_stage",
        "prior_resource_evidence_records",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"resource projection fields are missing: {sorted(missing)}")
    try:
        parts = str(manifest["requested_walltime"]).split(":")
        if len(parts) != 3:
            raise ValueError
        hours, minutes, seconds = (int(value) for value in parts)
    except ValueError as error:
        raise ValueError("requested_walltime must be HH:MM:SS") from error
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError("requested_walltime must be HH:MM:SS")
    nodes = int(manifest["nodes"])
    if nodes <= 0:
        raise ValueError("nodes must be positive")
    requested_node_hours = nodes * (hours + minutes / 60 + seconds / 3600)
    resource_type = manifest["resource_type"]
    stage = str(manifest["budget_stage"])
    if resource_type not in {"rt_HF", "rt_HC"}:
        raise ValueError("resource_type must be rt_HF or rt_HC")
    if resource_type == "rt_HC" and stage != "cpu":
        raise ValueError("rt_HC resource projection requires the cpu budget stage")
    if resource_type == "rt_HF" and stage not in config.resources.stage_node_hours:
        raise ValueError("rt_HF resource projection requires an approved fixed H200 stage")
    prior_evidence = manifest["prior_resource_evidence_records"]
    if not isinstance(prior_evidence, list) or not all(
        isinstance(item, str) and item for item in prior_evidence
    ):
        raise ValueError("prior_resource_evidence_records must be a list of non-empty paths")
    prior_usage = [derive_resource_usage(config, Path(item)) for item in prior_evidence]
    prior_job_ids = [str(item["pbs_job_id"]) for item in prior_usage]
    if len(prior_job_ids) != len(set(prior_job_ids)):
        raise ValueError("prior_resource_evidence_records contains a duplicate PBS job")
    prior_stage = {stage: 0.0 for stage in config.resources.stage_node_hours}
    for item in prior_usage:
        if item["resource_type"] == "rt_HF":
            prior_stage[str(item["budget_stage"])] += float(item["node_hours"])
    stage_after_job = dict(prior_stage)
    if resource_type == "rt_HF":
        stage_after_job[stage] += requested_node_hours
    stage_overages = {
        approved_stage: max(
            0.0,
            stage_after_job[approved_stage] - config.resources.stage_node_hours[approved_stage],
        )
        for approved_stage in config.resources.stage_node_hours
    }
    return {
        "requested_node_hours": requested_node_hours,
        "cpu_after_job": sum(
            float(item["node_hours"])
            for item in prior_usage
            if item["resource_type"] == "rt_HC"
        ) + (requested_node_hours if resource_type == "rt_HC" else 0.0),
        "h200_after_job": sum(
            float(item["node_hours"])
            for item in prior_usage
            if item["resource_type"] == "rt_HF"
        ) + (requested_node_hours if resource_type == "rt_HF" else 0.0),
        "stage_after_job": stage_after_job,
        "reserve_after_job": sum(stage_overages.values()),
    }


def render_pbs(config: ExperimentConfig, job_manifest: Path, policy_snapshot: Path, output_path: Path) -> str:
    experiment_id = config.require_assigned_id()
    manifest = json.loads(job_manifest.read_text(encoding="utf-8"))
    policy = json.loads(policy_snapshot.read_text(encoding="utf-8"))
    expected_root = str(config.experiment_root())
    required = {
        "experiment_id",
        "experiment_slug",
        "job_name",
        "group",
        "queue",
        "billing_mode",
        "rate_class",
        "resource_type",
        "resource_class",
        "nodes",
        "cpus_per_node",
        "gpus_per_node",
        "requested_walltime",
        "array_size",
        "max_array_concurrency",
        "priority",
        "main_command",
        "output_root",
        "budget_stage",
        "execution_venue",
        "venue_evidence",
        "predicted_new_files_upper_bound",
        "predicted_new_bytes_upper_bound",
        "many_file_workload",
        "large_output_workload",
        "cpu_workers",
        "thread_plan",
        "io_concurrency",
        "expected_bottleneck",
        "special_spot",
        "submit_account",
        "responsible_person",
        "experiment_owner",
        "user_execution_approval_ref",
        "team_approval_ref",
        "plan_sha256",
        "prior_resource_evidence_records",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"job manifest is missing required fields: {sorted(missing)}")
    failures = []
    if manifest["experiment_id"] != experiment_id or manifest["experiment_slug"] != config.identity.slug:
        failures.append("experiment identity mismatch")
    if manifest["group"] != config.identity.group:
        failures.append("ABCI group mismatch")
    if manifest["output_root"] != expected_root:
        failures.append("job output root is not the canonical experiment root")
    if not str(manifest["job_name"]).startswith(f"{experiment_id}_"):
        failures.append("job name lacks experiment ID prefix")
    if manifest["queue"] not in policy.get("allowed_queues", []):
        failures.append("queue is not present in the transient policy snapshot")
    if manifest["billing_mode"] not in policy.get("allowed_billing_modes", []):
        failures.append("billing mode is not allowed by the transient policy snapshot")
    if manifest["resource_type"] not in policy.get("allowed_resource_types", []):
        failures.append("resource type is not allowed by the transient policy snapshot")
    if manifest["rate_class"] not in policy.get("allowed_rate_classes", []):
        failures.append("rate class is not allowed by the transient policy snapshot")
    resource_shape = policy.get("resource_shapes", {}).get(manifest["resource_type"], {})
    if not isinstance(resource_shape, dict) or (
        manifest["cpus_per_node"] != resource_shape.get("cpus_per_node")
        or manifest["gpus_per_node"] != resource_shape.get("gpus_per_node")
    ):
        failures.append("CPU/GPU shape does not match current scheduler facts")
    command = manifest["main_command"]
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        failures.append("main_command must be a non-empty argv list")
    if manifest["nodes"] != 1:
        failures.append("approved pilot permits exactly one node per job")
    try:
        runtime_stage = config.runtime_stage_for_manifest(str(manifest["runtime_stage"]))
        runtime_profile = config.runtime_profile_for_stage(runtime_stage)
    except (KeyError, ValueError) as error:
        failures.append(f"runtime stage/profile is invalid: {error}")
        runtime_profile = None
    if runtime_profile is not None:
        if runtime_profile.resource_class == "h200" and (
            manifest["resource_type"] != "rt_HF"
            or manifest["resource_class"] != "multi_gpu_single_node"
            or int(manifest["gpus_per_node"]) != 8
        ):
            failures.append("H200 runtime profile requires rt_HF with 8 GPUs")
        if runtime_profile.resource_class == "cpu_only" and (
            manifest["resource_type"] != "rt_HC"
            or manifest["resource_class"] != "cpu_only"
            or int(manifest["gpus_per_node"]) != 0
        ):
            failures.append("CPU runtime profile requires rt_HC with zero GPUs")
        if runtime_profile.resource_class == "local_cpu":
            failures.append("local runtime profile cannot be rendered as PBS")
    if manifest["execution_venue"] != "compute" or not isinstance(manifest["venue_evidence"], dict):
        failures.append("PBS rendering requires a documented COMPUTE_REQUIRED venue decision")
    else:
        venue_fields = {
            "abci_only_dependency",
            "expected_duration_minutes",
            "cpu_threads",
            "memory_gib",
            "transfer_bytes",
            "files_touched_upper_bound",
            "recursive_or_bulk_work",
            "why_login_is_insufficient",
        }
        if venue_fields - manifest["venue_evidence"].keys():
            failures.append("venue evidence is incomplete")
    projected = derive_projected_resource_usage(config, manifest)
    requested_node_hours = projected["requested_node_hours"]
    stage = manifest["budget_stage"]
    if manifest["resource_type"] == "rt_HC":
        if stage != "cpu":
            failures.append("rt_HC jobs must use the cpu budget stage")
        if int(manifest["gpus_per_node"]) != 0:
            failures.append("rt_HC job must request zero GPUs")
        if projected["cpu_after_job"] > config.resources.cpu_node_hours_max:
            failures.append("cumulative CPU request exceeds the 4 node-hour limit")
    elif manifest["resource_type"] != "rt_HF":
        failures.append("GPU pilot jobs must use rt_HF; no resource fallback is allowed")
    elif stage not in config.resources.stage_node_hours:
        failures.append("budget_stage is not one of the approved fixed H200 stages")
    else:
        if requested_node_hours > config.resources.stage_node_hours[stage]:
            failures.append(f"requested walltime exceeds the {stage} stage budget")
        reserve_total = projected["reserve_after_job"]
        h200_after_job = projected["h200_after_job"]
        if h200_after_job > config.resources.h200_node_hours_absolute_max:
            failures.append("cumulative request exceeds the 16 node-hour absolute limit")
        reserve_used = bool(manifest.get("reserve_used", False))
        reserve_reason = manifest.get("reserve_reason")
        needs_reserve = reserve_total > 0 or h200_after_job > config.resources.h200_node_hours_planned
        if needs_reserve and (not reserve_used or reserve_reason != "fixed_stage_completion"):
            failures.append("reserve may be used only to complete a fixed stage")
        if reserve_total > config.resources.h200_node_hours_reserve:
            failures.append("evidence-derived cumulative reserve exceeds 2 node-hours")
        if reserve_used and not needs_reserve:
            failures.append("reserve_used is true but no planned budget is exceeded")
    if failures:
        raise RuntimeError("PBS render gate failed: " + "; ".join(failures))

    directives = [
        "#!/bin/bash",
        f"#PBS -N {manifest['job_name']}",
        f"#PBS -P {manifest['group']}",
        f"#PBS -q {manifest['queue']}",
        f"#PBS -v RTYPE={manifest['resource_type']}",
        (
            f"#PBS -l select={manifest['nodes']}:ncpus={manifest['cpus_per_node']}:"
            f"ngpus={manifest['gpus_per_node']}"
        ),
        f"#PBS -l walltime={manifest['requested_walltime']}",
        f"#PBS -o {expected_root}/logs",
        "#PBS -j oe",
    ]
    if int(manifest.get("priority", 0)):
        directives.append(f"#PBS -p {int(manifest['priority'])}")
    array_size = int(manifest.get("array_size", 1))
    concurrency = int(manifest.get("max_array_concurrency", 1))
    if array_size > 1:
        directives.append(f"#PBS -J 0-{array_size - 1}%{concurrency}")

    body = [
        "",
        "set -euo pipefail",
        f"readonly EXP_DIR={shlex.quote(expected_root)}",
        'readonly JOB_LOG_DIR="$EXP_DIR/logs/$PBS_JOBID"',
        'mkdir -p "$JOB_LOG_DIR"',
        'exec > >(tee -a "$JOB_LOG_DIR/stdout.log") 2> >(tee -a "$JOB_LOG_DIR/stderr.log" >&2)',
        'export HF_HOME="$EXP_DIR/cache/huggingface"',
        'export WANDB_DIR="$EXP_DIR/wandb"',
        'export TMPDIR="$EXP_DIR/tmp/$PBS_JOBID"',
        'mkdir -p "$HF_HOME" "$WANDB_DIR" "$TMPDIR"',
        *(
            [
                f"export {name}={shlex.quote(value)}"
                for name, value in sorted(runtime_profile.environment.items())
            ]
            if runtime_profile is not None
            else []
        ),
        'trap \'qstat -fx -F json "$PBS_JOBID" > "$JOB_LOG_DIR/qstat-final.json" 2>&1 || true\' EXIT',
        f"cd {shlex.quote(expected_root)}",
        shlex.join(command),
        "",
    ]
    script = "\n".join(directives + body)
    if any(token in script for token in ("#$", "SGE_TASK_ID", "qrsh", "rt_F")):
        raise RuntimeError("ABCI 2.0/SGE syntax is forbidden")
    if output_path.exists():
        raise FileExistsError(f"PBS output is immutable: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script, encoding="utf-8")
    return script

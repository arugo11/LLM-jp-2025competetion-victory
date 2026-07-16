import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tv_gptoss120b.cluster_gate import verify_qsub_gate
from tv_gptoss120b.config import load_config

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def test_qsub_gate_requires_fresh_policy_and_approvals(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    root = tmp_path / "9999_tv-gptoss120b"
    root.mkdir()
    config = config.model_copy(update={
        "identity": config.identity.model_copy(update={"experiment_id": "9999"}),
        "paths": config.paths.model_copy(update={"experiment_root": str(root)}),
    })
    plan_hash = "a" * 64
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({
        "experiment_id": "9999",
        "plan_sha256": plan_hash,
        "user_approved": True,
        "foundation_meeting_approved": True,
    }))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "checked_at": datetime.now(UTC).isoformat(),
        "allowed_queues": ["verified-fixture"],
        "queue_verified": True,
        "quota_verified": True,
        "resource_shapes": {"rt_HF": {"cpus_per_node": 192, "gpus_per_node": 8}},
    }))
    manifest = root / "job-manifest.json"
    manifest.write_text(json.dumps({
        "job_name": "9999_generation-smoke",
        "experiment_id": "9999",
        "group": "gcg51557",
        "output_root": str(root),
        "plan_sha256": plan_hash,
        "queue": "verified-fixture",
        "resource_type": "rt_HF",
        "nodes": 1,
        "cpus_per_node": 192,
        "gpus_per_node": 8,
        "requested_walltime": "00:10:00",
        "budget_stage": "generation_validation",
        "runtime_stage": "generate",
        "prior_resource_evidence_records": [],
        "main_command": ["uv", "run", "tv-gptoss120b", "generate"],
        "predicted_new_bytes_upper_bound": 1000,
    }))
    pbs = root / "stage.pbs"
    pbs.write_text(
        "#PBS -N 9999_generation-smoke\n"
        "#PBS -P gcg51557\n"
        "#PBS -q verified-fixture\n"
        "#PBS -v RTYPE=rt_HF\n"
        "#PBS -l select=1:ncpus=192:ngpus=8\n"
        "#PBS -l walltime=00:10:00\n"
        "export MKL_NUM_THREADS=8\n"
        "export OMP_NUM_THREADS=8\n"
        "export OPENBLAS_NUM_THREADS=8\n"
        "export TOKENIZERS_PARALLELISM=false\n"
        f"EXP_DIR={config.experiment_root()}\n"
        "uv run tv-gptoss120b generate\n"
    )
    storage = root / "storage-audit.txt"
    storage.write_text(
        "audit_mode=bounded-pre-qsub\n"
        f"storage_audit_target={root}\n"
        f"created_at={datetime.now(UTC).isoformat()}\n"
        "target_group=gcg51557\n"
        "bytes=1\n"
        "inode=10\n"
        "scan_status=complete\n"
        "file_scan_limit=100000\n"
    )
    report = verify_qsub_gate(
        config,
        approval_record=approval,
        policy_snapshot=policy,
        job_manifest=manifest,
        pbs_script=pbs,
        storage_audit=storage,
        plan_sha256=plan_hash,
    )
    assert report["status"] == "PASS"
    assert report["projected_resource_usage"]["h200_after_job"] == pytest.approx(1 / 6)
    assert report["measured_storage"] == {"bytes": 1, "inodes": 10}
    assert report["projected_storage_bytes"] == 1001

    manifest_payload = json.loads(manifest.read_text())
    manifest_payload["predicted_new_bytes_upper_bound"] = 250_000_000_000
    manifest.write_text(json.dumps(manifest_payload))
    with pytest.raises(RuntimeError, match="250 GB"):
        verify_qsub_gate(
            config,
            approval_record=approval,
            policy_snapshot=policy,
            job_manifest=manifest,
            pbs_script=pbs,
            storage_audit=storage,
            plan_sha256=plan_hash,
        )
    manifest_payload["predicted_new_bytes_upper_bound"] = 1000
    manifest.write_text(json.dumps(manifest_payload))

    payload = json.loads(approval.read_text())
    payload["foundation_meeting_approved"] = False
    approval.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="approvals"):
        verify_qsub_gate(
            config,
            approval_record=approval,
            policy_snapshot=policy,
            job_manifest=manifest,
            pbs_script=pbs,
            storage_audit=storage,
            plan_sha256=plan_hash,
        )


def test_qsub_gate_allows_only_scoped_preemptible_reserved_smoke_without_team_approval(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    root = tmp_path / "9999_tv-gptoss120b"
    root.mkdir()
    config = config.model_copy(update={
        "identity": config.identity.model_copy(update={"experiment_id": "9999"}),
        "paths": config.paths.model_copy(update={"experiment_root": str(root)}),
    })
    plan_hash = "b" * 64
    policy_ref = "https://llmjp.slack.com/archives/C07U29EDTPD/p1784095799644289"
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({
        "experiment_id": "9999",
        "plan_sha256": plan_hash,
        "user_approved": True,
        "approval_scope": "opportunistic_reserved_smoke",
        "foundation_meeting_approved": False,
    }))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "checked_at": datetime.now(UTC).isoformat(),
        "allowed_queues": ["R-fixture"],
        "queue_verified": True,
        "quota_verified": True,
        "resource_shapes": {"rt_HF": {"cpus_per_node": 192, "gpus_per_node": 8}},
        "opportunistic_reserved_smoke": {
            "allowed": True,
            "max_walltime": "00:30:00",
            "source_ref": policy_ref,
        },
    }))
    manifest_payload = {
        "job_name": "9999_generation-smoke",
        "experiment_id": "9999",
        "group": "gcg51557",
        "output_root": str(root),
        "plan_sha256": plan_hash,
        "queue": "R-fixture",
        "billing_mode": "reserved",
        "resource_type": "rt_HF",
        "nodes": 1,
        "cpus_per_node": 192,
        "gpus_per_node": 8,
        "requested_walltime": "00:30:00",
        "budget_stage": "generation_validation",
        "runtime_stage": "generate",
        "prior_resource_evidence_records": [],
        "main_command": ["uv", "run", "tv-gptoss120b", "generate"],
        "predicted_new_bytes_upper_bound": 1000,
        "smoke": True,
        "array_size": 1,
        "max_array_concurrency": 1,
        "preemptible": True,
        "automatic_retry": False,
        "team_approval_ref": None,
        "opportunistic_policy_ref": policy_ref,
    }
    manifest = root / "job-manifest.json"
    manifest.write_text(json.dumps(manifest_payload))
    pbs = root / "stage.pbs"
    pbs.write_text(
        "#PBS -N 9999_generation-smoke\n"
        "#PBS -P gcg51557\n"
        "#PBS -q R-fixture\n"
        "#PBS -v RTYPE=rt_HF\n"
        "#PBS -l select=1:ncpus=192:ngpus=8\n"
        "#PBS -l walltime=00:30:00\n"
        "export MKL_NUM_THREADS=8\n"
        "export OMP_NUM_THREADS=8\n"
        "export OPENBLAS_NUM_THREADS=8\n"
        "export TOKENIZERS_PARALLELISM=false\n"
        f"EXP_DIR={config.experiment_root()}\n"
        "uv run tv-gptoss120b generate\n"
    )
    storage = root / "storage-audit.txt"
    storage.write_text(
        "audit_mode=bounded-pre-qsub\n"
        f"storage_audit_target={root}\n"
        f"created_at={datetime.now(UTC).isoformat()}\n"
        "target_group=gcg51557\n"
        "bytes=1\n"
        "inode=10\n"
        "scan_status=complete\n"
        "file_scan_limit=100000\n"
    )

    report = verify_qsub_gate(
        config,
        approval_record=approval,
        policy_snapshot=policy,
        job_manifest=manifest,
        pbs_script=pbs,
        storage_audit=storage,
        plan_sha256=plan_hash,
    )
    assert report["approval_mode"] == "opportunistic_reserved_smoke"

    for field, unsafe_value in (
        ("smoke", False),
        ("automatic_retry", True),
        ("preemptible", False),
        ("requested_walltime", "00:30:01"),
        ("array_size", 2),
    ):
        unsafe = dict(manifest_payload)
        unsafe[field] = unsafe_value
        manifest.write_text(json.dumps(unsafe))
        with pytest.raises(RuntimeError, match="narrow opportunistic"):
            verify_qsub_gate(
                config,
                approval_record=approval,
                policy_snapshot=policy,
                job_manifest=manifest,
                pbs_script=pbs,
                storage_audit=storage,
                plan_sha256=plan_hash,
            )

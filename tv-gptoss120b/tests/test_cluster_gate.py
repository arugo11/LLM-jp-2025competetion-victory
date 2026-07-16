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
        "captured_at": datetime.now(UTC).isoformat(),
        "queue": "verified-fixture",
        "queue_verified": True,
        "quota_verified": True,
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
        "cpus_per_node": 96,
        "gpus_per_node": 8,
        "requested_walltime": "00:10:00",
        "budget_stage": "generation_validation",
        "runtime_stage": "generate",
        "prior_resource_evidence_records": [],
        "main_command": ["uv", "run", "tv-gptoss120b", "generate"],
    }))
    pbs = root / "stage.pbs"
    pbs.write_text(
        "#PBS -N 9999_generation-smoke\n"
        "#PBS -P gcg51557\n"
        "#PBS -q verified-fixture\n"
        "#PBS -v RTYPE=rt_HF\n"
        "#PBS -l select=1:ncpus=96:ngpus=8\n"
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
        "audit_mode=deep\n"
        f"storage_audit_target={root}\n"
        f"created_at={datetime.now(UTC).isoformat()}\n"
        "target_group=gcg51557\n"
        "bytes=1\n"
        "inode=10\n"
        "scan_status=complete\n"
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

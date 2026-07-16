import json
from pathlib import Path

import pytest

import tv_gptoss120b.pbs as pbs_module
from tv_gptoss120b.config import load_config
from tv_gptoss120b.pbs import render_pbs

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def gpu_job_payload(config, **updates) -> dict:
    payload = {
        "experiment_id": "9999",
        "experiment_slug": "tv-gptoss120b",
        "job_name": "9999_generation-smoke",
        "group": "gcg51557",
        "queue": "verified-fixture",
        "billing_mode": "reserved",
        "resource_type": "rt_HF",
        "nodes": 1,
        "cpus_per_node": 96,
        "gpus_per_node": 8,
        "requested_walltime": "00:10:00",
        "array_size": 1,
        "max_array_concurrency": 1,
        "priority": 0,
        "main_command": ["uv", "run", "tv-gptoss120b", "generate"],
        "output_root": str(config.experiment_root()),
        "budget_stage": "generation_validation",
        "runtime_stage": "generate",
        "prior_resource_evidence_records": [],
        "execution_venue": "compute",
        "venue_evidence": {
            "abci_only_dependency": "H200",
            "expected_duration_minutes": 120,
            "cpu_threads": 96,
            "memory_gib": 128,
            "transfer_bytes": 1000,
            "files_touched_upper_bound": 100,
            "recursive_or_bulk_work": True,
            "why_login_is_insufficient": "GPU inference",
        },
        "predicted_new_files": 100,
        "predicted_new_bytes": 1000000,
        "cpu_workers": 8,
        "thread_plan": "8 workers x 1 thread",
        "io_concurrency": 4,
        "submit_account": "fixture-account",
        "responsible_person": "fixture-responsible",
        "experiment_owner": "fixture-owner",
        "user_execution_approval_ref": "fixture-user-approval",
        "team_approval_ref": "fixture-team-approval",
        "plan_sha256": "a" * 64,
    }
    payload.update(updates)
    return payload


def test_pbs_is_rendered_only_from_manifest_and_policy(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    config = config.model_copy(update={"identity": config.identity.model_copy(update={"experiment_id": "9999"})})
    manifest = tmp_path / "job.json"
    manifest.write_text(json.dumps(gpu_job_payload(config)))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "queues": ["verified-fixture"],
        "allowed_billing_modes": ["reserved"],
        "allowed_resource_types": ["rt_HF"],
    }))
    output = tmp_path / "job.pbs"
    script = render_pbs(config, manifest, policy, output)
    assert "#PBS -v RTYPE=rt_HF" in script
    assert "#PBS -l select=1:ncpus=96:ngpus=8" in script
    assert str(config.experiment_root()) in script
    assert "#$" not in script
    assert "export OMP_NUM_THREADS=8" in script


def test_cpu_job_uses_separate_four_hour_budget(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    config = config.model_copy(update={"identity": config.identity.model_copy(update={"experiment_id": "9999"})})
    manifest = tmp_path / "cpu.json"
    manifest.write_text(json.dumps({
        "experiment_id": "9999",
        "experiment_slug": "tv-gptoss120b",
        "job_name": "9999_publish",
        "group": "gcg51557",
        "queue": "verified-cpu",
        "billing_mode": "reserved",
        "resource_type": "rt_HC",
        "nodes": 1,
        "cpus_per_node": 32,
        "gpus_per_node": 0,
        "requested_walltime": "01:00:00",
        "main_command": ["uv", "run", "tv-gptoss120b", "publish"],
        "output_root": str(config.experiment_root()),
        "budget_stage": "cpu",
        "runtime_stage": "publish",
        "prior_resource_evidence_records": [],
        "execution_venue": "compute",
        "venue_evidence": {
            "abci_only_dependency": "ABCI model outputs",
            "expected_duration_minutes": 60,
            "cpu_threads": 32,
            "memory_gib": 128,
            "transfer_bytes": 1000000000,
            "files_touched_upper_bound": 1000,
            "recursive_or_bulk_work": True,
            "why_login_is_insufficient": "full model validation",
        },
        "predicted_new_files": 1000,
        "predicted_new_bytes": 1000000000,
        "cpu_workers": 32,
        "thread_plan": "32 workers x 1 thread",
        "io_concurrency": 8,
        "submit_account": "fixture-account",
        "responsible_person": "fixture-responsible",
        "experiment_owner": "fixture-owner",
        "user_execution_approval_ref": "fixture-user-approval",
        "team_approval_ref": "fixture-team-approval",
        "plan_sha256": "a" * 64,
    }))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "queues": ["verified-cpu"],
        "allowed_billing_modes": ["reserved"],
        "allowed_resource_types": ["rt_HC"],
    }))
    script = render_pbs(config, manifest, policy, tmp_path / "cpu.pbs")
    assert "ngpus=0" in script


def test_pbs_rejects_caller_authored_cumulative_resource_totals(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    config = config.model_copy(update={"identity": config.identity.model_copy(update={"experiment_id": "9999"})})
    manifest = tmp_path / "job.json"
    manifest.write_text(json.dumps(gpu_job_payload(
        config,
        cumulative_h200_node_hours_after_job=0.1,
    )))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "queues": ["verified-fixture"],
        "allowed_billing_modes": ["reserved"],
        "allowed_resource_types": ["rt_HF"],
    }))

    with pytest.raises(ValueError, match="caller-authored cumulative resource fields are forbidden"):
        render_pbs(config, manifest, policy, tmp_path / "job.pbs")


def test_pbs_uses_evidence_derived_prior_usage_for_reserve_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG)
    config = config.model_copy(update={"identity": config.identity.model_copy(update={"experiment_id": "9999"})})
    evidence = tmp_path / "prior-resource-evidence.json"
    manifest = tmp_path / "job.json"
    manifest.write_text(json.dumps(gpu_job_payload(
        config,
        prior_resource_evidence_records=[str(evidence)],
    )))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "queues": ["verified-fixture"],
        "allowed_billing_modes": ["reserved"],
        "allowed_resource_types": ["rt_HF"],
    }))
    monkeypatch.setattr(pbs_module, "derive_resource_usage", lambda _config, path: {
        "pbs_job_id": "123.abci",
        "resource_type": "rt_HF",
        "budget_stage": "generation_validation",
        "node_hours": 1.9,
    } if path == evidence else None)

    with pytest.raises(RuntimeError, match="reserve may be used only to complete a fixed stage"):
        render_pbs(config, manifest, policy, tmp_path / "job.pbs")

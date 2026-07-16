import json
from pathlib import Path

import pytest
import yaml

from tv_gptoss120b.config import load_config
from tv_gptoss120b.hashing import sha256_file, sha256_value
from tv_gptoss120b.manifest import ArtifactRef, StageManifest
from tv_gptoss120b.release_audit import REQUIRED_STAGE_MANIFESTS, audit_release_evidence
from tv_gptoss120b.resource_evidence import (
    derive_pre_qsub_storage_metrics,
    derive_resource_usage,
    write_resource_evidence_record,
)

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def make_evidence(tmp_path: Path) -> tuple[object, Path, Path]:
    root = tmp_path / "9999_tv-gptoss120b"
    root.mkdir()
    config = load_config(CONFIG)
    config = config.model_copy(update={
        "identity": config.identity.model_copy(update={"experiment_id": "9999"}),
        "paths": config.paths.model_copy(update={"experiment_root": str(root)}),
    })
    manifest = root / "job-manifest.json"
    manifest.write_text(json.dumps({
        "experiment_id": "9999",
        "job_name": "9999_generate",
        "output_root": str(root),
        "resource_type": "rt_HF",
        "budget_stage": "generation_validation",
        "nodes": 1,
        "requested_walltime": "01:00:00",
        "reserve_used": False,
        "submit_account": "fixture-account",
        "responsible_person": "fixture-person",
        "experiment_owner": "arugo11",
        "user_execution_approval_ref": "fixture-user-approval",
        "team_approval_ref": "fixture-team-approval",
    }), encoding="utf-8")
    pbs = root / "job.pbs"
    pbs.write_text(
        "#!/bin/bash\n"
        "#PBS -N 9999_generate\n"
        "#PBS -P gcg51557\n"
        "#PBS -v RTYPE=rt_HF\n"
        "#PBS -l select=1:ncpus=8:ngpus=8\n"
        "#PBS -l walltime=01:00:00\n",
        encoding="utf-8",
    )
    qstat = root / "qstat-final.json"
    qstat.write_text(json.dumps({
        "Jobs": {
            "123.abci": {
                "Job_Name": "9999_generate",
                "job_state": "F",
                "Exit_status": 0,
                "Resource_List": {"nodect": 1, "walltime": "01:00:00"},
                "resources_used": {"walltime": "00:30:00"},
            }
        }
    }), encoding="utf-8")
    storage = root / "storage-audit.txt"
    storage.write_text(
        "audit_mode=deep\n"
        f"storage_audit_target={root}\n"
        "created_at=2026-07-17T06:00:00+09:00\n"
        "target_group=gcg51557\n"
        "bytes=123456\n"
        "inode=789\n"
        "scan_status=complete\n",
        encoding="utf-8",
    )
    evidence = root / "resource-evidence.json"
    write_resource_evidence_record(
        config,
        pbs_job_id="123.abci",
        job_manifest_path=manifest,
        pbs_script_path=pbs,
        qstat_final_path=qstat,
        storage_audit_path=storage,
        output_path=evidence,
    )
    return config, evidence, qstat


def test_resource_usage_is_derived_from_qstat_and_deep_storage_audit(tmp_path: Path) -> None:
    config, evidence, _ = make_evidence(tmp_path)

    usage = derive_resource_usage(config, evidence)

    assert usage["node_hours"] == pytest.approx(0.5)
    assert usage["measured_storage_bytes"] == 123456
    assert usage["measured_storage_inodes"] == 789
    assert set(usage["evidence_sha256"]) == {
        "record", "job_manifest", "pbs_script", "qstat_final", "storage_audit"
    }


def test_resource_usage_rejects_qstat_tampering(tmp_path: Path) -> None:
    config, evidence, qstat = make_evidence(tmp_path)
    payload = json.loads(qstat.read_text(encoding="utf-8"))
    payload["Jobs"]["123.abci"]["resources_used"]["walltime"] = "00:01:00"
    qstat.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="qstat evidence SHA-256 mismatch"):
        derive_resource_usage(config, evidence)


def test_resource_usage_accepts_narrow_opportunistic_reserved_smoke(tmp_path: Path) -> None:
    config, evidence, _ = make_evidence(tmp_path)
    record = json.loads(evidence.read_text(encoding="utf-8"))
    manifest = Path(record["job_manifest"]["path"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.update({
        "team_approval_ref": None,
        "billing_mode": "reserved",
        "smoke": True,
        "preemptible": True,
        "automatic_retry": False,
        "array_size": 1,
        "max_array_concurrency": 1,
        "opportunistic_policy_ref": "https://llmjp.slack.com/archives/C07U29EDTPD/p1784095799644289",
    })
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    record["job_manifest"]["sha256"] = sha256_file(manifest)
    evidence.write_text(json.dumps(record), encoding="utf-8")

    assert derive_resource_usage(config, evidence)["node_hours"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("billing_mode", "spot"),
        ("smoke", False),
        ("preemptible", False),
        ("automatic_retry", True),
        ("nodes", 2),
        ("array_size", 2),
        ("max_array_concurrency", 2),
        ("opportunistic_policy_ref", ""),
    ],
)
def test_resource_usage_rejects_broadened_opportunistic_scope(
    tmp_path: Path,
    field: str,
    unsafe_value: object,
) -> None:
    config, evidence, _ = make_evidence(tmp_path)
    record = json.loads(evidence.read_text(encoding="utf-8"))
    manifest = Path(record["job_manifest"]["path"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.update({
        "team_approval_ref": None,
        "billing_mode": "reserved",
        "smoke": True,
        "preemptible": True,
        "automatic_retry": False,
        "array_size": 1,
        "max_array_concurrency": 1,
        "opportunistic_policy_ref": "https://llmjp.slack.com/archives/C07U29EDTPD/p1784095799644289",
        field: unsafe_value,
    })
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    record["job_manifest"]["sha256"] = sha256_file(manifest)
    evidence.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="approval evidence is invalid"):
        derive_resource_usage(config, evidence)


def test_pre_qsub_storage_audit_is_distinct_from_release_deep_audit(tmp_path: Path) -> None:
    root = tmp_path / "9999_tv-gptoss120b"
    root.mkdir()
    config = load_config(CONFIG).model_copy(update={
        "identity": load_config(CONFIG).identity.model_copy(update={"experiment_id": "9999"}),
        "paths": load_config(CONFIG).paths.model_copy(update={"experiment_root": str(root)}),
    })
    audit = root / "pre-qsub-storage-audit.txt"
    audit.write_text(
        "audit_mode=bounded-pre-qsub\n"
        f"storage_audit_target={root}\n"
        "created_at=2026-07-17T06:00:00+09:00\n"
        "target_group=gcg51557\n"
        "bytes=1234\n"
        "inode=12\n"
        "scan_status=complete\n"
        "file_scan_limit=100000\n",
        encoding="utf-8",
    )

    assert derive_pre_qsub_storage_metrics(config, audit)["bytes"] == 1234


def test_pre_qsub_storage_accepts_stronger_complete_deep_audit(tmp_path: Path) -> None:
    root = tmp_path / "9999_tv-gptoss120b"
    root.mkdir()
    base = load_config(CONFIG)
    config = base.model_copy(update={
        "identity": base.identity.model_copy(update={"experiment_id": "9999"}),
        "paths": base.paths.model_copy(update={"experiment_root": str(root)}),
    })
    audit = root / "deep-storage-audit.txt"
    audit.write_text(
        "audit_mode=deep\n"
        f"storage_audit_target={root}\n"
        "created_at=2026-07-17T08:39:22+09:00\n"
        "target_group=gcg51557\n"
        "bytes=89089269134\n"
        "inode=50145\n"
        "scan_status=complete\n",
        encoding="utf-8",
    )

    metrics = derive_pre_qsub_storage_metrics(config, audit)

    assert metrics == {
        "audit_mode": "deep",
        "bytes": 89089269134,
        "inodes": 50145,
        "created_at": "2026-07-17T08:39:22+09:00",
        "file_scan_limit": None,
    }


def make_release_lineage(config, evidence: Path) -> tuple[Path, dict, dict[str, Path]]:
    root = config.experiment_root()
    config_path = root / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    artifact = root / "preserved-output.bin"
    artifact.write_bytes(b"immutable-stage-evidence")
    artifact_ref = ArtifactRef(
        name="fixture",
        kind="output-file",
        uri=str(artifact),
        digest=sha256_file(artifact),
    )
    positive_update = {"changed_keys": 1, "max_abs_diff": 0.125}
    def preprocessing(kind: str, arm: str) -> dict:
        return {
            "kind": kind,
            "arm": arm,
            "smoke": False,
            "assistant_only_loss": kind == "sft",
            "packing": False,
            "config_sha256": "6" * 64,
            "model_revision": "7" * 40,
            "chat_template_sha256": "8" * 64,
            "prompt_template_sha256": "9" * 64,
            "template_kwargs_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "source_records_sha256": "c" * 64,
            "dataset_records_sha256": "d" * 64,
            "dataset_payload_sha256": "e" * 64,
            "label_mask_policy_sha256": "1" * 64,
            "required_resource_class": "cpu_only",
            "preprocessing_workers": 8,
        }
    metrics = {
        "generate": {"spec_count": 512, "backend": "vllm", "smoke": False},
        "curate": {
            "quality_gate": "passed",
            "dataset_sha256": "1" * 64,
            "aime_reference_sha256": "2" * 64,
        },
        "difficulty": {
            "hardening_gate": True,
            "spec_count": 80,
            "pooled_delta": config.difficulty.pooled_delta_min,
            "thinking_delta": 0.11,
            "instruct_delta": 0.12,
        },
        "preprocess": {
            "artifact_count": 4,
            "smoke": False,
            "artifacts": {
                f"{arm}-{kind}-preprocessed": preprocessing(kind, arm)
                for arm in ("thinking", "instruct")
                for kind in ("sft", "grpo")
            },
        },
        "thinking-sft": {
            "weight_update": positive_update,
            "global_step": 10,
            "expected_steps": 10,
            "preprocessing_manifest_sha256": "f" * 64,
            "preprocessing": preprocessing("sft", "thinking"),
        },
        "instruct-sft": {
            "weight_update": positive_update,
            "global_step": 10,
            "expected_steps": 10,
            "preprocessing_manifest_sha256": "f" * 64,
            "preprocessing": preprocessing("sft", "instruct"),
        },
        "thinking-grpo": {
            "adapter_weight_update": positive_update,
            "global_step": 32,
            "preprocessing_manifest_sha256": "f" * 64,
            "preprocessing": preprocessing("grpo", "thinking"),
            "merge_parity": {
                "greedy_outputs_match": True,
                "max_abs_logit_diff": 0.001,
                "tolerance": 0.005,
            },
        },
        "instruct-grpo": {
            "adapter_weight_update": positive_update,
            "global_step": 32,
            "preprocessing_manifest_sha256": "f" * 64,
            "preprocessing": preprocessing("grpo", "instruct"),
            "merge_parity": {
                "greedy_outputs_match": True,
                "max_abs_logit_diff": 0.001,
                "tolerance": 0.005,
            },
        },
        "evaluate": {
            "normalized_sample_count": 960,
            "matched_config_sha256": "3" * 64,
            "harness_manifest_sha256": "4" * 64,
            "harness_worktree_sha256": "5" * 64,
            "summary": {
                "scientific_aime_gate": False,
                "arms": {"thinking": {}, "instruct": {}},
            },
        },
    }
    stage_paths = {}
    config_sha = sha256_file(config_path)
    for name, stage in REQUIRED_STAGE_MANIFESTS.items():
        runtime_stage = config.runtime_stage_for_manifest(name)
        runtime_profile_name = config.runtime.stage_profile[runtime_stage]
        runtime_profile = config.runtime.profiles[runtime_profile_name]
        manifest = StageManifest(
            experiment_id="9999",
            stage=stage,
            config_sha256=config_sha,
            git_sha="a" * 40,
            command=["fixture"],
            environment={"python": "3.12"},
            inputs=[artifact_ref],
            outputs=[artifact_ref],
            metrics=metrics[name],
            provenance={
                "runtime_stage": runtime_stage,
                "runtime_profile": runtime_profile_name,
                "runtime_resource_class": runtime_profile.resource_class,
                "runtime_profile_sha256": sha256_value(runtime_profile.model_dump(mode="json")),
                "runtime_declared_environment_sha256": sha256_value(runtime_profile.environment),
            },
        )
        path = root / f"{name}-stage-manifest.json"
        path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
        stage_paths[name] = path
    audit_paths = {}
    for name in ("pre_qsub", "pre_aime", "pre_publication"):
        path = root / f"{name}.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "audit_name": name,
            "status": "PASS",
            "experiment_id": "9999",
            "config_sha256": config_sha,
            "git_sha": "a" * 40,
            "checked_at": "2026-07-17T06:00:00+09:00",
            "checks": {"fixture": True},
        }), encoding="utf-8")
        audit_paths[name] = path
    lineage = {
        "stage_manifests": {name: str(path) for name, path in stage_paths.items()},
        "self_audits": {name: str(path) for name, path in audit_paths.items()},
        "resource_usage_records": [str(evidence)],
    }
    return config_path, lineage, stage_paths


def test_release_audit_recomputes_resource_and_stage_evidence(tmp_path: Path) -> None:
    config, evidence, _ = make_evidence(tmp_path)
    config_path, lineage, _ = make_release_lineage(config, evidence)

    report = audit_release_evidence(config, config_path, lineage)

    assert report["status"] == "PASS"
    assert report["h200_node_hours"] == pytest.approx(0.5)
    assert report["max_storage_bytes"] == 123456
    assert report["max_storage_inodes"] == 789


def test_release_audit_rejects_missing_grpo_weight_update(tmp_path: Path) -> None:
    config, evidence, _ = make_evidence(tmp_path)
    config_path, lineage, stage_paths = make_release_lineage(config, evidence)
    path = stage_paths["thinking-grpo"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metrics"]["adapter_weight_update"]["changed_keys"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="nonzero weight update"):
        audit_release_evidence(config, config_path, lineage)


def test_release_audit_rejects_missing_preprocessing_lineage(tmp_path: Path) -> None:
    config, evidence, _ = make_evidence(tmp_path)
    config_path, lineage, stage_paths = make_release_lineage(config, evidence)
    path = stage_paths["thinking-sft"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["metrics"]["preprocessing"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="preprocessing contract"):
        audit_release_evidence(config, config_path, lineage)


def test_release_audit_rejects_difficulty_below_scientific_gate(tmp_path: Path) -> None:
    config, evidence, _ = make_evidence(tmp_path)
    config_path, lineage, stage_paths = make_release_lineage(config, evidence)
    path = stage_paths["difficulty"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metrics"]["pooled_delta"] = config.difficulty.pooled_delta_min - 0.001
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="pooled delta is below"):
        audit_release_evidence(config, config_path, lineage)


def test_release_audit_rejects_missing_preprocessed_artifact(tmp_path: Path) -> None:
    config, evidence, _ = make_evidence(tmp_path)
    config_path, lineage, stage_paths = make_release_lineage(config, evidence)
    path = stage_paths["preprocess"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["metrics"]["artifacts"]["thinking-grpo-preprocessed"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="preprocess artifacts must be exactly"):
        audit_release_evidence(config, config_path, lineage)

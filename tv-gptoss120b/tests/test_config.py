from pathlib import Path

import pytest

from tv_gptoss120b.config import load_config

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def test_config_loads_with_approved_experiment_id() -> None:
    config = load_config(CONFIG)
    assert config.require_assigned_id() == "0399"
    assert config.identity.slug == "tv-gptoss120b"
    assert config.evaluation.task == "swallow|aime_N4|0|0"
    assert config.evaluation.aime_2025.split == "train"


def test_runtime_manifest_stage_aliases_are_canonical() -> None:
    config = load_config(CONFIG)
    assert config.runtime_stage_for_manifest("thinking-sft") == "sft"
    assert config.runtime_stage_for_manifest("evaluate-aime-matched") == "evaluate"
    assert config.runtime_stage_for_manifest("fixture") == "audit"
    assert config.runtime_stage_for_manifest("fixture-generate-smoke") == "audit"
    assert config.runtime_stage_for_manifest("fixture-curate") == "audit"
    assert config.publication.license_metadata is None
    assert config.require_assigned_id() == "0399"


def test_config_budget_is_exact() -> None:
    config = load_config(CONFIG)
    assert config.resources.h200_node_hours_planned == 14
    assert config.resources.h200_node_hours_reserve == 2
    assert config.resources.h200_node_hours_absolute_max == 16


def test_runtime_profiles_separate_cpu_gpu_and_publication() -> None:
    config = load_config(CONFIG)
    runtime = config.runtime
    assert runtime.stage_profile["preprocess"] == "cpu_pipeline"
    assert runtime.profiles["cpu_pipeline"].gpus_per_node == 0
    assert runtime.stage_profile["evaluate"] == "h200_pipeline"
    assert runtime.profiles["h200_pipeline"].hub_upload_allowed is False
    assert runtime.profiles["cpu_publication"].hub_upload_allowed is True
    assert runtime.profiles["cpu_publication"].venue == "decision_required"
    assert "queue" in runtime.transient_cluster_fields


def test_coordination_records_the_aime25_split_conflict() -> None:
    config = load_config(CONFIG)
    assert config.coordination.team_evaluation_harness.status == "slack-confirmed"
    assert config.coordination.common_evaluation_procedure.status == "unresolved"
    assert config.coordination.aime_2025_split.status == "user-fixed"
    assert config.evaluation.aime_2025.split == "train"


def test_production_contract_rejects_setting_drift() -> None:
    config = load_config(CONFIG)
    config = config.model_copy(update={"sft": config.sft.model_copy(update={"num_train_epochs": 2.0})})
    with pytest.raises(ValueError, match="sft.num_train_epochs"):
        config.require_production_contract()


def test_production_contract_rejects_previously_unchecked_drift() -> None:
    config = load_config(CONFIG)
    config = config.model_copy(update={
        "generation": config.generation.model_copy(update={"solution_temperature": 0.25})
    })
    with pytest.raises(ValueError, match="unapproved production setting drift"):
        config.require_production_contract()


def test_production_contract_accepts_the_checked_in_yaml() -> None:
    load_config(CONFIG).require_production_contract()

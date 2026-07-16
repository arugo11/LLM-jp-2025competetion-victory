from pathlib import Path

import pytest

from tv_gptoss120b.config import load_config
from tv_gptoss120b.tracking import ArtifactRun, require_immutable_artifact_ref

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


@pytest.mark.parametrize("ref", ["dataset", "dataset:latest", "dataset:best", "dataset:v1-extra"])
def test_wandb_latest_input_is_forbidden(ref: str) -> None:
    with pytest.raises(ValueError):
        require_immutable_artifact_ref(ref)


def test_wandb_version_input_is_allowed() -> None:
    require_immutable_artifact_ref("validated-paired-dataset:v0")


def test_wandb_offline_artifact_contains_real_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    config = load_config(CONFIG)
    config = config.model_copy(update={"identity": config.identity.model_copy(update={"experiment_id": "9999"})})
    payload = tmp_path / "paired.jsonl"
    payload.write_text('{"spec_id":"fixture"}\n')
    with ArtifactRun(config, stage="fixture", mode="offline", run_config={"fixture": True}) as tracked:
        tracked.use("generation-spec:v0")
        tracked.log_files(
            name="validated-paired-dataset",
            kind="dataset",
            files=[payload],
            metadata={"fixture": True},
        )
    journals = list((tmp_path / "wandb-offline-lineage").glob("*.json"))
    assert len(journals) == 1
    text = journals[0].read_text()
    assert '"generation-spec:v0"' in text
    assert '"artifact_name": "validated-paired-dataset"' in text
    assert '"content_sha256"' in text

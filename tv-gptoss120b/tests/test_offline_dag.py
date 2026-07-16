from pathlib import Path

import pytest

from tv_gptoss120b.config import load_config
from tv_gptoss120b.tracking import (
    REQUIRED_ARTIFACTS,
    REQUIRED_EDGES,
    ArtifactRun,
    _verify_offline_entry_content,
    verify_offline_journal_dag,
)

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def test_all_stage_offline_wandb_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    config = load_config(CONFIG)
    config = config.model_copy(update={"identity": config.identity.model_copy(update={"experiment_id": "9999"})})
    for name, kind in REQUIRED_ARTIFACTS.items():
        payload = tmp_path / f"{name}.json"
        payload.write_text(f'{{"artifact":"{name}"}}\n')
        with ArtifactRun(config, stage=f"fixture-{name}", mode="offline", run_config={"fixture": True}) as tracked:
            for input_name in REQUIRED_EDGES.get(name, set()):
                tracked.use_logical(input_name)
            tracked.log_files(name=name, kind=kind, files=[payload], metadata={"fixture": True})
    journals = list((tmp_path / "wandb-offline-lineage").glob("*.json"))
    result = verify_offline_journal_dag(journals)
    assert result == {"status": "PASS", "artifact_count": len(REQUIRED_ARTIFACTS)}


def test_offline_journal_rejects_changed_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    config = load_config(CONFIG)
    config = config.model_copy(update={"identity": config.identity.model_copy(update={"experiment_id": "9999"})})
    payload = tmp_path / "code.json"
    payload.write_text('{"version":1}\n')
    with ArtifactRun(config, stage="fixture-code", mode="offline", run_config={"fixture": True}) as tracked:
        tracked.log_files(name="code", kind="code", files=[payload], metadata={"fixture": True})
    payload.write_text('{"version":2}\n')
    journal = next((tmp_path / "wandb-offline-lineage").glob("*.json"))
    with pytest.raises(RuntimeError, match="content changed"):
        import json

        _verify_offline_entry_content(json.loads(journal.read_text()))

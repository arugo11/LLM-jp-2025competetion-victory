import json
from pathlib import Path

import pytest

from tv_gptoss120b.cards import render_dataset_card, render_model_card
from tv_gptoss120b.config import load_config
from tv_gptoss120b.hashing import sha256_value
from tv_gptoss120b.publication import (
    card_metadata,
    make_public,
    validate_card,
    validate_private_validation_report,
)

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def test_cards_have_provenance_without_license_metadata(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    config = config.model_copy(update={"identity": config.identity.model_copy(update={"experiment_id": "9999"})})
    dataset = tmp_path / "dataset.md"
    dataset.write_text(render_dataset_card(config, manifest={"quality_gate": "passed"}, wandb_artifact="x:v0"))
    model = tmp_path / "model.md"
    model.write_text(render_model_card(
        config,
        arm="thinking",
        dataset_commit="a" * 40,
        aime_summary={"delta": 0.0},
        merge_summary={"parity": True},
        wandb_artifact="y:v0",
    ))
    for card in (dataset, model):
        metadata = card_metadata(card)
        assert not {"license", "license_name", "license_link"}.intersection(metadata)
        validate_card(card, required_phrases=("W&B Artifact", "このrepoのlicenseは未指定"))
        text = card.read_text(encoding="utf-8")
        assert "元repoのライセンス表示" in text
        assert "model card metadataは`apache-2.0`" in text
        assert "本repoへの新しいライセンス付与を意味しません" in text
    assert "AIME24/25はcontamination検査の参照にだけ使い" in dataset.read_text(encoding="utf-8")


def test_card_rejects_license_metadata(tmp_path: Path) -> None:
    card = tmp_path / "README.md"
    card.write_text("---\nlicense: other\n---\nこのrepoのlicenseは未指定\nW&B Artifact")
    with pytest.raises(ValueError, match="license metadata"):
        validate_card(card, required_phrases=("W&B Artifact",))


class FakeApi:
    def __init__(self, sha: str, private: bool = True) -> None:
        self.sha = sha
        self.private = private
        self.updated = False

    def repo_info(self, **kwargs):
        return type("RepoInfo", (), {"sha": self.sha, "private": self.private})()

    def update_repo_settings(self, **kwargs) -> None:
        self.updated = True
        self.private = kwargs["private"]


def test_public_gate_requires_validated_private_head() -> None:
    validated = "a" * 40
    api = FakeApi(validated)
    make_public(repo_kind="dataset", repo_id="argo11/fixture", validated_commit=validated, api=api)
    assert api.updated is True

    with pytest.raises(RuntimeError, match="head changed"):
        make_public(
            repo_kind="dataset",
            repo_id="argo11/fixture",
            validated_commit=validated,
            api=FakeApi("b" * 40),
        )
    with pytest.raises(RuntimeError, match="still be private"):
        make_public(
            repo_kind="dataset",
            repo_id="argo11/fixture",
            validated_commit=validated,
            api=FakeApi(validated, private=False),
        )


def test_private_dataset_validation_report_is_schema_checked(tmp_path: Path) -> None:
    checksums = {"README.md": "a" * 64, "paired.jsonl": "b" * 64}
    payload = {
        "repo_kind": "dataset",
        "repo_id": "argo11/fixture",
        "commit_sha": "c" * 40,
        "file_count": 2,
        "checksums": checksums,
        "tree_sha256": sha256_value(checksums),
        "private_head_verified_before_and_after": True,
        "loaded_rows": 512,
        "viewer": {"viewer": True},
    }
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    validated = validate_private_validation_report(
        report,
        repo_kind="dataset",
        repo_id="argo11/fixture",
        commit_sha="c" * 40,
    )
    assert validated["loaded_rows"] == 512

    payload["viewer"] = {"viewer": False}
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="viewer"):
        validate_private_validation_report(
            report,
            repo_kind="dataset",
            repo_id="argo11/fixture",
            commit_sha="c" * 40,
        )

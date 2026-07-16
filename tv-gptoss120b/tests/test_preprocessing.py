import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset

from tv_gptoss120b.cli import grpo, preprocess, sft
from tv_gptoss120b.config import load_config
from tv_gptoss120b.preprocessing import (
    PREPROCESSING_MANIFEST,
    bind_pretokenized_grpo_prompts,
    load_verified_preprocessed_dataset,
    tokenize_assistant_only,
    write_preprocessed_dataset_artifact,
)

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_assistant_tokens_mask"] is True
        return {
            "input_ids": [10, 11, 12, 13],
            "attention_mask": [1, 1, 1, 1],
            "assistant_masks": [0, 0, 1, 1],
        }


class PrefixFallbackTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        if len(messages) == 1:
            return [10, 11]
        return {"input_ids": [10, 11, 12, 13], "attention_mask": [1, 1, 1, 1]}


def test_assistant_only_loss_mask() -> None:
    row = tokenize_assistant_only(FakeTokenizer(), "problem", "solution", 4096)
    assert row["labels"] == [-100, -100, 12, 13]


def test_assistant_only_prefix_fallback() -> None:
    row = tokenize_assistant_only(PrefixFallbackTokenizer(), "problem", "solution", 4096)
    assert row["labels"] == [-100, -100, 12, 13]


class InvalidPrefixTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        if len(messages) == 1:
            return [99]
        return {
            "input_ids": [10, 11, 12, 13],
            "attention_mask": [1, 1, 1, 1],
            "assistant_masks": [0, 0, 0, 0],
        }


def test_invalid_prompt_prefix_fails() -> None:
    with pytest.raises(ValueError, match="not an exact prefix"):
        tokenize_assistant_only(InvalidPrefixTokenizer(), "problem", "solution", 4096)


class ArtifactTokenizer:
    chat_template = "fixture-official-chat-template"
    pad_token_id = 0
    truncation_side = "right"
    padding_side = "right"

    def apply_chat_template(self, conversation, *args, **kwargs):
        raise AssertionError("pretokenized GRPO prompt unexpectedly reached raw tokenization")


def assigned_config():
    config = load_config(CONFIG)
    return config.model_copy(update={
        "identity": config.identity.model_copy(update={"experiment_id": "9999"}),
    })


def make_sft_artifact(tmp_path: Path):
    config = assigned_config()
    rows = [
        {
            "spec_id": f"sft-{index:03d}",
            "source_content_sha256": f"{index + 1:064x}",
            "input_ids": [10, 11, 12],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 11, 12],
        }
        for index in range(16)
    ]
    source = tmp_path / "curated.jsonl"
    source.write_text("fixture\n", encoding="utf-8")
    output = tmp_path / "sft-preprocessed"
    write_preprocessed_dataset_artifact(
        config,
        kind="sft",
        arm="thinking",
        dataset=Dataset.from_list(rows),
        output_dir=output,
        tokenizer=ArtifactTokenizer(),
        source_path=source,
        source_records=[{"spec_id": row["spec_id"]} for row in rows],
        max_length=config.sft.max_length,
        smoke=True,
    )
    return config, output


def test_sft_preprocessing_artifact_is_verified_before_consumption(tmp_path: Path) -> None:
    config, output = make_sft_artifact(tmp_path)

    dataset, manifest = load_verified_preprocessed_dataset(
        config,
        kind="sft",
        arm="thinking",
        dataset_dir=output,
        tokenizer=ArtifactTokenizer(),
        smoke=True,
    )

    assert len(dataset) == 16
    assert manifest.row_count == 16
    assert manifest.assistant_only_loss is True
    assert "source_path" not in type(manifest).model_fields


def test_sft_preprocessing_artifact_rejects_manifest_record_tampering(tmp_path: Path) -> None:
    config, output = make_sft_artifact(tmp_path)
    path = output / PREPROCESSING_MANIFEST
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset_records_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="record SHA-256 mismatch"):
        load_verified_preprocessed_dataset(
            config,
            kind="sft",
            arm="thinking",
            dataset_dir=output,
            tokenizer=ArtifactTokenizer(),
            smoke=True,
        )


def test_preprocessing_artifact_rejects_chat_template_drift(tmp_path: Path) -> None:
    config, output = make_sft_artifact(tmp_path)

    class DriftedTokenizer(ArtifactTokenizer):
        chat_template = "drifted-template"

    with pytest.raises(ValueError, match="manifest/config/tokenizer drift"):
        load_verified_preprocessed_dataset(
            config,
            kind="sft",
            arm="thinking",
            dataset_dir=output,
            tokenizer=DriftedTokenizer(),
            smoke=True,
        )


def test_grpo_tokenizer_uses_materialized_prompt_ids_without_raw_tokenization() -> None:
    prompt_a = [{"role": "user", "content": "A"}]
    prompt_b = [{"role": "user", "content": "B"}]
    dataset = Dataset.from_list([
        {"prompt": prompt_a, "prompt_input_ids": [11, 12]},
        {"prompt": prompt_b, "prompt_input_ids": [21, 22, 23]},
    ])
    tokenizer = ArtifactTokenizer()
    bind_pretokenized_grpo_prompts(tokenizer, dataset)

    encoded = tokenizer.apply_chat_template(
        [prompt_a, prompt_b],
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    )

    assert encoded["input_ids"].tolist() == [[0, 11, 12], [21, 22, 23]]
    assert encoded["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]

    with pytest.raises(ValueError, match="absent from the verified pretokenized dataset"):
        tokenizer.apply_chat_template(
            [[{"role": "user", "content": "unknown"}]],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )


def test_gpu_training_commands_accept_only_preprocessed_dataset_inputs() -> None:
    sft_parameters = inspect.signature(sft).parameters
    grpo_parameters = inspect.signature(grpo).parameters

    assert "preprocessed" in sft_parameters and "curated" not in sft_parameters and "num_proc" not in sft_parameters
    assert "preprocessed" in grpo_parameters and "curated" not in grpo_parameters


def test_preprocess_command_materializes_and_tracks_four_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = assigned_config()
    curated = tmp_path / "curated.jsonl"
    curated.write_text("fixture\n", encoding="utf-8")
    refs = tmp_path / "artifact-refs.json"
    refs.write_text(json.dumps({
        "code": "code:v0",
        "validated-paired-dataset": "validated-paired-dataset:v0",
    }), encoding="utf-8")
    output = tmp_path / "preprocessed"
    prepared = []
    tracked = []
    stages = []

    def prepare(kind, arm, directory, workers=None):
        directory.mkdir()
        (directory / "payload.bin").write_bytes(f"{arm}-{kind}".encode())
        prepared.append((arm, kind, workers))
        return SimpleNamespace(model_dump=lambda mode: {
            "arm": arm,
            "kind": kind,
            "required_resource_class": "cpu_only",
            "smoke": False,
        })

    monkeypatch.setattr("tv_gptoss120b.cli._load", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        "tv_gptoss120b.cli.prepare_sft_dataset",
        lambda config, arm, curated, directory, workers, smoke: prepare("sft", arm, directory, workers),
    )
    monkeypatch.setattr(
        "tv_gptoss120b.cli.prepare_grpo_dataset",
        lambda config, arm, curated, directory, smoke: prepare("grpo", arm, directory),
    )
    monkeypatch.setattr(
        "tv_gptoss120b.cli._stage_manifest",
        lambda config_path, config, stage, output_dir, metrics, input_paths: stages.append((stage, metrics)),
    )
    monkeypatch.setattr(
        "tv_gptoss120b.cli._track_output",
        lambda *args, **kwargs: tracked.append(kwargs["name"]) or f"{kwargs['name']}:v0",
    )

    preprocess(
        config_path=CONFIG,
        curated=curated,
        output_dir=output,
        smoke=False,
        artifact_refs=refs,
        wandb_mode="online",
    )

    expected = [
        "thinking-sft-preprocessed",
        "thinking-grpo-preprocessed",
        "instruct-sft-preprocessed",
        "instruct-grpo-preprocessed",
    ]
    assert tracked == expected
    assert prepared == [
        ("thinking", "sft", 8),
        ("thinking", "grpo", None),
        ("instruct", "sft", 8),
        ("instruct", "grpo", None),
    ]
    assert stages[0][0] == "preprocess"
    assert stages[0][1]["artifact_count"] == 4
    receipt = json.loads((output / "artifact-receipt.json").read_text(encoding="utf-8"))
    assert set(receipt["artifact_refs"]) == {
        "code",
        "validated-paired-dataset",
        *expected,
    }


def test_preprocessing_worker_count_is_bounded_by_yaml_cpu_profile(tmp_path: Path) -> None:
    config = assigned_config()
    source = tmp_path / "curated.jsonl"
    source.write_text("fixture\n", encoding="utf-8")
    rows = [{
        "spec_id": "sft-000",
        "source_content_sha256": "1" * 64,
        "input_ids": [10, 11],
        "attention_mask": [1, 1],
        "labels": [-100, 11],
    }]
    with pytest.raises(ValueError, match="exceeds the YAML CPU profile"):
        write_preprocessed_dataset_artifact(
            config,
            kind="sft",
            arm="thinking",
            dataset=Dataset.from_list(rows),
            output_dir=tmp_path / "too-many-workers",
            tokenizer=ArtifactTokenizer(),
            source_path=source,
            source_records=[{"spec_id": "sft-000"}],
            max_length=config.sft.max_length,
            smoke=True,
            num_proc=9,
        )

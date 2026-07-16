from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path
from types import MethodType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import ExperimentConfig
from .hashing import canonical_json, sha256_file, sha256_value
from .jsonl import read_jsonl
from .prompts import EVALUATION_PROMPT
from .schema import CuratedRecord

PREPROCESSING_MANIFEST = "preprocessing-manifest.json"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class PreprocessingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    kind: Literal["sft", "grpo"]
    experiment_id: str
    arm: Literal["thinking", "instruct"]
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    model_repo_id: str
    model_revision: str
    tokenizer_class: str
    tokenizer_truncation_side: Literal["left", "right"]
    tokenizer_padding_side: Literal["left", "right"]
    chat_template_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_template_sha256: str = Field(pattern=SHA256_PATTERN)
    template_kwargs_sha256: str = Field(pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_records_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_records_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    columns: list[str]
    row_count: int = Field(gt=0)
    max_length: int = Field(gt=0)
    assistant_only_loss: bool
    packing: bool
    label_mask_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    preprocessing_workers: int = Field(gt=0)
    required_resource_class: Literal["cpu_only"]
    datasets_version: str
    transformers_version: str
    smoke: bool


def _tokenizer_contract(tokenizer) -> tuple[str, str]:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizer must expose a non-empty official chat_template")
    return tokenizer.__class__.__name__, sha256_value(template)


def _dataset_rows(dataset) -> list[dict[str, Any]]:
    return [dataset[index] for index in range(len(dataset))]


def _payload_sha256(path: Path) -> str:
    entries = [
        {"path": str(item.relative_to(path)), "sha256": sha256_file(item)}
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.name != PREPROCESSING_MANIFEST
    ]
    if not entries:
        raise ValueError(f"preprocessed dataset payload is empty: {path}")
    return sha256_value(entries)


def _expected_columns(kind: Literal["sft", "grpo"]) -> list[str]:
    if kind == "sft":
        return ["spec_id", "source_content_sha256", "input_ids", "attention_mask", "labels"]
    return ["spec_id", "source_content_sha256", "prompt", "prompt_input_ids", "expected_answer"]


def _validate_rows(kind: Literal["sft", "grpo"], rows: list[dict[str, Any]], max_length: int) -> None:
    spec_ids = [row.get("spec_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in spec_ids) or len(spec_ids) != len(set(spec_ids)):
        raise ValueError("preprocessed dataset spec_id values must be non-empty and unique")
    for row in rows:
        source_digest = row.get("source_content_sha256")
        if not isinstance(source_digest, str) or re.fullmatch(SHA256_PATTERN, source_digest) is None:
            raise ValueError("preprocessed row lacks a source content SHA-256")
        if kind == "sft":
            input_ids = row.get("input_ids")
            attention_mask = row.get("attention_mask")
            labels = row.get("labels")
            if not all(isinstance(values, list) and values for values in (input_ids, attention_mask, labels)):
                raise ValueError("SFT rows require non-empty input_ids/attention_mask/labels lists")
            if not len(input_ids) == len(attention_mask) == len(labels) <= max_length:
                raise ValueError("SFT token arrays disagree or exceed max_length")
            if any(type(value) is not int for values in (input_ids, attention_mask, labels) for value in values):
                raise ValueError("SFT token arrays must contain integers")
            if any(value != 1 for value in attention_mask):
                raise ValueError("materialized SFT rows must be unpadded with an all-one attention mask")
            if not any(value == -100 for value in labels) or not any(value != -100 for value in labels):
                raise ValueError("SFT labels must contain both masked prompt and trainable assistant tokens")
            if any(label != -100 and label != token for token, label in zip(input_ids, labels, strict=True)):
                raise ValueError("SFT labels may contain only -100 or the corresponding input token")
        else:
            prompt = row.get("prompt")
            prompt_ids = row.get("prompt_input_ids")
            if (
                not isinstance(prompt, list)
                or not prompt
                or not isinstance(prompt_ids, list)
                or not prompt_ids
                or len(prompt_ids) > max_length
                or any(type(value) is not int for value in prompt_ids)
                or not isinstance(row.get("expected_answer"), str)
                or not row["expected_answer"]
            ):
                raise ValueError("GRPO rows require prompt, pretokenized prompt IDs, and expected_answer")


def write_preprocessed_dataset_artifact(
    config: ExperimentConfig,
    *,
    kind: Literal["sft", "grpo"],
    arm: Literal["thinking", "instruct"],
    dataset,
    output_dir: Path,
    tokenizer,
    source_path: Path,
    source_records: list[dict[str, Any]],
    max_length: int,
    smoke: bool,
    num_proc: int | None = None,
) -> PreprocessingManifest:
    if output_dir.exists():
        raise FileExistsError(f"preprocessed output is immutable: {output_dir}")
    max_workers = config.runtime.profiles["cpu_pipeline"].cpu_workers
    if num_proc is not None and num_proc > max_workers:
        raise ValueError(f"preprocessing num_proc exceeds the YAML CPU profile: {num_proc} > {max_workers}")
    expected_columns = _expected_columns(kind)
    if dataset.column_names != expected_columns:
        raise ValueError(
            f"preprocessed {kind} columns must be exactly {expected_columns}, got {dataset.column_names}"
        )
    rows = _dataset_rows(dataset)
    _validate_rows(kind, rows, max_length)
    save_kwargs = {}
    if num_proc is not None and num_proc > 1:
        save_kwargs["num_proc"] = min(num_proc, max(1, len(dataset)))
    dataset.save_to_disk(output_dir, **save_kwargs)
    tokenizer_class, template_sha256 = _tokenizer_contract(tokenizer)
    model = getattr(config.models, arm)
    manifest = PreprocessingManifest(
        kind=kind,
        experiment_id=config.require_assigned_id(),
        arm=arm,
        config_sha256=sha256_value(config.model_dump(mode="json")),
        model_repo_id=model.repo_id,
        model_revision=model.revision,
        tokenizer_class=tokenizer_class,
        tokenizer_truncation_side=tokenizer.truncation_side,
        tokenizer_padding_side=tokenizer.padding_side,
        chat_template_sha256=template_sha256,
        prompt_template_sha256=sha256_value(EVALUATION_PROMPT),
        template_kwargs_sha256=sha256_value({"reasoning_effort": "medium"}),
        source_sha256=sha256_file(source_path),
        source_records_sha256=sha256_value(source_records),
        dataset_records_sha256=sha256_value(rows),
        dataset_payload_sha256=_payload_sha256(output_dir),
        columns=expected_columns,
        row_count=len(rows),
        max_length=max_length,
        assistant_only_loss=kind == "sft",
        packing=False,
        label_mask_policy_sha256=sha256_value(
            "official assistant token mask, otherwise exact prompt-prefix difference; no heuristic boundary"
        ),
        preprocessing_workers=num_proc or 1,
        required_resource_class="cpu_only",
        datasets_version=version("datasets"),
        transformers_version=version("transformers"),
        smoke=smoke,
    )
    with (output_dir / PREPROCESSING_MANIFEST).open("xb") as handle:
        handle.write(canonical_json(manifest.model_dump(mode="json")) + b"\n")
    return manifest


def tokenize_assistant_only(
    tokenizer,
    problem: str,
    solution: str,
    max_length: int,
    *,
    arm: Literal["thinking", "instruct"] = "instruct",
    expected_answer: str | None = None,
) -> dict:
    if arm == "thinking" and expected_answer is None:
        raise ValueError("thinking arm requires expected_answer for the final channel")
    assistant = (
        {"role": "assistant", "thinking": solution, "content": rf"\boxed{{{expected_answer}}}"}
        if arm == "thinking"
        else {"role": "assistant", "content": solution}
    )
    messages = [
        {"role": "user", "content": EVALUATION_PROMPT.format(problem=problem)},
        assistant,
    ]
    template_kwargs = {"reasoning_effort": "medium"}
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        truncation=True,
        max_length=max_length,
        add_generation_prompt=False,
        **template_kwargs,
    )
    assistant_mask = encoded.get("assistant_masks") or encoded.get("assistant_tokens_mask")
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
    if assistant_mask is None or not any(assistant_mask):
        prompt_ids = list(tokenizer.apply_chat_template(
            messages[:1],
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=max_length,
            **template_kwargs,
        ))
        if len(prompt_ids) >= len(input_ids) or input_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("official chat template prompt is not an exact prefix of the full training sequence")
        assistant_mask = [0] * len(prompt_ids) + [1] * (len(input_ids) - len(prompt_ids))
    assistant_mask = list(assistant_mask)
    if not (len(input_ids) == len(attention_mask) == len(assistant_mask)):
        raise ValueError("tokenization output lengths disagree")
    labels = [token if is_assistant else -100 for token, is_assistant in zip(input_ids, assistant_mask, strict=True)]
    if all(label == -100 for label in labels):
        raise ValueError("assistant-only loss mask contains no trainable tokens")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def prepare_sft_dataset(
    config: ExperimentConfig,
    arm: Literal["thinking", "instruct"],
    curated_path: Path,
    output_dir: Path,
    num_proc: int,
    *,
    smoke: bool = False,
) -> PreprocessingManifest:
    from datasets import Dataset
    from transformers import AutoTokenizer

    if num_proc < 1:
        raise ValueError("SFT preprocessing num_proc must be positive")
    max_workers = config.runtime.profiles["cpu_pipeline"].cpu_workers
    if num_proc > max_workers:
        raise ValueError(f"SFT preprocessing num_proc exceeds the YAML CPU profile: {num_proc} > {max_workers}")
    model = getattr(config.models, arm)
    tokenizer = AutoTokenizer.from_pretrained(model.repo_id, revision=model.revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    rows = [
        row for row in read_jsonl(curated_path, CuratedRecord)
        if row.generator_model == "generator_120b" and row.split == "sft"
    ]
    if len(rows) < config.validation.valid_pair_min.sft:
        raise ValueError(f"SFT dataset below gate: {len(rows)}")
    if smoke:
        rows = sorted(rows, key=lambda row: row.spec_id)[:16]
        if len(rows) != 16:
            raise ValueError(f"SFT smoke requires exactly 16 examples, got {len(rows)}")
    source_records = [row.model_dump(mode="json") for row in rows]
    source = Dataset.from_list(source_records)

    def encode(row: dict) -> dict:
        return {
            "spec_id": row["spec_id"],
            "source_content_sha256": row["content_sha256"],
            **tokenize_assistant_only(
                tokenizer,
                row["problem"],
                row["solution_cot"],
                config.sft.max_length,
                arm=arm,
                expected_answer=row["expected_answer"],
            ),
        }

    tokenized = source.map(encode, num_proc=num_proc, remove_columns=source.column_names, desc=f"tokenize-{arm}")
    tokenized = tokenized.select_columns(_expected_columns("sft"))
    return write_preprocessed_dataset_artifact(
        config,
        kind="sft",
        arm=arm,
        dataset=tokenized,
        output_dir=output_dir,
        tokenizer=tokenizer,
        source_path=curated_path,
        source_records=source_records,
        max_length=config.sft.max_length,
        smoke=smoke,
        num_proc=num_proc,
    )


def prepare_grpo_dataset(
    config: ExperimentConfig,
    arm: Literal["thinking", "instruct"],
    curated_path: Path,
    output_dir: Path,
    *,
    smoke: bool = False,
) -> PreprocessingManifest:
    from datasets import Dataset
    from transformers import AutoTokenizer

    model = getattr(config.models, arm)
    tokenizer = AutoTokenizer.from_pretrained(model.repo_id, revision=model.revision, trust_remote_code=False)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    rows = [
        row for row in read_jsonl(curated_path, CuratedRecord)
        if row.generator_model == "generator_120b" and row.split == "grpo"
    ]
    if len(rows) < config.validation.valid_pair_min.grpo:
        raise ValueError(f"GRPO dataset below gate: {len(rows)}")
    if smoke:
        rows = sorted(rows, key=lambda row: row.spec_id)[:8]
        if len(rows) != 8:
            raise ValueError(f"GRPO smoke requires exactly 8 prompts, got {len(rows)}")
    source_records = [row.model_dump(mode="json") for row in rows]
    dataset_rows = []
    for row in rows:
        prompt = [{"role": "user", "content": EVALUATION_PROMPT.format(problem=row.problem)}]
        prompt_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=config.grpo.max_prompt_length,
            reasoning_effort="medium",
        )
        dataset_rows.append({
            "spec_id": row.spec_id,
            "source_content_sha256": row.content_sha256,
            "prompt": prompt,
            "prompt_input_ids": list(prompt_ids),
            "expected_answer": row.expected_answer,
        })
    dataset = Dataset.from_list(dataset_rows).select_columns(_expected_columns("grpo"))
    return write_preprocessed_dataset_artifact(
        config,
        kind="grpo",
        arm=arm,
        dataset=dataset,
        output_dir=output_dir,
        tokenizer=tokenizer,
        source_path=curated_path,
        source_records=source_records,
        max_length=config.grpo.max_prompt_length,
        smoke=smoke,
    )


def load_verified_preprocessed_dataset(
    config: ExperimentConfig,
    *,
    kind: Literal["sft", "grpo"],
    arm: Literal["thinking", "instruct"],
    dataset_dir: Path,
    tokenizer,
    smoke: bool,
):
    from datasets import load_from_disk

    manifest_path = dataset_dir / PREPROCESSING_MANIFEST
    manifest = PreprocessingManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    tokenizer_class, template_sha256 = _tokenizer_contract(tokenizer)
    model = getattr(config.models, arm)
    max_length = config.sft.max_length if kind == "sft" else config.grpo.max_prompt_length
    expected = {
        "kind": kind,
        "experiment_id": config.require_assigned_id(),
        "arm": arm,
        "config_sha256": sha256_value(config.model_dump(mode="json")),
        "model_repo_id": model.repo_id,
        "model_revision": model.revision,
        "tokenizer_class": tokenizer_class,
        "tokenizer_truncation_side": tokenizer.truncation_side,
        "tokenizer_padding_side": tokenizer.padding_side,
        "chat_template_sha256": template_sha256,
        "prompt_template_sha256": sha256_value(EVALUATION_PROMPT),
        "template_kwargs_sha256": sha256_value({"reasoning_effort": "medium"}),
        "columns": _expected_columns(kind),
        "max_length": max_length,
        "assistant_only_loss": kind == "sft",
        "packing": False,
        "label_mask_policy_sha256": sha256_value(
            "official assistant token mask, otherwise exact prompt-prefix difference; no heuristic boundary"
        ),
        "required_resource_class": "cpu_only",
        "datasets_version": version("datasets"),
        "transformers_version": version("transformers"),
        "smoke": smoke,
    }
    actual = manifest.model_dump(mode="json")
    drift = [key for key, value in expected.items() if actual[key] != value]
    if drift:
        raise ValueError(f"preprocessing manifest/config/tokenizer drift: {drift}")
    if _payload_sha256(dataset_dir) != manifest.dataset_payload_sha256:
        raise ValueError("preprocessed dataset payload SHA-256 mismatch")
    dataset = load_from_disk(str(dataset_dir))
    if dataset.column_names != manifest.columns or len(dataset) != manifest.row_count:
        raise ValueError("preprocessed dataset columns/row count mismatch")
    rows = _dataset_rows(dataset)
    _validate_rows(kind, rows, max_length)
    if sha256_value(rows) != manifest.dataset_records_sha256:
        raise ValueError("preprocessed dataset record SHA-256 mismatch")
    minimum_rows = (16 if smoke else config.validation.valid_pair_min.sft) if kind == "sft" else (
        8 if smoke else config.validation.valid_pair_min.grpo
    )
    if len(dataset) < minimum_rows:
        raise ValueError(f"preprocessed {kind} dataset below gate: {len(dataset)} < {minimum_rows}")
    return dataset, manifest


def bind_pretokenized_grpo_prompts(tokenizer, dataset) -> None:
    """Use CPU-materialized prompt IDs for the exact TRL 0.27 conversational generation call."""
    import torch
    from transformers import BatchEncoding

    lookup = {
        sha256_value(row["prompt"]): list(row["prompt_input_ids"])
        for row in _dataset_rows(dataset)
    }
    if len(lookup) != len(dataset):
        raise ValueError("GRPO prompt content must be unique for pretokenized lookup")
    original = tokenizer.apply_chat_template

    def apply_chat_template(self, conversation, *args, **kwargs):
        is_batch = bool(conversation) and isinstance(conversation[0], list)
        conversations = conversation if is_batch else [conversation]
        is_generation_tokenization = (
            kwargs.get("tokenize", True) and kwargs.get("add_generation_prompt") is True
        )
        if not is_generation_tokenization:
            return original(conversation, *args, **kwargs)
        missing = [sha256_value(item) for item in conversations if sha256_value(item) not in lookup]
        if missing:
            raise ValueError(
                "GRPO generation prompt is absent from the verified pretokenized dataset: "
                f"{missing[:3]}"
            )
        ids = [lookup[sha256_value(item)] for item in conversations]
        max_length = max(len(item) for item in ids)
        padded = [[self.pad_token_id] * (max_length - len(item)) + item for item in ids]
        masks = [[0] * (max_length - len(item)) + [1] * len(item) for item in ids]
        if kwargs.get("return_tensors") == "pt":
            return BatchEncoding({
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            })
        payload: dict[str, Any] = {
            "input_ids": padded if is_batch else padded[0],
            "attention_mask": masks if is_batch else masks[0],
        }
        return BatchEncoding(payload) if kwargs.get("return_dict") else payload["input_ids"]

    tokenizer.apply_chat_template = MethodType(apply_chat_template, tokenizer)

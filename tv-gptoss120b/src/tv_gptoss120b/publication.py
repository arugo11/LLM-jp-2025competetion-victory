from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal

import yaml

from .config import ExperimentConfig
from .hashing import sha256_file, sha256_value

FORBIDDEN_LICENSE_KEYS = {"license", "license_name", "license_link"}
FORBIDDEN_WEIGHT_PARTS = ("optimizer", "scheduler", "zero_pp_rank", "global_step", "rng_state")
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def card_metadata(card: Path) -> dict[str, Any]:
    text = card.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"card lacks YAML frontmatter: {card}")
    try:
        _, frontmatter, _ = text.split("---", 2)
    except ValueError as error:
        raise ValueError(f"malformed card frontmatter: {card}") from error
    value = yaml.safe_load(frontmatter) or {}
    if not isinstance(value, dict):
        raise ValueError(f"card metadata must be a mapping: {card}")
    return value


def validate_card(card: Path, *, required_phrases: tuple[str, ...]) -> None:
    metadata = card_metadata(card)
    forbidden = FORBIDDEN_LICENSE_KEYS.intersection(metadata)
    if forbidden:
        raise ValueError(f"license metadata is forbidden by the approved plan: {sorted(forbidden)}")
    text = card.read_text(encoding="utf-8")
    if "このrepoのlicenseは未指定" not in text:
        raise ValueError("card must state that this repository's license is unspecified")
    missing = [phrase for phrase in required_phrases if phrase not in text]
    if missing:
        raise ValueError(f"card is missing required provenance: {missing}")


def assert_cpu_publication_context() -> None:
    if os.environ.get("PBS_RESOURCE_TYPE") == "rt_HF":
        raise RuntimeError("Hugging Face publication must run on a CPU node, not rt_HF")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and visible not in {"-1", "NoDevFiles"}:
        raise RuntimeError("Hugging Face publication refuses a GPU-visible process")


def validate_model_tree(model_dir: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in model_dir.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if any(part in lower for part in FORBIDDEN_WEIGHT_PARTS):
            raise ValueError(f"optimizer/FSDP state must not be published: {path}")
        checksums[str(path.relative_to(model_dir))] = sha256_file(path)
    if not any(name.endswith(".safetensors") for name in checksums):
        raise ValueError(f"merged full model has no safetensors weights: {model_dir}")
    return checksums


def upload_private_then_validate(
    config: ExperimentConfig,
    *,
    repo_kind: Literal["dataset", "model"],
    repo_id: str,
    source_dir: Path,
    card: Path,
    required_phrases: tuple[str, ...],
    validation_dir: Path,
    api: Any | None = None,
) -> str:
    """Upload one immutable private snapshot. Visibility is changed by a separate release call."""
    assert_cpu_publication_context()
    config.require_assigned_id()
    validate_card(card, required_phrases=required_phrases)
    if repo_kind == "model":
        validate_model_tree(source_dir)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_kind, private=True, exist_ok=True)
    before_upload = api.repo_info(repo_id=repo_id, repo_type=repo_kind)
    if getattr(before_upload, "private", None) is not True:
        raise RuntimeError("Hugging Face upload target must be private before upload")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_kind,
        folder_path=str(source_dir),
        commit_message=f"add: {config.require_assigned_id()} validated private release candidate",
    )
    sha = getattr(commit, "oid", None)
    if not isinstance(sha, str) or not SHA40.fullmatch(sha):
        raise RuntimeError("Hugging Face upload did not return an immutable commit SHA")
    info = api.repo_info(repo_id=repo_id, repo_type=repo_kind, revision=sha, files_metadata=True)
    current = api.repo_info(repo_id=repo_id, repo_type=repo_kind)
    if getattr(current, "private", None) is not True or getattr(current, "sha", None) != sha:
        raise RuntimeError("Hugging Face upload target changed or became public before validation")
    siblings = {item.rfilename for item in info.siblings}
    if "README.md" not in siblings:
        raise RuntimeError("private repository validation failed: README.md is absent")
    validate_private_snapshot(
        repo_kind=repo_kind,
        repo_id=repo_id,
        commit_sha=sha,
        source_dir=source_dir,
        validation_dir=validation_dir,
        api=api,
    )
    return sha


def _tree_checksums(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    }


def _validate_dataset_viewer(repo_id: str) -> dict[str, Any]:
    from huggingface_hub import get_token

    url = "https://datasets-server.huggingface.co/is-valid?dataset=" + urllib.parse.quote(repo_id, safe="")
    request = urllib.request.Request(url)
    token = get_token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read())
    if not payload.get("viewer", False):
        raise RuntimeError(f"Hugging Face dataset viewer is not valid yet: {payload}")
    return payload


def validate_private_snapshot(
    *,
    repo_kind: Literal["dataset", "model"],
    repo_id: str,
    commit_sha: str,
    source_dir: Path,
    validation_dir: Path,
    api: Any | None = None,
) -> dict[str, Any]:
    assert_cpu_publication_context()
    if validation_dir.exists():
        raise FileExistsError(f"HF validation directory is immutable: {validation_dir}")
    from huggingface_hub import HfApi, snapshot_download

    api = api or HfApi()
    before = api.repo_info(repo_id=repo_id, repo_type=repo_kind)
    if getattr(before, "sha", None) != commit_sha or getattr(before, "private", None) is not True:
        raise RuntimeError("Hugging Face private snapshot HEAD changed before redownload validation")

    downloaded = Path(snapshot_download(
        repo_id=repo_id,
        repo_type=repo_kind,
        revision=commit_sha,
        local_dir=validation_dir,
        force_download=True,
    ))
    source_checksums = _tree_checksums(source_dir)
    downloaded_checksums = _tree_checksums(downloaded)
    if source_checksums != downloaded_checksums:
        raise RuntimeError("Hugging Face redownload checksum mismatch")
    evidence: dict[str, Any] = {
        "repo_kind": repo_kind,
        "repo_id": repo_id,
        "commit_sha": commit_sha,
        "file_count": len(downloaded_checksums),
        "checksums": downloaded_checksums,
        "tree_sha256": sha256_value(downloaded_checksums),
        "private_head_verified_before_and_after": True,
    }
    if repo_kind == "dataset":
        from datasets import load_dataset

        data_file = downloaded / "paired.jsonl"
        if not data_file.is_file():
            raise RuntimeError("published dataset snapshot must contain paired.jsonl at repository root")
        loaded = load_dataset("json", data_files=str(data_file), split="train")
        if len(loaded) == 0:
            raise RuntimeError("published dataset loads as an empty dataset")
        evidence["loaded_rows"] = len(loaded)
        evidence["viewer"] = _validate_dataset_viewer(repo_id)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(downloaded, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            downloaded,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
        )
        evidence["model_class"] = model.__class__.__name__
        evidence["tokenizer_class"] = tokenizer.__class__.__name__
        evidence["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        del model
    after = api.repo_info(repo_id=repo_id, repo_type=repo_kind)
    if getattr(after, "sha", None) != commit_sha or getattr(after, "private", None) is not True:
        raise RuntimeError("Hugging Face private snapshot HEAD changed during validation")
    report = validation_dir / "hf-private-validation.json"
    report.write_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return evidence


def validate_private_validation_report(
    path: Path,
    *,
    repo_kind: Literal["dataset", "model"],
    repo_id: str,
    commit_sha: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    common = {
        "repo_kind",
        "repo_id",
        "commit_sha",
        "file_count",
        "checksums",
        "tree_sha256",
        "private_head_verified_before_and_after",
    }
    kind_fields = (
        {"loaded_rows", "viewer"}
        if repo_kind == "dataset"
        else {"model_class", "tokenizer_class", "parameter_count"}
    )
    if not isinstance(payload, dict) or set(payload) != common | kind_fields:
        raise ValueError(f"HF private validation report fields are invalid for {repo_kind}")
    if (payload["repo_kind"], payload["repo_id"], payload["commit_sha"]) != (
        repo_kind,
        repo_id,
        commit_sha,
    ):
        raise ValueError("HF private validation report repo/commit identity mismatch")
    if not SHA40.fullmatch(commit_sha) or payload["private_head_verified_before_and_after"] is not True:
        raise ValueError("HF private validation report lacks private immutable HEAD evidence")
    checksums = payload["checksums"]
    if not isinstance(checksums, dict) or not checksums or payload["file_count"] != len(checksums):
        raise ValueError("HF private validation report file count/checksums are invalid")
    for name, digest in checksums.items():
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts or not isinstance(digest, str):
            raise ValueError("HF private validation report contains an unsafe checksum entry")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("HF private validation report contains a non-SHA-256 checksum")
    if "README.md" not in checksums or payload["tree_sha256"] != sha256_value(checksums):
        raise ValueError("HF private validation report tree digest or card evidence is invalid")
    if repo_kind == "dataset":
        if not isinstance(payload["loaded_rows"], int) or payload["loaded_rows"] <= 0:
            raise ValueError("HF dataset validation report has no loaded rows")
        if not isinstance(payload["viewer"], dict) or payload["viewer"].get("viewer") is not True:
            raise ValueError("HF dataset validation report lacks a successful viewer check")
    elif (
        not isinstance(payload["model_class"], str)
        or not payload["model_class"]
        or not isinstance(payload["tokenizer_class"], str)
        or not payload["tokenizer_class"]
        or not isinstance(payload["parameter_count"], int)
        or payload["parameter_count"] <= 0
    ):
        raise ValueError("HF model validation report lacks a successful full model load")
    return payload


def make_public(
    *,
    repo_kind: Literal["dataset", "model"],
    repo_id: str,
    validated_commit: str,
    api: Any | None = None,
) -> None:
    assert_cpu_publication_context()
    if not SHA40.fullmatch(validated_commit):
        raise ValueError("public gate requires the exact validated private commit SHA")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    current = api.repo_info(repo_id=repo_id, repo_type=repo_kind)
    if getattr(current, "sha", None) != validated_commit:
        raise RuntimeError("repository head changed after private validation")
    if getattr(current, "private", None) is not True:
        raise RuntimeError("repository must still be private at the public release gate")
    api.update_repo_settings(repo_id=repo_id, repo_type=repo_kind, private=False)
    released = api.repo_info(repo_id=repo_id, repo_type=repo_kind)
    if getattr(released, "sha", None) != validated_commit or getattr(released, "private", None) is not False:
        raise RuntimeError("repository visibility or HEAD verification failed after public release")

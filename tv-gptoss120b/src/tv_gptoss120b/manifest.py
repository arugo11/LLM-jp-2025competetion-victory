from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .hashing import canonical_json, sha256_directory, sha256_file, sha256_value


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    kind: str
    uri: str
    digest: str


class StageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    experiment_id: str
    stage: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    config_sha256: str
    git_sha: str
    command: list[str]
    environment: dict[str, str]
    inputs: list[ArtifactRef]
    outputs: list[ArtifactRef]
    metrics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, str | None] = Field(default_factory=dict)

    def digest(self) -> str:
        return sha256_value(self.model_dump(mode="json", exclude={"created_at"}))


def current_git_sha(repo_root: Path) -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        text=True,
    ).strip()
    if status:
        changed = status.splitlines()
        raise RuntimeError(
            "stage manifests require a clean git worktree; commit or remove these changes first: "
            f"{changed[:10]}"
        )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def environment_snapshot() -> dict[str, str]:
    keep = ["PBS_JOBID", "WANDB_MODE", "CUDA_VISIBLE_DEVICES"]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **{key: os.environ[key] for key in keep if key in os.environ},
    }


def immutable_provenance() -> dict[str, str | None]:
    return {
        "container_digest": os.environ.get("CONTAINER_DIGEST"),
        "pbs_script_sha256": os.environ.get("PBS_SCRIPT_SHA256"),
        "wandb_artifact_version": os.environ.get("WANDB_ARTIFACT_VERSION"),
        "hf_commit_sha": os.environ.get("HF_COMMIT_SHA"),
        "resource_usage_record": os.environ.get("RESOURCE_USAGE_RECORD"),
    }


def file_ref(name: str, kind: str, path: Path) -> ArtifactRef:
    return ArtifactRef(name=name, kind=kind, uri=str(path.resolve()), digest=sha256_file(path))


def path_ref(name: str, path: Path, *, role: str) -> ArtifactRef:
    if path.is_file():
        return file_ref(name, f"{role}-file", path)
    if path.is_dir():
        return ArtifactRef(
            name=name,
            kind=f"{role}-directory",
            uri=str(path.resolve()),
            digest=sha256_directory(path),
        )
    raise ValueError(f"manifest path does not exist or is not a regular file/directory: {path}")


def write_immutable_manifest(path: Path, manifest: StageManifest) -> None:
    payload = manifest.model_dump(mode="json")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if sha256_value({k: v for k, v in existing.items() if k != "created_at"}) != manifest.digest():
            raise FileExistsError(f"immutable manifest already exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload) + b"\n")

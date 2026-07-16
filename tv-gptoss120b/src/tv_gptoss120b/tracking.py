from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from .config import ExperimentConfig
from .hashing import sha256_directory, sha256_file, sha256_value

ArtifactKind = Literal["code", "dataset", "model", "evaluation", "publication"]
REQUIRED_ARTIFACTS: dict[str, ArtifactKind] = {
    "code": "code",
    "generation-spec": "dataset",
    "raw-generation-20b": "dataset",
    "raw-generation-120b": "dataset",
    "validated-paired-dataset": "dataset",
    "thinking-sft-preprocessed": "dataset",
    "thinking-grpo-preprocessed": "dataset",
    "instruct-sft-preprocessed": "dataset",
    "instruct-grpo-preprocessed": "dataset",
    "thinking-sft": "model",
    "instruct-sft": "model",
    "thinking-grpo-adapter": "model",
    "instruct-grpo-adapter": "model",
    "thinking-merged": "model",
    "instruct-merged": "model",
    "difficulty-evaluation": "evaluation",
    "aime-evaluation": "evaluation",
    "publication-manifest": "publication",
}
REQUIRED_EDGES: dict[str, set[str]] = {
    "generation-spec": {"code"},
    "raw-generation-20b": {"code", "generation-spec"},
    "raw-generation-120b": {"code", "generation-spec"},
    "validated-paired-dataset": {
        "code",
        "generation-spec",
        "raw-generation-20b",
        "raw-generation-120b",
    },
    "difficulty-evaluation": {"code", "validated-paired-dataset"},
    "thinking-sft-preprocessed": {"code", "validated-paired-dataset"},
    "thinking-grpo-preprocessed": {"code", "validated-paired-dataset"},
    "instruct-sft-preprocessed": {"code", "validated-paired-dataset"},
    "instruct-grpo-preprocessed": {"code", "validated-paired-dataset"},
    "thinking-sft": {"code", "thinking-sft-preprocessed"},
    "instruct-sft": {"code", "instruct-sft-preprocessed"},
    "thinking-grpo-adapter": {"code", "thinking-grpo-preprocessed", "thinking-sft"},
    "instruct-grpo-adapter": {"code", "instruct-grpo-preprocessed", "instruct-sft"},
    "thinking-merged": {"code", "thinking-sft", "thinking-grpo-adapter"},
    "instruct-merged": {"code", "instruct-sft", "instruct-grpo-adapter"},
    "aime-evaluation": {"code", "thinking-merged", "instruct-merged"},
    "publication-manifest": {
        "code",
        "validated-paired-dataset",
        "thinking-merged",
        "instruct-merged",
        "difficulty-evaluation",
        "aime-evaluation",
    },
}
IMMUTABLE_ARTIFACT_REF = re.compile(r"^[^:]+:v[0-9]+$")


def require_immutable_artifact_ref(ref: str) -> None:
    if not IMMUTABLE_ARTIFACT_REF.fullmatch(ref):
        raise ValueError(f"W&B input must use an immutable numeric version such as :v0: {ref}")


class ArtifactRun:
    """Small fail-closed wrapper around W&B artifact lineage."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        stage: str,
        mode: Literal["online", "offline"],
        run_config: dict[str, Any],
        journal_dir: Path | None = None,
    ) -> None:
        config.require_assigned_id()
        if mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
        import wandb

        self._wandb = wandb
        self.mode = mode
        self._input_refs: list[str] = []
        self.run = wandb.init(
            entity=config.tracking.wandb_entity,
            project=config.render(config.tracking.wandb_project),
            job_type=stage,
            config=run_config,
            mode=mode,
        )
        if self.run is None:
            raise RuntimeError("wandb.init returned no run")
        self.journal_dir = journal_dir or Path(os.environ.get("WANDB_DIR", ".")) / "wandb-offline-lineage"

    def use(self, ref: str) -> Any:
        require_immutable_artifact_ref(ref)
        self._input_refs.append(ref)
        if self.mode == "offline":
            return ref
        return self.run.use_artifact(ref)

    def use_logical(self, name: str) -> str:
        if self.mode != "offline" or name not in REQUIRED_ARTIFACTS:
            raise ValueError("logical artifact inputs are allowed only in offline mode for a required artifact name")
        ref = f"artifact://{name}"
        self._input_refs.append(ref)
        return ref

    def _record_offline_output(
        self,
        *,
        name: str,
        kind: ArtifactKind,
        paths: list[Path],
        directory: bool,
        metadata: dict[str, Any],
    ) -> None:
        if self.mode != "offline":
            return
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "schema_version": 1,
            "stage": self.run.job_type,
            "offline_run_id": self.run.id,
            "artifact_name": name,
            "artifact_kind": kind,
            "input_artifacts": self._input_refs,
            "paths": [str(path.resolve()) for path in paths],
            "directory": directory,
            "metadata": metadata,
            "content_sha256": (
                sha256_directory(paths[0])
                if directory
                else {str(path.resolve()): sha256_file(path) for path in paths}
            ),
        }
        path = self.journal_dir / f"{self.run.id}-{name}.json"
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, indent=2) + "\n")

    def log_files(
        self,
        *,
        name: str,
        kind: ArtifactKind,
        files: Iterable[Path],
        metadata: dict[str, Any],
    ) -> Any:
        paths = list(files)
        logged_name = f"offline-{name}" if self.mode == "offline" else name
        artifact = self._wandb.Artifact(name=logged_name, type=kind, metadata=metadata)
        count = 0
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            artifact.add_file(str(path), name=path.name)
            count += 1
        if count == 0:
            raise ValueError("artifact must contain at least one real file")
        result = self.run.log_artifact(artifact)
        self._record_offline_output(
            name=name,
            kind=kind,
            paths=paths,
            directory=False,
            metadata=metadata,
        )
        return result

    def log_directory(
        self,
        *,
        name: str,
        kind: ArtifactKind,
        directory: Path,
        metadata: dict[str, Any],
    ) -> Any:
        if not directory.is_dir() or not any(directory.rglob("*")):
            raise ValueError(f"artifact directory is empty: {directory}")
        logged_name = f"offline-{name}" if self.mode == "offline" else name
        artifact = self._wandb.Artifact(name=logged_name, type=kind, metadata=metadata)
        artifact.add_dir(str(directory))
        result = self.run.log_artifact(artifact)
        self._record_offline_output(
            name=name,
            kind=kind,
            paths=[directory],
            directory=True,
            metadata=metadata,
        )
        return result

    def log_hf_reference(
        self,
        *,
        name: str,
        repo_id: str,
        commit_sha: str,
        metadata: dict[str, Any],
    ) -> Any:
        if len(commit_sha) != 40 or not all(c in "0123456789abcdef" for c in commit_sha):
            raise ValueError("Hugging Face reference must use a 40-character commit SHA")
        artifact = self._wandb.Artifact(name=name, type="model", metadata={**metadata, "hf_commit": commit_sha})
        artifact.add_reference(f"https://huggingface.co/{repo_id}/tree/{commit_sha}")
        return self.run.log_artifact(artifact)

    def finish(self) -> None:
        self.run.finish()

    def __enter__(self) -> ArtifactRun:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.finish()


def verify_artifact_refs(api: Any, refs: Iterable[str]) -> dict[str, str]:
    """Resolve only caller-supplied immutable versions; never discover or substitute latest."""
    resolved: dict[str, str] = {}
    for ref in refs:
        require_immutable_artifact_ref(ref)
        artifact = api.artifact(ref)
        qualified = str(artifact.qualified_name)
        if not qualified.endswith(ref):
            raise RuntimeError(f"W&B artifact resolution mismatch: requested={ref}, resolved={qualified}")
        resolved[ref] = str(artifact.digest)
    return resolved


def verify_artifact_dag(api: Any, refs_by_name: dict[str, str]) -> dict[str, Any]:
    missing = set(REQUIRED_ARTIFACTS) - refs_by_name.keys()
    extra = refs_by_name.keys() - set(REQUIRED_ARTIFACTS)
    if missing or extra:
        raise ValueError(f"artifact manifest mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    artifacts = {}
    for name, ref in refs_by_name.items():
        require_immutable_artifact_ref(ref)
        artifact = api.artifact(ref)
        if artifact.type != REQUIRED_ARTIFACTS[name]:
            raise RuntimeError(f"artifact type mismatch for {name}: {artifact.type}")
        artifacts[name] = artifact
    evidence: dict[str, Any] = {}

    def collection_name(artifact: Any) -> str:
        return str(artifact.name).split("/")[-1].split(":")[0]

    def version_chain(artifact: Any, seen: set[str] | None = None) -> list[Any]:
        visited = seen or set()
        if artifact.digest in visited:
            return []
        visited.add(artifact.digest)
        chain = [artifact]
        run = artifact.logged_by()
        if run is None:
            return chain
        for used in run.used_artifacts():
            if collection_name(used) == collection_name(artifact):
                chain.extend(version_chain(used, visited))
        return chain

    for output_name, required_inputs in REQUIRED_EDGES.items():
        chain = version_chain(artifacts[output_name])
        if not chain or chain[0].logged_by() is None:
            raise RuntimeError(f"artifact has no logging run: {output_name}")
        used_digests = {
            used.digest
            for version in chain
            if version.logged_by() is not None
            for used in version.logged_by().used_artifacts()
        }
        absent = {
            input_name
            for input_name in required_inputs
            if not ({version.digest for version in version_chain(artifacts[input_name])} & used_digests)
        }
        if absent:
            raise RuntimeError(f"artifact DAG edge missing for {output_name}: {sorted(absent)}")
        evidence[output_name] = {
            "logging_run": chain[0].logged_by().path,
            "required_inputs": sorted(required_inputs),
            "output_digest": artifacts[output_name].digest,
        }
    return {"status": "PASS", "artifacts": evidence}


def verify_publication_manifest_binding(api: Any, lineage: dict[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "experiment_id",
        "artifact_refs",
        "hf_commits",
        "hf_validation_report_sha256",
        "release_audit_sha256",
        "publication_candidate_sha256",
    }
    if not isinstance(lineage, dict) or set(lineage) != expected_fields:
        raise ValueError("final publication manifest schema is invalid")
    refs = lineage["artifact_refs"]
    if not isinstance(refs, dict) or set(refs) != set(REQUIRED_ARTIFACTS):
        raise ValueError("final publication manifest does not contain the exact Artifact set")
    publication_ref = str(refs["publication-manifest"])
    require_immutable_artifact_ref(publication_ref)
    candidate = {
        "schema_version": lineage["schema_version"],
        "experiment_id": lineage["experiment_id"],
        "artifact_refs": {
            name: ref for name, ref in refs.items() if name != "publication-manifest"
        },
        "hf_commits": lineage["hf_commits"],
        "hf_validation_report_sha256": lineage["hf_validation_report_sha256"],
        "release_audit_sha256": lineage["release_audit_sha256"],
    }
    expected_digest = sha256_value(candidate)
    if lineage["publication_candidate_sha256"] != expected_digest:
        raise RuntimeError("local publication manifest changed after candidate Artifact creation")
    artifact = api.artifact(publication_ref)
    metadata = getattr(artifact, "metadata", None)
    if (
        not isinstance(metadata, dict)
        or metadata.get("publication_candidate_sha256") != expected_digest
        or metadata.get("release_audit_sha256") != lineage["release_audit_sha256"]
    ):
        raise RuntimeError("W&B publication Artifact metadata does not bind the candidate manifest")
    with tempfile.TemporaryDirectory(prefix="tv-publication-manifest-") as directory:
        downloaded = Path(artifact.download(root=directory))
        files = {path.name: path for path in downloaded.rglob("*") if path.is_file()}
        candidate_name = metadata.get("publication_candidate_filename")
        audit_name = metadata.get("release_audit_filename")
        if (
            not isinstance(candidate_name, str)
            or not isinstance(audit_name, str)
            or set(files) != {candidate_name, audit_name}
        ):
            raise RuntimeError("W&B publication Artifact must contain exactly the candidate and release audit")
        remote_candidate = json.loads(files[candidate_name].read_text(encoding="utf-8"))
        if sha256_file(files[audit_name]) != lineage["release_audit_sha256"]:
            raise RuntimeError("W&B publication Artifact release audit digest mismatch")
    if remote_candidate != candidate or sha256_value(remote_candidate) != expected_digest:
        raise RuntimeError("W&B publication Artifact candidate content does not match the final manifest")
    return {
        "status": "PASS",
        "publication_artifact": publication_ref,
        "publication_candidate_sha256": expected_digest,
    }


def track_stage_directory(
    config: ExperimentConfig,
    *,
    stage: str,
    artifact_name: str,
    artifact_kind: ArtifactKind,
    output_dir: Path,
    input_artifacts: Iterable[str],
    mode: Literal["online", "offline"],
    metadata: dict[str, Any],
) -> None:
    with ArtifactRun(config, stage=stage, mode=mode, run_config=metadata) as tracked:
        for ref in input_artifacts:
            tracked.use(ref)
        tracked.log_directory(name=artifact_name, kind=artifact_kind, directory=output_dir, metadata=metadata)


def replay_offline_lineage(config: ExperimentConfig, journal_paths: Iterable[Path]) -> dict[str, str]:
    """Run on a connected CPU node after `wandb sync`; creates canonical artifacts with real DAG edges."""
    replayed: dict[str, str] = {}
    paths = list(journal_paths)
    verify_offline_journal_dag(paths, require_complete=False)
    pending = {path: json.loads(path.read_text(encoding="utf-8")) for path in paths}
    while pending:
        ready_path = next((
            path
            for path, entry in pending.items()
            if all(
                not str(ref).startswith("artifact://") or str(ref).removeprefix("artifact://") in replayed
                for ref in entry["input_artifacts"]
            )
        ), None)
        if ready_path is None:
            unresolved = {path.name: entry["input_artifacts"] for path, entry in pending.items()}
            raise RuntimeError(f"offline lineage journal has unresolved or cyclic inputs: {unresolved}")
        journal_path = ready_path
        entry = pending.pop(ready_path)
        entry = json.loads(journal_path.read_text(encoding="utf-8"))
        _verify_offline_entry_content(entry)
        journal_sha256 = sha256_file(journal_path)
        with ArtifactRun(
            config,
            stage=f"lineage-replay-{entry['stage']}",
            mode="online",
            run_config={
                "offline_lineage_journal": str(journal_path.resolve()),
                "offline_lineage_journal_sha256": journal_sha256,
            },
        ) as tracked:
            for ref in entry["input_artifacts"]:
                exact_ref = (
                    replayed[str(ref).removeprefix("artifact://")]
                    if str(ref).startswith("artifact://")
                    else ref
                )
                tracked.use(exact_ref)
            paths = [Path(path) for path in entry["paths"]]
            if entry["directory"]:
                if len(paths) != 1:
                    raise ValueError("directory lineage entry must contain exactly one path")
                result = tracked.log_directory(
                    name=entry["artifact_name"],
                    kind=entry["artifact_kind"],
                    directory=paths[0],
                    metadata={**entry["metadata"], "offline_lineage_journal_sha256": journal_sha256},
                )
            else:
                result = tracked.log_files(
                    name=entry["artifact_name"],
                    kind=entry["artifact_kind"],
                    files=paths,
                    metadata={**entry["metadata"], "offline_lineage_journal_sha256": journal_sha256},
                )
            result.wait()
            replayed[entry["artifact_name"]] = result.qualified_name
    return replayed


def verify_offline_journal_dag(
    journal_paths: Iterable[Path], *, require_complete: bool = True
) -> dict[str, Any]:
    entries = [json.loads(path.read_text(encoding="utf-8")) for path in journal_paths]
    by_name = {entry["artifact_name"]: entry for entry in entries}
    if len(by_name) != len(entries):
        raise ValueError("offline lineage journal contains duplicate artifact names")
    extra = by_name.keys() - set(REQUIRED_ARTIFACTS)
    if extra:
        raise ValueError(f"offline lineage journal contains unknown artifacts: {sorted(extra)}")
    missing = set(REQUIRED_ARTIFACTS) - by_name.keys()
    if require_complete and missing:
        raise ValueError(f"offline lineage journal is incomplete: {sorted(missing)}")
    for output_name, required_inputs in REQUIRED_EDGES.items():
        if output_name not in by_name:
            continue
        input_names = {
            (
                str(ref).removeprefix("artifact://")
                if str(ref).startswith("artifact://")
                else str(ref).split("/")[-1].split(":")[0]
            )
            for ref in by_name[output_name]["input_artifacts"]
        }
        absent = {name for name in required_inputs if name not in input_names and name in by_name}
        if absent:
            raise RuntimeError(f"offline lineage edge missing for {output_name}: {sorted(absent)}")
        for path in by_name[output_name]["paths"]:
            if not Path(path).exists():
                raise FileNotFoundError(f"offline lineage output was removed before replay: {path}")
        _verify_offline_entry_content(by_name[output_name])
    return {"status": "PASS", "artifact_count": len(by_name)}


def _verify_offline_entry_content(entry: dict[str, Any]) -> None:
    paths = [Path(path) for path in entry["paths"]]
    expected = entry.get("content_sha256")
    if entry["directory"]:
        if len(paths) != 1 or not paths[0].is_dir():
            raise FileNotFoundError("offline lineage directory is missing")
        actual: str | dict[str, str] = sha256_directory(paths[0])
    else:
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError("offline lineage file is missing")
        actual = {str(path.resolve()): sha256_file(path) for path in paths}
    if expected != actual:
        raise RuntimeError(f"offline lineage content changed before replay: {entry['artifact_name']}")

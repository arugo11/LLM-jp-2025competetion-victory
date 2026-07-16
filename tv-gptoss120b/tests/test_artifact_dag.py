import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from tv_gptoss120b.hashing import sha256_file, sha256_value
from tv_gptoss120b.tracking import (
    REQUIRED_ARTIFACTS,
    REQUIRED_EDGES,
    verify_artifact_dag,
    verify_publication_manifest_binding,
)


@dataclass
class FakeArtifact:
    name: str
    type: str
    digest: str
    inputs: set[str]

    def logged_by(self):
        return FakeRun(self.inputs)


class FakeRun:
    path = ["argo-lab", "project", "run"]

    def __init__(self, inputs: set[str]) -> None:
        self.inputs = inputs

    def used_artifacts(self):
        return [FakeArtifact(name, REQUIRED_ARTIFACTS[name], f"digest-{name}", set()) for name in self.inputs]


class FakeApi:
    def __init__(self, omit_edge: tuple[str, str] | None = None) -> None:
        self.omit_edge = omit_edge

    def artifact(self, ref: str):
        name = ref.split("/")[-1].split(":")[0]
        inputs = set(REQUIRED_EDGES.get(name, set()))
        if self.omit_edge and self.omit_edge[0] == name:
            inputs.discard(self.omit_edge[1])
        return FakeArtifact(name, REQUIRED_ARTIFACTS[name], f"digest-{name}", inputs)


def refs() -> dict[str, str]:
    return {name: f"argo-lab/project/{name}:v0" for name in REQUIRED_ARTIFACTS}


def test_complete_artifact_dag_passes() -> None:
    assert verify_artifact_dag(FakeApi(), refs())["status"] == "PASS"


def test_preprocessing_artifacts_are_four_explicit_arm_kind_nodes() -> None:
    expected = {
        f"{arm}-{kind}-preprocessed"
        for arm in ("thinking", "instruct")
        for kind in ("sft", "grpo")
    }
    assert {name for name in REQUIRED_ARTIFACTS if name.endswith("-preprocessed")} == expected
    for name in expected:
        assert REQUIRED_ARTIFACTS[name] == "dataset"
        assert REQUIRED_EDGES[name] == {"code", "validated-paired-dataset"}
    for arm in ("thinking", "instruct"):
        assert REQUIRED_EDGES[f"{arm}-sft"] == {"code", f"{arm}-sft-preprocessed"}
        assert REQUIRED_EDGES[f"{arm}-grpo-adapter"] == {
            "code",
            f"{arm}-grpo-preprocessed",
            f"{arm}-sft",
        }


def test_missing_artifact_edge_fails() -> None:
    with pytest.raises(RuntimeError, match="edge missing"):
        verify_artifact_dag(FakeApi(("thinking-merged", "thinking-grpo-adapter")), refs())


class PublicationArtifact:
    def __init__(self, candidate: dict, source: Path, audit: Path) -> None:
        self.metadata = {
            "publication_candidate_sha256": sha256_value(candidate),
            "publication_candidate_filename": source.name,
            "release_audit_sha256": candidate["release_audit_sha256"],
            "release_audit_filename": audit.name,
        }
        self.sources = [source, audit]

    def download(self, root: str) -> str:
        destination = Path(root) / "artifact"
        destination.mkdir()
        for source in self.sources:
            shutil.copyfile(source, destination / source.name)
        return str(destination)


class PublicationApi:
    def __init__(self, artifact: PublicationArtifact) -> None:
        self._artifact = artifact

    def artifact(self, ref: str):
        return self._artifact


def test_publication_manifest_is_bound_to_remote_artifact_content(tmp_path: Path) -> None:
    all_refs = refs()
    candidate = {
        "schema_version": 1,
        "experiment_id": "9999",
        "artifact_refs": {
            name: ref for name, ref in all_refs.items() if name != "publication-manifest"
        },
        "hf_commits": {"argo11/dataset": "a" * 40},
        "hf_validation_report_sha256": {"argo11/dataset": "b" * 64},
        "release_audit_sha256": "d" * 64,
    }
    source = tmp_path / "candidate.json"
    source.write_text(json.dumps(candidate), encoding="utf-8")
    lineage = {
        **candidate,
        "artifact_refs": all_refs,
        "publication_candidate_sha256": sha256_value(candidate),
    }
    audit = tmp_path / "release-audit.json"
    audit.write_bytes(b"release-audit")
    candidate["release_audit_sha256"] = sha256_file(audit)
    source.write_text(json.dumps(candidate), encoding="utf-8")
    lineage["release_audit_sha256"] = candidate["release_audit_sha256"]
    lineage["publication_candidate_sha256"] = sha256_value(candidate)
    api = PublicationApi(PublicationArtifact(candidate, source, audit))

    assert verify_publication_manifest_binding(api, lineage)["status"] == "PASS"

    lineage["hf_commits"] = {"argo11/dataset": "c" * 40}
    with pytest.raises(RuntimeError, match="changed after candidate"):
        verify_publication_manifest_binding(api, lineage)

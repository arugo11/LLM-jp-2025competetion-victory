import subprocess
from pathlib import Path

import pytest

from tv_gptoss120b.cli import _stage_manifest
from tv_gptoss120b.config import load_config
from tv_gptoss120b.hashing import sha256_directory, sha256_file
from tv_gptoss120b.manifest import StageManifest, current_git_sha, path_ref, write_immutable_manifest

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def make_manifest(metric: int = 1) -> StageManifest:
    return StageManifest(
        experiment_id="0421",
        stage="fixture",
        config_sha256="a" * 64,
        git_sha="b" * 40,
        command=["fixture"],
        environment={"python": "3.12"},
        inputs=[],
        outputs=[],
        metrics={"value": metric},
    )


def test_manifest_is_idempotent_but_not_mutable(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_immutable_manifest(path, make_manifest())
    write_immutable_manifest(path, make_manifest())
    with pytest.raises(FileExistsError):
        write_immutable_manifest(path, make_manifest(metric=2))


def test_path_ref_hashes_files_and_directories(tmp_path: Path) -> None:
    file_path = tmp_path / "input.json"
    file_path.write_text("{}\n", encoding="utf-8")
    directory = tmp_path / "dataset"
    directory.mkdir()
    (directory / "part.jsonl").write_text("{}\n", encoding="utf-8")

    file_input = path_ref("config", file_path, role="input")
    directory_output = path_ref("dataset", directory, role="output")

    assert file_input.kind == "input-file"
    assert file_input.digest == sha256_file(file_path)
    assert directory_output.kind == "output-directory"
    assert directory_output.digest == sha256_directory(directory)


def test_stage_manifest_records_real_inputs_and_outputs(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "stage"
    output_dir.mkdir()
    output_path = output_dir / "result.json"
    output_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("tv_gptoss120b.cli.current_git_sha", lambda _: "b" * 40)

    _stage_manifest(
        CONFIG,
        load_config(CONFIG),
        "fixture",
        output_dir,
        {"passed": True},
        input_paths=[input_path],
    )

    manifest = StageManifest.model_validate_json(
        (output_dir / "stage-manifest.json").read_text(encoding="utf-8")
    )
    assert [item.name for item in manifest.inputs] == ["config", "input-0:input.jsonl"]
    assert manifest.inputs[1].digest == sha256_file(input_path)
    assert [item.name for item in manifest.outputs] == ["result.json"]
    assert manifest.outputs[0].digest == sha256_file(output_path)
    assert manifest.provenance["runtime_stage"] == "audit"
    assert manifest.provenance["runtime_profile"] == "local_audit"
    assert manifest.provenance["runtime_resource_class"] == "local_cpu"


def test_current_git_sha_rejects_dirty_worktree(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)

    clean_sha = current_git_sha(tmp_path)
    assert len(clean_sha) == 40

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean git worktree"):
        current_git_sha(tmp_path)

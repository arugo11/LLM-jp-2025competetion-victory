from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer

from .aime import summarize_aime, verify_matched_fingerprint
from .cards import render_dataset_card, render_model_card, write_card
from .cluster_gate import verify_qsub_gate
from .config import ExperimentConfig, load_config
from .curation import finalize_curation, prepare_review, verify_generation_inputs
from .evaluation import run_difficulty_inference, run_matched_aime_suite, run_swallow_aime, summarize_difficulty
from .generation import generate_problems, validate_problems
from .hashing import sha256_directory, sha256_file, sha256_value
from .jsonl import read_jsonl, write_jsonl
from .manifest import (
    StageManifest,
    current_git_sha,
    environment_snapshot,
    file_ref,
    immutable_provenance,
    path_ref,
    write_immutable_manifest,
)
from .model_lineage import (
    LINEAGE_FILENAME,
    verify_sft_model_lineage,
    write_merged_model_lineage,
    write_sft_model_lineage,
)
from .pbs import render_pbs
from .preprocessing import prepare_grpo_dataset, prepare_sft_dataset
from .prompts import PROMPT_REVISION
from .publication import make_public, upload_private_then_validate, validate_private_validation_report
from .specs import build_specs
from .tracking import REQUIRED_ARTIFACTS, REQUIRED_EDGES, ArtifactRun, require_immutable_artifact_ref
from .training import merge_adapter, run_grpo, run_sft, verify_merge_parity

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


def _load(path: Path, *, require_id: bool = True) -> ExperimentConfig:
    config = load_config(path)
    if require_id:
        config.require_assigned_id()
        config.require_production_contract()
    return config


def _stage_manifest(
    config_path: Path,
    config: ExperimentConfig,
    stage: str,
    output_dir: Path,
    metrics: dict,
    *,
    input_paths: list[Path] | None = None,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    runtime_stage = config.runtime_stage_for_manifest(stage)
    runtime_profile_name = config.runtime.stage_profile[runtime_stage]
    runtime_profile = config.runtime.profiles[runtime_profile_name]
    inputs = [path_ref("config", config_path, role="input")]
    inputs.extend(
        path_ref(f"input-{index}:{path.name}", path, role="input")
        for index, path in enumerate(input_paths or [])
    )
    output_files = [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name not in {"stage-manifest.json", "artifact-receipt.json"}
    ]
    if not output_files:
        raise ValueError(f"stage produced no files to record: {stage}")
    manifest = StageManifest(
        experiment_id=config.identity.experiment_id,
        stage=stage,
        config_sha256=sha256_file(config_path),
        git_sha=current_git_sha(repo_root),
        command=sys.argv,
        environment=environment_snapshot(),
        inputs=inputs,
        outputs=[
            file_ref(str(path.relative_to(output_dir)), "output-file", path)
            for path in output_files
        ],
        metrics=metrics,
        provenance={
            **immutable_provenance(),
            "prompt_hash": PROMPT_REVISION,
            "runtime_stage": runtime_stage,
            "runtime_profile": runtime_profile_name,
            "runtime_resource_class": runtime_profile.resource_class,
            "runtime_profile_sha256": sha256_value(runtime_profile.model_dump(mode="json")),
            "runtime_declared_environment_sha256": sha256_value(runtime_profile.environment),
        },
    )
    write_immutable_manifest(output_dir / "stage-manifest.json", manifest)


def _load_artifact_refs(
    path: Path | None,
    required_names: set[str],
    mode: Literal["online", "offline"],
) -> dict[str, str]:
    if path is None:
        raise typer.BadParameter(f"--artifact-refs is required with keys {sorted(required_names)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    refs = payload.get("artifact_refs", payload)
    if not isinstance(refs, dict) or set(refs) != required_names:
        raise typer.BadParameter(f"artifact ref keys must be exactly {sorted(required_names)}")
    for name, ref in refs.items():
        if mode == "offline" and ref == f"artifact://{name}":
            continue
        require_immutable_artifact_ref(str(ref))
    return {str(name): str(ref) for name, ref in refs.items()}


def _track_output(
    config: ExperimentConfig,
    *,
    stage: str,
    name: str,
    kind: Literal["code", "dataset", "model", "evaluation", "publication"],
    inputs: dict[str, str],
    mode: Literal["online", "offline"],
    metadata: dict,
    directory: Path | None = None,
    files: list[Path] | None = None,
) -> str:
    if kind != REQUIRED_ARTIFACTS[name]:
        raise ValueError(f"artifact kind mismatch for {name}: {kind}")
    expected_inputs = REQUIRED_EDGES.get(name, set())
    if set(inputs) != expected_inputs:
        raise ValueError(f"artifact inputs for {name} must be exactly {sorted(expected_inputs)}")
    with ArtifactRun(config, stage=stage, mode=mode, run_config=metadata) as tracked:
        for input_name, ref in inputs.items():
            if ref == f"artifact://{input_name}":
                tracked.use_logical(input_name)
            else:
                tracked.use(ref)
        if directory is not None:
            result = tracked.log_directory(name=name, kind=kind, directory=directory, metadata=metadata)
        elif files is not None:
            result = tracked.log_files(name=name, kind=kind, files=files, metadata=metadata)
        else:
            raise ValueError("artifact output requires directory or files")
        if mode == "offline":
            return f"artifact://{name}"
        result.wait()
        return str(result.qualified_name)


def _write_artifact_receipt(output_dir: Path, refs: dict[str, str], mode: str) -> None:
    path = output_dir / "artifact-receipt.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"artifact_refs": refs, "wandb_mode": mode}, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


@app.command()
def generate(
    config_path: ConfigOption,
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    backend: Annotated[Literal["fixture", "vllm"], typer.Option()] = "vllm",
    smoke: Annotated[bool, typer.Option()] = False,
    wandb_mode: Annotated[Literal["online", "offline"], typer.Option()] = "online",
) -> None:
    config = _load(config_path, require_id=backend != "fixture")
    output_dir.mkdir(parents=True, exist_ok=False)
    specs = build_specs(config.data)
    if smoke:
        specs = specs[:8]
    specs_path = output_dir / "generation-spec.jsonl"
    write_jsonl(specs_path, specs)
    for model_key in ("generator_20b", "generator_120b"):
        generate_problems(config, specs_path, output_dir / f"raw-{model_key}.jsonl", model_key, backend)
    _stage_manifest(
        config_path,
        config,
        (
            f"fixture-generate{'-smoke' if smoke else ''}"
            if backend == "fixture"
            else "generate-smoke" if smoke else "generate"
        ),
        output_dir,
        {"spec_count": len(specs), "backend": backend, "smoke": smoke},
        input_paths=[],
    )
    if backend != "fixture" and not smoke:
        repo_root = Path(__file__).resolve().parents[3]
        code_archive = output_dir / "code-snapshot.tar.gz"
        subprocess.run(
            ["git", "archive", "--format=tar.gz", f"--output={code_archive}", "HEAD", "tv-gptoss120b"],
            cwd=repo_root,
            check=True,
        )
        metadata = {
            "stage": "generate-smoke" if smoke else "generate",
            "config_sha256": sha256_file(config_path),
            "git_sha": current_git_sha(repo_root),
            "upstream_revision": config.identity.upstream_revision,
            "container_digest": immutable_provenance()["container_digest"],
        }
        refs = {}
        refs["code"] = _track_output(
            config,
            stage="code",
            name="code",
            kind="code",
            inputs={},
            mode=wandb_mode,
            metadata=metadata,
            files=[code_archive, config_path],
        )
        refs["generation-spec"] = _track_output(
            config,
            stage="generation-spec",
            name="generation-spec",
            kind="dataset",
            inputs={"code": refs["code"]},
            mode=wandb_mode,
            metadata=metadata,
            files=[specs_path],
        )
        for model_key, artifact_name in (
            ("generator_20b", "raw-generation-20b"),
            ("generator_120b", "raw-generation-120b"),
        ):
            refs[artifact_name] = _track_output(
                config,
                stage=artifact_name,
                name=artifact_name,
                kind="dataset",
                inputs={"code": refs["code"], "generation-spec": refs["generation-spec"]},
                mode=wandb_mode,
                metadata=metadata,
                files=[output_dir / f"raw-{model_key}.jsonl"],
            )
        _write_artifact_receipt(output_dir, refs, wandb_mode)


@app.command()
def curate(
    config_path: ConfigOption,
    generation_spec: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    raw_20b: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    raw_120b: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    aime_reference: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    decisions: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    backend: Annotated[Literal["fixture", "vllm"], typer.Option()] = "vllm",
    artifact_refs: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    wandb_mode: Annotated[Literal["online", "offline"], typer.Option()] = "online",
) -> None:
    config = _load(config_path, require_id=backend != "fixture")
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_paths = [raw_20b, raw_120b]
    verify_generation_inputs(config, generation_spec, raw_paths)
    votes = []
    for validator in ("generator_20b", "generator_120b"):
        path = output_dir / f"votes-{validator}.jsonl"
        validate_problems(config, raw_paths, path, validator, backend)
        votes.append(path)
    review = output_dir / "review-queue.jsonl"
    review_metrics = prepare_review(config, generation_spec, raw_paths, votes, aime_reference, review)
    metrics: dict = {**review_metrics, "quality_gate": "awaiting_manual_decisions"}
    if decisions is not None:
        metrics.update(
            finalize_curation(
                config,
                generation_spec,
                raw_paths,
                votes,
                review,
                decisions,
                output_dir / "paired.jsonl",
            )
        )
        shutil.copyfile(decisions, output_dir / "manual-decisions.jsonl")
        metrics["quality_gate"] = "passed"
    _stage_manifest(
        config_path,
        config,
        "fixture-curate" if backend == "fixture" else "curate",
        output_dir,
        metrics,
        input_paths=[
            generation_spec,
            raw_20b,
            raw_120b,
            aime_reference,
            *([decisions] if decisions is not None else []),
        ],
    )
    if backend != "fixture" and decisions is not None:
        required = REQUIRED_EDGES["validated-paired-dataset"]
        inputs = _load_artifact_refs(artifact_refs, required, wandb_mode)
        ref = _track_output(
            config,
            stage="curate",
            name="validated-paired-dataset",
            kind="dataset",
            inputs=inputs,
            mode=wandb_mode,
            metadata=metrics,
            directory=output_dir,
        )
        _write_artifact_receipt(
            output_dir,
            {"code": inputs["code"], "validated-paired-dataset": ref},
            wandb_mode,
        )


@app.command()
def difficulty(
    config_path: ConfigOption,
    curated: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    artifact_refs: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    wandb_mode: Annotated[Literal["online", "offline"], typer.Option()] = "online",
) -> None:
    config = _load(config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = []
    for arm in ("thinking", "instruct"):
        path = output_dir / f"{arm}.jsonl"
        run_difficulty_inference(config, arm, curated, path)
        paths.append(path)
    summary = summarize_difficulty(config, paths, output_dir / "summary.json")
    _stage_manifest(
        config_path,
        config,
        "difficulty",
        output_dir,
        summary,
        input_paths=[curated, artifact_refs],
    )
    inputs = _load_artifact_refs(artifact_refs, REQUIRED_EDGES["difficulty-evaluation"], wandb_mode)
    ref = _track_output(
        config,
        stage="difficulty",
        name="difficulty-evaluation",
        kind="evaluation",
        inputs=inputs,
        mode=wandb_mode,
        metadata=summary,
        directory=output_dir,
    )
    _write_artifact_receipt(
        output_dir,
        {**inputs, "difficulty-evaluation": ref},
        wandb_mode,
    )


@app.command()
def preprocess(
    config_path: ConfigOption,
    curated: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    smoke: Annotated[bool, typer.Option()] = False,
    artifact_refs: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    wandb_mode: Annotated[Literal["online", "offline"], typer.Option()] = "online",
) -> None:
    config = _load(config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    workers = config.runtime.profiles["cpu_pipeline"].cpu_workers
    manifests = {}
    artifact_dirs: dict[str, Path] = {}
    for arm in ("thinking", "instruct"):
        sft_name = f"{arm}-sft-preprocessed"
        sft_dir = output_dir / sft_name
        sft_manifest = prepare_sft_dataset(
            config,
            arm,
            curated,
            sft_dir,
            workers,
            smoke=smoke,
        )
        manifests[sft_name] = sft_manifest.model_dump(mode="json")
        artifact_dirs[sft_name] = sft_dir

        grpo_name = f"{arm}-grpo-preprocessed"
        grpo_dir = output_dir / grpo_name
        grpo_manifest = prepare_grpo_dataset(
            config,
            arm,
            curated,
            grpo_dir,
            smoke=smoke,
        )
        manifests[grpo_name] = grpo_manifest.model_dump(mode="json")
        artifact_dirs[grpo_name] = grpo_dir

    metrics = {
        "artifact_count": len(manifests),
        "artifacts": manifests,
        "smoke": smoke,
    }
    _stage_manifest(
        config_path,
        config,
        "preprocess-smoke" if smoke else "preprocess",
        output_dir,
        metrics,
        input_paths=[curated, *([artifact_refs] if artifact_refs is not None else [])],
    )
    if not smoke:
        first_name = "thinking-sft-preprocessed"
        inputs = _load_artifact_refs(artifact_refs, REQUIRED_EDGES[first_name], wandb_mode)
        refs = dict(inputs)
        for name, directory in artifact_dirs.items():
            refs[name] = _track_output(
                config,
                stage=name,
                name=name,
                kind="dataset",
                inputs=inputs,
                mode=wandb_mode,
                metadata=manifests[name],
                directory=directory,
            )
        _write_artifact_receipt(output_dir, refs, wandb_mode)


@app.command()
def sft(
    config_path: ConfigOption,
    arm: Annotated[Literal["thinking", "instruct"], typer.Option()],
    preprocessed: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    smoke: Annotated[bool, typer.Option()] = False,
    artifact_refs: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    wandb_mode: Annotated[Literal["online", "offline"], typer.Option()] = "online",
) -> None:
    config = _load(config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics = run_sft(config, arm, preprocessed, output_dir / "training", smoke=smoke)
    write_sft_model_lineage(
        config,
        arm,
        output_dir / "training" / "final",
        preprocessed,
        output_dir / "training" / "training-metrics.json",
    )
    _stage_manifest(
        config_path,
        config,
        f"{arm}-sft",
        output_dir,
        metrics,
        input_paths=[preprocessed, *([artifact_refs] if artifact_refs is not None else [])],
    )
    if not smoke:
        artifact_name = f"{arm}-sft"
        inputs = _load_artifact_refs(artifact_refs, REQUIRED_EDGES[artifact_name], wandb_mode)
        ref = _track_output(
            config,
            stage=artifact_name,
            name=artifact_name,
            kind="model",
            inputs=inputs,
            mode=wandb_mode,
            metadata=metrics,
            directory=output_dir / "training" / "final",
        )
        _write_artifact_receipt(output_dir, {**inputs, artifact_name: ref}, wandb_mode)


@app.command()
def grpo(
    config_path: ConfigOption,
    arm: Annotated[Literal["thinking", "instruct"], typer.Option()],
    preprocessed: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    sft_model: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    smoke: Annotated[bool, typer.Option()] = False,
    artifact_refs: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    wandb_mode: Annotated[Literal["online", "offline"], typer.Option()] = "online",
) -> None:
    config = _load(config_path)
    verify_sft_model_lineage(config, arm, sft_model)
    output_dir.mkdir(parents=True, exist_ok=False)
    training_dir = output_dir / "training"
    metrics = run_grpo(config, arm, sft_model, preprocessed, training_dir, smoke=smoke)
    merged = output_dir / "merged"
    merge_adapter(sft_model, training_dir / "adapter", merged)
    parity = verify_merge_parity(
        sft_model,
        training_dir / "adapter",
        merged,
        [f"固定merge検証prompt {index}" for index in range(8)],
        output_dir / "merge-parity.json",
    )
    merged_lineage = write_merged_model_lineage(
        config,
        arm,
        merged,
        sft_model,
        training_dir / "adapter",
        preprocessed,
        output_dir / "merge-parity.json",
    )
    _stage_manifest(
        config_path,
        config,
        f"{arm}-grpo",
        output_dir,
        {**metrics, "merge_parity": parity},
        input_paths=[preprocessed, sft_model, *([artifact_refs] if artifact_refs is not None else [])],
    )
    if not smoke:
        adapter_name = f"{arm}-grpo-adapter"
        merged_name = f"{arm}-merged"
        adapter_inputs = _load_artifact_refs(artifact_refs, REQUIRED_EDGES[adapter_name], wandb_mode)
        adapter_ref = _track_output(
            config,
            stage=adapter_name,
            name=adapter_name,
            kind="model",
            inputs=adapter_inputs,
            mode=wandb_mode,
            metadata=metrics,
            directory=training_dir / "adapter",
        )
        merged_metadata_dir = output_dir / "merged-artifact"
        merged_metadata_dir.mkdir()
        merged_metadata = {
            "local_model_path": str(merged.resolve()),
            "local_model_sha256": sha256_directory(merged),
            "model_lineage_sha256": sha256_file(merged / LINEAGE_FILENAME),
            "model_lineage": merged_lineage.model_dump(mode="json"),
            "merge_parity": parity,
            "hf_commit": None,
        }
        (merged_metadata_dir / "merged-model-reference.json").write_text(
            json.dumps(merged_metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        merged_inputs = {
            "code": adapter_inputs["code"],
            f"{arm}-sft": adapter_inputs[f"{arm}-sft"],
            adapter_name: adapter_ref,
        }
        merged_ref = _track_output(
            config,
            stage=merged_name,
            name=merged_name,
            kind="model",
            inputs=merged_inputs,
            mode=wandb_mode,
            metadata=merged_metadata,
            directory=merged_metadata_dir,
        )
        _write_artifact_receipt(
            output_dir,
            {**adapter_inputs, adapter_name: adapter_ref, merged_name: merged_ref},
            wandb_mode,
        )


@app.command()
def evaluate(
    config_path: ConfigOption,
    harness_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    port: Annotated[int, typer.Option(min=1024, max=65535)],
    models_manifest: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    model_path: Annotated[str | None, typer.Option()] = None,
    label: Annotated[
        Literal["thinking-base", "thinking-post", "instruct-base", "instruct-post"] | None,
        typer.Option(),
    ] = None,
    model_revision: Annotated[str | None, typer.Option()] = None,
    smoke: Annotated[bool, typer.Option()] = False,
    artifact_refs: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    wandb_mode: Annotated[Literal["online", "offline"], typer.Option()] = "online",
) -> None:
    config = _load(config_path)
    if smoke:
        if models_manifest is not None or model_path is None or label is None:
            raise typer.BadParameter("smoke evaluation requires --model-path and --label, without --models-manifest")
        output_dir.mkdir(parents=True, exist_ok=False)
        result = run_swallow_aime(
            config,
            harness_dir,
            model_path,
            model_revision,
            label,
            output_dir / "raw",
            output_dir / "logs",
            port,
            smoke=True,
        )
        stage = f"evaluate-{label}-smoke"
    else:
        if models_manifest is None or model_path is not None or label is not None or model_revision is not None:
            raise typer.BadParameter("production evaluation requires only --models-manifest")
        result = run_matched_aime_suite(config, harness_dir, models_manifest, output_dir, port)
        stage = "evaluate-aime-matched"
    evaluation_inputs = [harness_dir]
    if models_manifest is not None:
        evaluation_inputs.append(models_manifest)
    if artifact_refs is not None:
        evaluation_inputs.append(artifact_refs)
    if model_path is not None and Path(model_path).exists():
        evaluation_inputs.append(Path(model_path))
    _stage_manifest(
        config_path,
        config,
        stage,
        output_dir,
        result,
        input_paths=evaluation_inputs,
    )
    if not smoke:
        inputs = _load_artifact_refs(artifact_refs, REQUIRED_EDGES["aime-evaluation"], wandb_mode)
        ref = _track_output(
            config,
            stage="aime-evaluation",
            name="aime-evaluation",
            kind="evaluation",
            inputs=inputs,
            mode=wandb_mode,
            metadata={
                "matched_config_sha256": result["matched_config_sha256"],
                "normalized_sample_count": result["normalized_sample_count"],
                "summary": result["summary"],
            },
            directory=output_dir,
        )
        _write_artifact_receipt(output_dir, {**inputs, "aime-evaluation": ref}, wandb_mode)


@app.command()
def summarize_evaluation(
    config_path: ConfigOption,
    records: Annotated[list[Path], typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
) -> None:
    config = _load(config_path)
    combined = [row for path in records for row in read_jsonl(path)]
    from .evaluation import matched_evaluation_fingerprint

    verify_matched_fingerprint(combined, matched_evaluation_fingerprint(config))
    summary = summarize_aime(combined)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


@app.command("render-pbs")
def render_pbs_command(
    config_path: ConfigOption,
    job_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    policy_snapshot: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Render an immutable PBS script; this command never calls qsub."""
    config = _load(config_path)
    render_pbs(config, job_manifest, policy_snapshot, output)
    typer.echo(str(output))


@app.command("verify-qsub-local")
def verify_qsub_local_command(
    config_path: ConfigOption,
    approval_record: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    policy_snapshot: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    job_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    pbs_script: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    storage_audit: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    plan_sha256: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Run the package gate; the external common cluster preflight is still required."""
    config = _load(config_path)
    if len(plan_sha256) != 64 or any(char not in "0123456789abcdef" for char in plan_sha256):
        raise typer.BadParameter("--plan-sha256 must be a lowercase SHA-256")
    report = verify_qsub_gate(
        config,
        approval_record=approval_record,
        policy_snapshot=policy_snapshot,
        job_manifest=job_manifest,
        pbs_script=pbs_script,
        storage_audit=storage_audit,
        plan_sha256=plan_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    typer.echo(str(output))


@app.command()
def publish(
    config_path: ConfigOption,
    kind: Annotated[Literal["dataset", "model"] | None, typer.Option()] = None,
    repo_id: Annotated[str | None, typer.Option()] = None,
    source_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    validated_commit: Annotated[str | None, typer.Option()] = None,
    lineage_manifest: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    offline_journal: Annotated[list[Path] | None, typer.Option(exists=True, dir_okay=False)] = None,
    finalize_lineage: Annotated[bool, typer.Option("--finalize-lineage")] = False,
    publication_manifest_output: Annotated[Path | None, typer.Option()] = None,
    prepare_card: Annotated[bool, typer.Option("--prepare-card")] = False,
    card_context: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    make_repo_public: Annotated[bool, typer.Option("--make-public")] = False,
) -> None:
    config = _load(config_path)
    if offline_journal:
        if any(value is not None for value in (kind, repo_id, source_dir, validated_commit, lineage_manifest)):
            raise typer.BadParameter("--offline-journal cannot be combined with repository publication options")
        from .tracking import replay_offline_lineage

        typer.echo(json.dumps(replay_offline_lineage(config, offline_journal), sort_keys=True))
        return
    if finalize_lineage:
        if any(value is not None for value in (kind, repo_id, source_dir, validated_commit, offline_journal)):
            raise typer.BadParameter("--finalize-lineage cannot be combined with upload or visibility options")
        if lineage_manifest is None or publication_manifest_output is None:
            raise typer.BadParameter("--lineage-manifest and --publication-manifest-output are required")
        import wandb

        from .tracking import verify_artifact_dag, verify_publication_manifest_binding

        lineage = json.loads(lineage_manifest.read_text(encoding="utf-8"))
        release_evidence = lineage.get("release_evidence")
        if not isinstance(release_evidence, dict):
            raise typer.BadParameter("lineage manifest must contain release_evidence")
        from .release_audit import audit_release_evidence

        release_audit = audit_release_evidence(config, config_path, release_evidence)
        artifact_refs = lineage.get("artifact_refs")
        hf_commits = lineage.get("hf_commits")
        validation_reports = lineage.get("hf_validation_reports")
        expected_prepublication = set(REQUIRED_ARTIFACTS) - {"publication-manifest"}
        expected_hf_repos = {
            config.render(config.publication.dataset_repo),
            config.render(config.publication.thinking_model_repo),
            config.render(config.publication.instruct_model_repo),
        }
        if not isinstance(artifact_refs, dict) or set(artifact_refs) != expected_prepublication:
            raise typer.BadParameter(f"prepublication artifact refs must be {sorted(expected_prepublication)}")
        if not isinstance(hf_commits, dict) or set(hf_commits) != expected_hf_repos:
            raise typer.BadParameter(f"hf_commits keys must be {sorted(expected_hf_repos)}")
        if not isinstance(validation_reports, dict) or set(validation_reports) != expected_hf_repos:
            raise typer.BadParameter(f"hf_validation_reports keys must be {sorted(expected_hf_repos)}")
        for ref in artifact_refs.values():
            require_immutable_artifact_ref(str(ref))
        for sha in hf_commits.values():
            if not isinstance(sha, str) or len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
                raise typer.BadParameter("every Hugging Face commit must be a 40-character SHA")
        report_checksums = {}
        for repo, report_path in validation_reports.items():
            path = Path(str(report_path))
            if not path.is_file():
                raise typer.BadParameter(f"HF validation report is missing for {repo}: {path}")
            repo_kind = "dataset" if repo == config.render(config.publication.dataset_repo) else "model"
            validate_private_validation_report(
                path,
                repo_kind=repo_kind,
                repo_id=repo,
                commit_sha=hf_commits[repo],
            )
            report_checksums[repo] = sha256_file(path)
        for arm in ("thinking", "instruct"):
            name = f"{arm}-merged"
            repo = config.render(getattr(config.publication, f"{arm}_model_repo"))
            with ArtifactRun(
                config,
                stage=f"{name}-hf-release",
                mode="online",
                run_config={"hf_repo": repo, "hf_commit": hf_commits[repo]},
            ) as tracked:
                tracked.use(artifact_refs[name])
                result = tracked.log_hf_reference(
                    name=name,
                    repo_id=repo,
                    commit_sha=hf_commits[repo],
                    metadata={
                        "prepublication_artifact": artifact_refs[name],
                        "hf_validation_report_sha256": report_checksums[repo],
                    },
                )
                result.wait()
                artifact_refs[name] = str(result.qualified_name)
        release_audit_path = publication_manifest_output.with_suffix(".release-audit.json")
        release_audit_path.parent.mkdir(parents=True, exist_ok=True)
        with release_audit_path.open("x", encoding="utf-8") as handle:
            json.dump(release_audit, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        release_audit_sha256 = sha256_file(release_audit_path)
        candidate_path = publication_manifest_output.with_suffix(".wandb-input.json")
        candidate = {
            "schema_version": 1,
            "experiment_id": config.require_assigned_id(),
            "artifact_refs": artifact_refs,
            "hf_commits": hf_commits,
            "hf_validation_report_sha256": report_checksums,
            "release_audit_sha256": release_audit_sha256,
        }
        candidate_sha256 = sha256_value(candidate)
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        with candidate_path.open("x", encoding="utf-8") as handle:
            json.dump(candidate, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        publication_inputs = {
            name: artifact_refs[name]
            for name in REQUIRED_EDGES["publication-manifest"]
        }
        publication_ref = _track_output(
            config,
            stage="publication-manifest",
            name="publication-manifest",
            kind="publication",
            inputs=publication_inputs,
            mode="online",
            metadata={
                "hf_commits": hf_commits,
                "validation_checksums": report_checksums,
                "publication_candidate_sha256": candidate_sha256,
                "publication_candidate_filename": candidate_path.name,
                "release_audit_sha256": release_audit_sha256,
                "release_audit_filename": release_audit_path.name,
            },
            files=[candidate_path, release_audit_path],
        )
        final_refs = {**artifact_refs, "publication-manifest": publication_ref}
        final = {
            **candidate,
            "artifact_refs": final_refs,
            "publication_candidate_sha256": candidate_sha256,
        }
        with publication_manifest_output.open("x", encoding="utf-8") as handle:
            json.dump(final, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        api = wandb.Api(timeout=60)
        verify_artifact_dag(api, final_refs)
        verify_publication_manifest_binding(api, final)
        typer.echo(str(publication_manifest_output))
        return
    if kind is None or repo_id is None or source_dir is None:
        raise typer.BadParameter("--kind, --repo-id, and --source-dir are required for repository publication")
    allowed_repos = (
        {config.render(config.publication.dataset_repo)}
        if kind == "dataset"
        else {
            config.render(config.publication.thinking_model_repo),
            config.render(config.publication.instruct_model_repo),
        }
    )
    if repo_id not in allowed_repos:
        raise typer.BadParameter(f"repo_id is outside the approved publication set: {sorted(allowed_repos)}")
    card = source_dir / "README.md"
    if prepare_card:
        if make_repo_public or validated_commit is not None or lineage_manifest is not None or card_context is None:
            raise typer.BadParameter("--prepare-card requires only --kind, --repo-id, --source-dir, and --card-context")
        context = json.loads(card_context.read_text(encoding="utf-8"))
        if kind == "dataset":
            content = render_dataset_card(
                config,
                manifest=context["manifest"],
                wandb_artifact=context["wandb_artifact"],
            )
        else:
            arm = context.get("arm")
            if arm not in {"thinking", "instruct"}:
                raise typer.BadParameter("model card context arm must be thinking or instruct")
            expected_repo = config.render(getattr(config.publication, f"{arm}_model_repo"))
            if repo_id != expected_repo:
                raise typer.BadParameter(f"model card arm/repo mismatch: expected {expected_repo}")
            content = render_model_card(
                config,
                arm=arm,
                dataset_commit=context["dataset_commit"],
                aime_summary=context["aime_summary"],
                merge_summary=context["merge_summary"],
                wandb_artifact=context["wandb_artifact"],
            )
        if card.exists():
            raise FileExistsError(f"card output is immutable: {card}")
        write_card(card, content)
        typer.echo(str(card))
        return
    if make_repo_public:
        if validated_commit is None or lineage_manifest is None:
            raise typer.BadParameter("--validated-commit and --lineage-manifest are required with --make-public")
        import wandb

        from .tracking import verify_artifact_dag, verify_publication_manifest_binding

        lineage = json.loads(lineage_manifest.read_text(encoding="utf-8"))
        artifact_refs = lineage.get("artifact_refs")
        if not isinstance(artifact_refs, dict):
            raise typer.BadParameter("lineage manifest must contain an artifact_refs mapping")
        expected_prefix = f"{config.tracking.wandb_entity}/{config.render(config.tracking.wandb_project)}/"
        if any(not str(ref).startswith(expected_prefix) for ref in artifact_refs.values()):
            raise typer.BadParameter(f"every artifact ref must belong to {expected_prefix}")
        api = wandb.Api(timeout=60)
        verify_artifact_dag(api, artifact_refs)
        verify_publication_manifest_binding(api, lineage)
        if lineage.get("hf_commits", {}).get(repo_id) != validated_commit:
            raise typer.BadParameter("validated commit does not match the finalized publication manifest")
        make_public(repo_kind=kind, repo_id=repo_id, validated_commit=validated_commit)
        typer.echo(validated_commit)
        return
    sha = upload_private_then_validate(
        config,
        repo_kind=kind,
        repo_id=repo_id,
        source_dir=source_dir,
        card=card,
        required_phrases=("W&B Artifact", "このrepoのlicenseは未指定"),
        validation_dir=source_dir.parent / f"{source_dir.name}-hf-validation",
    )
    typer.echo(sha)


if __name__ == "__main__":
    app()

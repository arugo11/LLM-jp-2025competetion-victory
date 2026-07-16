from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from importlib.resources import files
from pathlib import Path
from typing import Literal

from .aime import normalize_lighteval_details, summarize_aime, verify_matched_fingerprint
from .config import ExperimentConfig
from .generation import Request, VllmBackend
from .hashing import sha256_directory, sha256_file, sha256_value
from .jsonl import read_jsonl, write_jsonl
from .math_utils import extract_last_boxed, math_equivalent
from .model_lineage import verify_merged_model_lineage
from .prompts import EVALUATION_PROMPT
from .schema import CuratedRecord
from .statistics import difficulty_summary

HARNESS_MANIFEST = "tv-gptoss120b-harness-manifest.json"
HARNESS_IGNORED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


def _harness_tree_digest(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.name == HARNESS_MANIFEST or HARNESS_IGNORED_PARTS.intersection(relative.parts):
            continue
        if path.is_symlink():
            entries.append({"path": str(relative), "symlink": os.readlink(path)})
        elif path.is_file():
            entries.append({"path": str(relative), "sha256": sha256_file(path)})
    if not entries:
        raise ValueError(f"Swallow harness worktree is empty: {root}")
    return sha256_value(entries)


def verify_swallow_harness(config: ExperimentConfig, harness_dir: Path) -> dict:
    manifest_path = harness_dir / HARNESS_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError("pinned AIME revision patch manifest is missing")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "swallow_tag",
        "swallow_commit",
        "aime_revision_patch_sha256",
        "resulting_diff_sha256",
        "worktree_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("Swallow harness manifest schema mismatch")
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=harness_dir, text=True
    ).strip()
    patch = Path(str(files("tv_gptoss120b").joinpath("resources/swallow-v202604-pin-aime-revisions.patch")))
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=harness_dir, text=True)
    expected = {
        "swallow_tag": config.evaluation.swallow_revision,
        "swallow_commit": config.evaluation.swallow_commit,
        "aime_revision_patch_sha256": sha256_file(patch),
        "resulting_diff_sha256": sha256_value(diff),
        "worktree_sha256": _harness_tree_digest(harness_dir),
    }
    if actual_commit != config.evaluation.swallow_commit or payload != expected:
        raise RuntimeError("Swallow harness commit, patch, or worktree digest mismatch")
    return payload


def prepare_swallow_harness(config: ExperimentConfig, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Swallow harness output is immutable: {output_dir}")
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "https://github.com/swallow-llm/swallow-evaluation-instruct.git",
            str(output_dir),
        ],
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", config.evaluation.swallow_commit], cwd=output_dir, check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=output_dir, text=True).strip()
    if actual != config.evaluation.swallow_commit:
        raise RuntimeError(f"Swallow commit mismatch: {actual}")
    patch = Path(str(files("tv_gptoss120b").joinpath("resources/swallow-v202604-pin-aime-revisions.patch")))
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=output_dir, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=output_dir, check=True)
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=output_dir)
    metadata = {
        "swallow_tag": config.evaluation.swallow_revision,
        "swallow_commit": actual,
        "aime_revision_patch_sha256": sha256_file(patch),
        "resulting_diff_sha256": sha256_value(diff.decode("utf-8")),
        "worktree_sha256": _harness_tree_digest(output_dir),
    }
    (output_dir / HARNESS_MANIFEST).write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def run_difficulty_inference(
    config: ExperimentConfig,
    arm: Literal["thinking", "instruct"],
    curated_path: Path,
    output_path: Path,
) -> list[dict]:
    config.require_assigned_id()
    model = getattr(config.models, arm)
    backend = VllmBackend(model, config, max_model_len=config.difficulty.max_model_len)
    records = [row for row in read_jsonl(curated_path, CuratedRecord) if row.split == "difficulty"]
    requests = []
    metadata = []
    for row in records:
        for sample_index in range(config.difficulty.num_samples):
            requests.append(Request(
                messages=[{"role": "user", "content": EVALUATION_PROMPT.format(problem=row.problem)}],
                temperature=config.difficulty.temperature,
                top_p=config.difficulty.top_p,
                max_tokens=config.difficulty.max_new_tokens,
                seed=config.difficulty.seed + sample_index + int(row.content_sha256[:8], 16),
            ))
            metadata.append((row, sample_index))
    responses = backend.chat(requests, config.difficulty.reasoning_effort)
    results = []
    for (row, sample_index), response in zip(metadata, responses, strict=True):
        answer = extract_last_boxed(response.text)
        results.append({
            "spec_id": row.spec_id,
            "generator_model": row.generator_model,
            "arm": arm,
            "sample_index": sample_index,
            "seed": requests[len(results)].seed,
            "raw_response": response.text,
            "parsed_answer": answer,
            "expected_answer": row.expected_answer,
            "correct": bool(answer and math_equivalent(answer, row.expected_answer, strict=True)),
            "finish_reason": response.finish_reason,
        })
    write_jsonl(output_path, results)
    return results


def summarize_difficulty(config: ExperimentConfig, result_paths: list[Path], output_path: Path) -> dict:
    rows = [row for path in result_paths for row in read_jsonl(path)]
    summary = difficulty_summary(
        rows,
        config.difficulty.bootstrap_samples,
        config.difficulty.seed,
        expected_samples=config.difficulty.num_samples,
    )
    if summary["spec_count"] < config.validation.valid_pair_min.difficulty:
        raise ValueError(
            f"difficulty holdout below quality gate: {summary['spec_count']} < "
            f"{config.validation.valid_pair_min.difficulty}"
        )
    summary["hardening_gate"] = bool(
        summary["pooled_delta"] >= config.difficulty.pooled_delta_min
        and summary["thinking_delta"] > config.difficulty.arm_delta_min
        and summary["instruct_delta"] > config.difficulty.arm_delta_min
    )
    if output_path.exists():
        raise FileExistsError(f"difficulty summary is immutable: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def _wait_for_server(process: subprocess.Popen, base_url: str, timeout_seconds: int = 1800) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited before readiness: {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(10)
    raise TimeoutError("vLLM server readiness timed out")


def matched_evaluation_fingerprint(config: ExperimentConfig) -> str:
    return sha256_value({
        "task": config.evaluation.task,
        "parser": config.evaluation.parser,
        "reasoning_effort": config.evaluation.reasoning_effort,
        "temperature": config.evaluation.temperature,
        "top_p": config.evaluation.top_p,
        "max_new_tokens": config.evaluation.max_new_tokens,
        "max_model_len": config.evaluation.max_model_len,
        "seed": config.evaluation.seed,
        "aime_2024": config.evaluation.aime_2024.model_dump(),
        "aime_2025": config.evaluation.aime_2025.model_dump(),
        "swallow_commit": config.evaluation.swallow_commit,
        "swallow_revision": config.evaluation.swallow_revision,
        "llm_jp_eval_revision": config.evaluation.llm_jp_eval_revision,
        "llm_jp_vllm_revision": config.evaluation.llm_jp_vllm_revision,
        "vllm_version": config.evaluation.vllm_version,
    })


def run_swallow_aime(
    config: ExperimentConfig,
    harness_dir: Path,
    model_path: str,
    model_revision: str | None,
    label: str,
    output_dir: Path,
    log_dir: Path,
    port: int,
    *,
    smoke: bool = False,
) -> dict:
    config.require_assigned_id()
    harness_evidence = verify_swallow_harness(config, harness_dir)
    harness_manifest_sha256 = sha256_file(harness_dir / HARNESS_MANIFEST)
    output_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}/v1"
    served_name = f"{config.identity.experiment_id}-{label}"
    package_root = Path(__file__).resolve().parents[2]
    environment_prefix = [
        "uv",
        "run",
        "--project",
        str(package_root),
        "--locked",
        "--extra",
        "generation",
    ]
    plugin_path = subprocess.check_output(
        [
            *environment_prefix,
            "python",
            "-c",
            "import llm_jp_vllm.llmjp4.reasoning_parser as p; print(p.__file__)",
        ],
        text=True,
    ).strip()
    server_command = [
        *environment_prefix, "vllm", "serve", model_path,
        "--host", "127.0.0.1", "--port", str(port), "--served-model-name", served_name,
        "--reasoning-parser-plugin", plugin_path,
        "--reasoning-parser", config.evaluation.parser,
        "--max-model-len", str(config.evaluation.max_model_len),
        "--data-parallel-size", str(config.generation.data_parallel_size),
        "--seed", str(config.evaluation.seed),
    ]
    if model_revision:
        server_command.extend(["--revision", model_revision])
    actual_vllm = subprocess.check_output(
        [
            *environment_prefix,
            "python",
            "-c",
            (
                "from importlib.metadata import version; "
                "import vllm; print(vllm.__version__ + ',' + version('llm-jp-vllm'))"
            ),
        ],
        text=True,
    ).strip()
    actual_vllm, actual_plugin = actual_vllm.split(",", maxsplit=1)
    if (
        actual_vllm != config.evaluation.vllm_version
        or actual_plugin != config.evaluation.llm_jp_vllm_revision.removeprefix("v")
    ):
        raise RuntimeError(
            f"evaluation runtime mismatch: vllm={actual_vllm}, llm-jp-vllm={actual_plugin}"
        )
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "EMPTY"
    with (log_dir / f"{label}-vllm.log").open("wb") as server_log:
        process = subprocess.Popen(
            server_command,
            cwd=harness_dir,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_server(process, base_url)
            generation = (
                f"{{temperature:{config.evaluation.temperature},top_p:{config.evaluation.top_p},"
                f"max_new_tokens:{config.evaluation.max_new_tokens},"
                f"reasoning_effort:{config.evaluation.reasoning_effort},max_n:4}}"
            )
            model_args = (
                f"model=hosted_vllm/{served_name},api_key=EMPTY,base_url={base_url},"
                f"generation_parameters={generation}"
            )
            command = [
                "uv", "run", "--isolated", "--locked", "--extra", "lighteval",
                "lighteval", "endpoint", "litellm", model_args, config.evaluation.task,
                "--use-chat-template", "--output-dir", str(output_dir), "--save-details",
            ]
            if smoke:
                command.extend(["--max-samples", "2"])
            completed = subprocess.run(command, cwd=harness_dir, env=env, capture_output=True, text=True)
            (log_dir / f"{label}-lighteval.stdout").write_text(completed.stdout, encoding="utf-8")
            (log_dir / f"{label}-lighteval.stderr").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"Swallow AIME failed for {label}: exit={completed.returncode}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    verify_swallow_harness(config, harness_dir)
    result_files = sorted(output_dir.glob("results/**/*.json"))
    detail_files = sorted(output_dir.glob("details/**/*.parquet"))
    if not result_files or not detail_files:
        raise RuntimeError("Swallow evaluation did not produce both result JSON and detail Parquet")
    normalized_path = output_dir / "normalized-aime.jsonl"
    normalized = normalize_lighteval_details(
        label,
        detail_files,
        normalized_path,
        expected_problem_count=2 if smoke else 30,
        metadata={
            "matched_config_sha256": matched_evaluation_fingerprint(config),
            "model_path": model_path,
            "model_revision": model_revision,
        },
    )
    if not smoke and len(normalized) != 240:
        raise RuntimeError(f"matched AIME evaluation must normalize to 240 samples per model, got {len(normalized)}")
    request_manifest = {
        "label": label,
        "model_path": model_path,
        "model_revision": model_revision,
        "matched_config_sha256": matched_evaluation_fingerprint(config),
        "server_command": server_command,
        "vllm_version": actual_vllm,
        "llm_jp_vllm_version": actual_plugin,
        "reasoning_parser_plugin": plugin_path,
        "harness_manifest_sha256": harness_manifest_sha256,
        "harness_worktree_sha256": harness_evidence["worktree_sha256"],
        "result_files": [str(path) for path in result_files],
        "detail_files": [str(path) for path in detail_files],
        "normalized_file": str(normalized_path),
        "normalized_sample_count": len(normalized),
    }
    (output_dir / "evaluation-request.json").write_text(
        json.dumps(request_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return request_manifest


def run_matched_aime_suite(
    config: ExperimentConfig,
    harness_dir: Path,
    models_manifest_path: Path,
    output_dir: Path,
    port: int,
) -> dict:
    config.require_assigned_id()
    manifest = json.loads(models_manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("models")
    if not isinstance(models, list):
        raise ValueError("matched model manifest must contain a models list")
    expected_labels = ["thinking-base", "thinking-post", "instruct-base", "instruct-post"]
    if [item.get("label") for item in models] != expected_labels:
        raise ValueError(f"matched model order must be exactly {expected_labels}")
    expected_base = {
        "thinking-base": config.models.thinking,
        "instruct-base": config.models.instruct,
    }
    experiment_root = config.experiment_root().resolve()
    for item in models:
        label = item["label"]
        if label in expected_base:
            model = expected_base[label]
            if (item.get("model_path"), item.get("model_revision")) != (model.repo_id, model.revision):
                raise ValueError(f"base model pin mismatch for {label}")
        else:
            model_path = Path(str(item.get("model_path", ""))).resolve()
            if not model_path.is_dir() or not model_path.is_relative_to(experiment_root):
                raise ValueError(f"post model must be an existing directory under EXP_DIR: {label}")
            if item.get("model_revision") is not None:
                raise ValueError(f"local merged model revision must be null: {label}")
            if not isinstance(item.get("model_sha256"), str) or len(item["model_sha256"]) != 64:
                raise ValueError(f"post model requires an immutable model_sha256: {label}")
            if sha256_directory(model_path) != item["model_sha256"]:
                raise ValueError(f"post model directory checksum mismatch: {label}")
            arm = "thinking" if label == "thinking-post" else "instruct"
            verify_merged_model_lineage(config, arm, model_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    requests = []
    normalized_paths = []
    for item in models:
        label = item["label"]
        model_output = output_dir / label
        request = run_swallow_aime(
            config,
            harness_dir,
            item["model_path"],
            item.get("model_revision"),
            label,
            model_output / "raw",
            model_output / "logs",
            port,
            smoke=False,
        )
        requests.append(request)
        normalized_paths.append(Path(request["normalized_file"]))
    combined = [row for path in normalized_paths for row in read_jsonl(path)]
    harness_manifest_hashes = {request["harness_manifest_sha256"] for request in requests}
    harness_worktree_hashes = {request["harness_worktree_sha256"] for request in requests}
    if len(harness_manifest_hashes) != 1 or len(harness_worktree_hashes) != 1:
        raise RuntimeError("matched AIME models did not use one immutable Swallow harness")
    fingerprint = matched_evaluation_fingerprint(config)
    verify_matched_fingerprint(combined, fingerprint)
    summary = summarize_aime(combined, bootstrap_samples=10_000, seed=config.evaluation.seed)
    combined_path = output_dir / "aime-all-models.jsonl"
    write_jsonl(combined_path, combined)
    summary_path = output_dir / "aime-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "matched_config_sha256": fingerprint,
        "harness_manifest_sha256": next(iter(harness_manifest_hashes)),
        "harness_worktree_sha256": next(iter(harness_worktree_hashes)),
        "model_requests": requests,
        "normalized_sample_count": len(combined),
        "combined_path": str(combined_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }

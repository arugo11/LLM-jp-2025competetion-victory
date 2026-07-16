from __future__ import annotations

import gc
import json
import multiprocessing
import os
import re
import selectors
import signal
import socket
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, Literal, Protocol

from .config import ExperimentConfig, ModelRef
from .hashing import sha256_file, sha256_value
from .jsonl import read_jsonl, write_jsonl
from .math_utils import extract_last_boxed, math_equivalent, problem_rule_rejections, solution_rule_rejections
from .prompts import PROMPT_REVISION, QUESTION_PROMPT, SOLUTION_PROMPT, VALIDATOR_PROMPT
from .schema import ProblemSpec, RawGeneration, ValidationVote


@dataclass(frozen=True)
class Request:
    messages: list[dict[str, str]]
    temperature: float
    top_p: float
    max_tokens: int
    seed: int


@dataclass(frozen=True)
class Response:
    text: str
    finish_reason: str


class ChatBackend(Protocol):
    def chat(self, requests: list[Request], reasoning_effort: str) -> list[Response]: ...

    def provenance(self) -> dict[str, str]: ...

    def close(self) -> None: ...


class FixtureBackend:
    def provenance(self) -> dict[str, str]:
        return {"backend": "fixture", "renderer_revision": "fixture-v1"}

    def chat(self, requests: list[Request], reasoning_effort: str) -> list[Response]:
        responses = []
        for request in requests:
            content = request.messages[-1]["content"]
            number = request.seed % 1_000_003 + 2
            if "入試テスト問題を一つ" in content:
                text = f"整数 $x$ が $x+1={number + 1}$ を満たすとき、$x$ の値を求めよ。"
            else:
                match = re.search(r"x\+1=(\d+)", content)
                if match:
                    number = int(match.group(1)) - 1
                text = f"方程式を整理すると $x={number}$ である。したがって $\\boxed{{{number}}}$。"
            responses.append(Response(text=text, finish_reason="stop"))
        return responses

    def close(self) -> None:
        return None


@contextmanager
def _managed_backend(backend: ChatBackend):
    try:
        yield backend
    except BaseException as primary:
        try:
            backend.close()
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "generation backend operation and cleanup both failed",
                [primary, cleanup],
            ) from None
        raise
    else:
        backend.close()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _shard_bounds(total: int, rank: int, size: int) -> tuple[int, int]:
    if size < 1 or not 0 <= rank < size:
        raise ValueError(f"invalid data-parallel rank {rank} for size {size}")
    floor, remainder = divmod(total, size)
    start = rank * floor + min(rank, remainder)
    end = (rank + 1) * floor + min(rank + 1, remainder)
    return start, end


def _shard_requests(requests: list[Request], size: int) -> list[list[tuple[int, Request]]]:
    shards = []
    for rank in range(size):
        start, end = _shard_bounds(len(requests), rank, size)
        shards.append(list(enumerate(requests[start:end], start=start)))
    return shards


def _merge_rank_responses(
    rank_payloads: dict[int, list[tuple[int, str, str]]],
    *,
    expected_indices_by_rank: dict[int, set[int]],
    expected_count: int,
    data_parallel_size: int,
) -> list[Response]:
    if set(rank_payloads) != set(range(data_parallel_size)):
        raise RuntimeError("data-parallel response is missing one or more ranks")
    if set(expected_indices_by_rank) != set(range(data_parallel_size)):
        raise RuntimeError("data-parallel request ownership is missing one or more ranks")
    indexed: dict[int, Response] = {}
    for rank in range(data_parallel_size):
        actual_indices = {index for index, _, _ in rank_payloads[rank]}
        if actual_indices != expected_indices_by_rank[rank]:
            raise RuntimeError(
                f"data-parallel rank ownership mismatch for rank {rank}: "
                f"expected={sorted(expected_indices_by_rank[rank])}, actual={sorted(actual_indices)}"
            )
        for index, text, finish_reason in rank_payloads[rank]:
            if index in indexed:
                raise RuntimeError(f"duplicate data-parallel response index: {index}")
            indexed[index] = Response(text=text, finish_reason=finish_reason)
    expected = set(range(expected_count))
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise RuntimeError(f"data-parallel response index mismatch: missing={missing}, extra={extra}")
    return [indexed[index] for index in range(expected_count)]


def _request_payload(request: Request) -> dict[str, Any]:
    return {
        "messages": request.messages,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "seed": request.seed,
    }


def _send_control(connection: Connection, message: dict[str, Any], *, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("timed out before vLLM DP control dispatch")
    with selectors.DefaultSelector() as selector:
        selector.register(connection.fileno(), selectors.EVENT_WRITE)
        if not selector.select(timeout=remaining):
            raise TimeoutError("timed out dispatching vLLM DP control message")
    connection.send(message)


def _write_request_shard(path: Path, shard: list[tuple[int, Request]]) -> str:
    payload = [(index, _request_payload(request)) for index, request in shard]
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    return sha256_file(path)


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def _signal_process_group(process_group: int, signal_number: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process_group, signal_number)


def _vllm_dp_worker(
    connection: Connection,
    *,
    rank: int,
    local_rank: int,
    data_parallel_size: int,
    master_ip: str,
    master_port: int,
    engine_kwargs: dict[str, Any],
    isolate_process_group: bool = True,
) -> None:
    # vLLM 0.18 offline DP requires one LLM process per rank; passing DP>1 to
    # one LLM instance is rejected. This mirrors examples/offline_inference/data_parallel.py.
    if isolate_process_group:
        os.setsid()
    process_group = os.getpgrp()
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_rank)
    os.environ["VLLM_DP_SIZE"] = str(data_parallel_size)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    llm: Any | None = None
    try:
        if "data_parallel_size" in engine_kwargs:
            raise ValueError("single-process LLM must not receive data_parallel_size > 1")
        from vllm import LLM, SamplingParams

        llm = LLM(**engine_kwargs)
        tokenizer = llm.get_tokenizer()
        connection.send({
            "kind": "ready",
            "rank": rank,
            "process_group": process_group,
            "provenance": {
                "backend": "vllm",
                "vllm_version": _package_version("vllm"),
                "transformers_version": _package_version("transformers"),
                "openai_harmony_version": _package_version("openai-harmony"),
                "tokenizer_class": tokenizer.__class__.__name__,
                "chat_template_sha256": sha256_value(tokenizer.chat_template),
            },
        })
        while True:
            message = connection.recv()
            if message["kind"] == "shutdown":
                break
            if message["kind"] != "chat":
                raise ValueError(f"unknown worker command: {message['kind']}")
            request_path = Path(message["request_path"])
            if sha256_file(request_path) != message["request_sha256"]:
                raise RuntimeError("data-parallel request shard digest mismatch")
            indexed_requests = json.loads(request_path.read_text(encoding="utf-8"))
            if not isinstance(indexed_requests, list):
                raise ValueError("data-parallel request shard must be a list")
            real_count = len(indexed_requests)
            if not indexed_requests:
                indexed_requests = [(-1, {
                    "messages": [{"role": "user", "content": "0"}],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1,
                    "seed": 0,
                })]
            params = [
                SamplingParams(
                    temperature=item["temperature"],
                    top_p=item["top_p"],
                    max_tokens=item["max_tokens"],
                    seed=item["seed"],
                )
                for _, item in indexed_requests
            ]
            outputs = llm.chat(
                [item["messages"] for _, item in indexed_requests],
                sampling_params=params,
                chat_template_kwargs={"reasoning_effort": message["reasoning_effort"]},
            )
            if len(outputs) != len(indexed_requests):
                raise RuntimeError("vLLM returned an unexpected number of responses")
            payload = []
            if real_count:
                payload = [
                    (index, output.outputs[0].text, str(output.outputs[0].finish_reason))
                    for (index, _), output in zip(indexed_requests, outputs, strict=True)
                ]
            connection.send({"kind": "result", "rank": rank, "responses": payload})
    except BaseException as error:
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send({
                "kind": "error",
                "rank": rank,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            })
        raise
    finally:
        if llm is not None:
            del llm
            gc.collect()
        connection.close()


class _ExplicitDataParallelPool:
    def __init__(self, model: ModelRef, config: ExperimentConfig, *, max_model_len: int | None = None) -> None:
        generation = config.generation
        if generation.data_parallel_size <= 1:
            raise ValueError("explicit data-parallel pool requires data_parallel_size > 1")
        actual_vllm = _package_version("vllm")
        if actual_vllm != generation.vllm_version:
            raise RuntimeError(
                f"generation runtime requires vLLM {generation.vllm_version}, found {actual_vllm}"
            )
        actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        if actual_python != config.runtime.python:
            raise RuntimeError(
                f"generation runtime requires Python {config.runtime.python}, found {actual_python}"
            )
        self._size = generation.data_parallel_size
        self._request_timeout = generation.worker_request_timeout_seconds
        self._shutdown_timeout = generation.worker_shutdown_timeout_seconds
        self._closed = False
        self._context = multiprocessing.get_context("spawn")
        self._connections: dict[int, Connection] = {}
        self._processes: dict[int, Any] = {}
        self._process_groups: dict[int, int] = {}
        self._request_sequence = 0
        self._request_directory = tempfile.TemporaryDirectory(prefix="tv-gptoss120b-vllm-dp-")
        engine_kwargs = {
            "model": model.repo_id,
            "revision": model.revision,
            "max_model_len": max_model_len or generation.max_model_len,
            "gpu_memory_utilization": generation.gpu_memory_utilization,
            "tensor_parallel_size": generation.tensor_parallel_size,
            "trust_remote_code": False,
        }
        master_ip = "127.0.0.1"
        master_port = _open_port()
        try:
            for rank in range(self._size):
                parent, child = self._context.Pipe(duplex=True)
                try:
                    process = self._context.Process(
                        target=_vllm_dp_worker,
                        kwargs={
                            "connection": child,
                            "rank": rank,
                            "local_rank": rank,
                            "data_parallel_size": self._size,
                            "master_ip": master_ip,
                            "master_port": master_port,
                            "engine_kwargs": engine_kwargs,
                        },
                        name=f"vllm-dp-rank-{rank}",
                    )
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self._connections[rank] = parent
                self._processes[rank] = process
                if process.pid is None:
                    raise RuntimeError(f"vLLM DP rank {rank} has no process ID after start")
                self._process_groups[rank] = process.pid
            ready = self._collect("ready", generation.worker_startup_timeout_seconds)
            for rank, message in ready.items():
                if message.get("process_group") != self._process_groups[rank]:
                    raise RuntimeError(f"vLLM DP rank {rank} did not establish its dedicated process group")
            rank_provenance = {rank: message["provenance"] for rank, message in ready.items()}
            if any(value != rank_provenance[0] for value in rank_provenance.values()):
                raise RuntimeError("renderer provenance differs across data-parallel ranks")
            self.provenance = {
                **rank_provenance[0],
                "parallel_launcher": "explicit-multiprocess-vllm-v0.18",
                "tensor_parallel_size": str(generation.tensor_parallel_size),
                "data_parallel_size": str(generation.data_parallel_size),
                "sharding": "contiguous-balanced-input-order-v1",
                "empty_rank_policy": "single-placeholder-discarded-v1",
                "worker_lifecycle": "process-group-per-model-v1",
                "ipc_transport": "control-pipe-json-file-v1",
            }
        except BaseException:
            self._force_stop()
            raise

    def _collect(
        self,
        expected_kind: str,
        timeout_seconds: int | None = None,
        *,
        deadline: float | None = None,
    ) -> dict[int, dict[str, Any]]:
        if deadline is None:
            if timeout_seconds is None:
                raise ValueError("vLLM DP collection requires a timeout or deadline")
            deadline = time.monotonic() + timeout_seconds
        pending = dict(self._connections)
        messages: dict[int, dict[str, Any]] = {}
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for vLLM DP {expected_kind}: ranks={sorted(pending)}")
            ready_connections = wait(list(pending.values()), timeout=min(1.0, remaining))
            for connection in ready_connections:
                rank = next(key for key, value in pending.items() if value is connection)
                try:
                    message = connection.recv()
                except EOFError as error:
                    raise RuntimeError(f"vLLM DP rank {rank} exited without a message") from error
                if message.get("kind") == "error":
                    raise RuntimeError(
                        f"vLLM DP rank {rank} failed: {message['error_type']}: {message['message']}\n"
                        f"{message['traceback']}"
                    )
                if message.get("kind") != expected_kind or message.get("rank") != rank:
                    raise RuntimeError(f"unexpected vLLM DP message from rank {rank}: {message!r}")
                messages[rank] = message
                del pending[rank]
            for rank in list(pending):
                process = self._processes[rank]
                if process.exitcode is not None:
                    raise RuntimeError(f"vLLM DP rank {rank} exited early with code {process.exitcode}")
        return messages

    def chat(self, requests: list[Request], reasoning_effort: str) -> list[Response]:
        if self._closed:
            raise RuntimeError("vLLM data-parallel pool is closed")
        shards = _shard_requests(requests, self._size)
        expected_indices = {
            rank: {index for index, _ in shard}
            for rank, shard in enumerate(shards)
        }
        deadline = time.monotonic() + self._request_timeout
        request_paths: list[Path] = []
        try:
            for rank, shard in enumerate(shards):
                path = Path(self._request_directory.name) / f"request-{self._request_sequence:04d}-rank-{rank}.json"
                request_paths.append(path)
                digest = _write_request_shard(path, shard)
                _send_control(
                    self._connections[rank],
                    {
                        "kind": "chat",
                        "reasoning_effort": reasoning_effort,
                        "request_path": str(path),
                        "request_sha256": digest,
                    },
                    deadline=deadline,
                )
            self._request_sequence += 1
            messages = self._collect("result", deadline=deadline)
            rank_payloads = {rank: message["responses"] for rank, message in messages.items()}
            return _merge_rank_responses(
                rank_payloads,
                expected_indices_by_rank=expected_indices,
                expected_count=len(requests),
                data_parallel_size=self._size,
            )
        except BaseException:
            self._force_stop()
            raise
        finally:
            for path in request_paths:
                path.unlink(missing_ok=True)

    def _force_stop(self) -> None:
        self._closed = True
        for connection in self._connections.values():
            connection.close()
        for process_group in self._process_groups.values():
            _signal_process_group(process_group, signal.SIGTERM)
        for process in self._processes.values():
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 5
        for process in self._processes.values():
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process_group in self._process_groups.values():
            if _process_group_exists(process_group):
                _signal_process_group(process_group, signal.SIGKILL)
        for process in self._processes.values():
            if process.is_alive():
                process.kill()
            process.join(timeout=5)
            if process.exitcode is not None:
                process.close()
        self._request_directory.cleanup()

    def close(self) -> None:
        if self._closed:
            return
        try:
            deadline = time.monotonic() + self._shutdown_timeout
            for connection in self._connections.values():
                _send_control(connection, {"kind": "shutdown"}, deadline=deadline)
            for rank, process in self._processes.items():
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    process.join(timeout=remaining)
                if process.is_alive():
                    raise TimeoutError(f"vLLM DP rank {rank} did not shut down")
                if process.exitcode != 0:
                    raise RuntimeError(f"vLLM DP rank {rank} exited with code {process.exitcode}")
                if _process_group_exists(self._process_groups[rank]):
                    raise RuntimeError(f"vLLM DP rank {rank} left descendant processes running")
            self._closed = True
            for connection in self._connections.values():
                connection.close()
            for process in self._processes.values():
                process.close()
            self._request_directory.cleanup()
        except BaseException:
            self._force_stop()
            raise


class VllmBackend:
    def __init__(self, model: ModelRef, config: ExperimentConfig, *, max_model_len: int | None = None) -> None:
        self._pool = _ExplicitDataParallelPool(model, config, max_model_len=max_model_len)

    def provenance(self) -> dict[str, str]:
        return dict(self._pool.provenance)

    def chat(self, requests: list[Request], reasoning_effort: str) -> list[Response]:
        return self._pool.chat(requests, reasoning_effort)

    def close(self) -> None:
        self._pool.close()


def _model_ref(config: ExperimentConfig, model_key: str) -> ModelRef:
    return getattr(config.models, model_key)


def _backend(name: Literal["fixture", "vllm"], model: ModelRef, config: ExperimentConfig) -> ChatBackend:
    return FixtureBackend() if name == "fixture" else VllmBackend(model, config)


def generate_problems(
    config: ExperimentConfig,
    specs_path: Path,
    output_path: Path,
    model_key: Literal["generator_20b", "generator_120b"],
    backend_name: Literal["fixture", "vllm"],
) -> list[RawGeneration]:
    specs = read_jsonl(specs_path, ProblemSpec)
    model = _model_ref(config, model_key)
    backend = _backend(backend_name, model, config)
    with _managed_backend(backend):
        renderer_provenance = backend.provenance()
        question_requests = [
            Request(
                messages=[{"role": "user", "content": QUESTION_PROMPT.format(**spec.model_dump())}],
                temperature=config.generation.question_temperature,
                top_p=config.generation.top_p,
                max_tokens=config.generation.max_tokens,
                seed=spec.generation_seed,
            )
            for spec in specs
        ]
        question_responses = backend.chat(question_requests, config.generation.reasoning_effort)
        solution_requests = [
            Request(
                messages=[{"role": "user", "content": SOLUTION_PROMPT.format(problem=response.text)}],
                temperature=config.generation.solution_temperature,
                top_p=config.generation.top_p,
                max_tokens=config.generation.max_tokens,
                seed=spec.generation_seed ^ 0x5A5A5A5A,
            )
            for spec, response in zip(specs, question_responses, strict=True)
        ]
        solution_responses = backend.chat(solution_requests, config.generation.reasoning_effort)

        rows = []
        for spec, q_request, q_response, s_request, s_response in zip(
            specs, question_requests, question_responses, solution_requests, solution_responses, strict=True
        ):
            answer = extract_last_boxed(s_response.text)
            reasons = problem_rule_rejections(q_response.text)
            reasons.extend(solution_rule_rejections(s_response.text, answer))
            payload = {
                **spec.model_dump(exclude={"spec_sha256"}),
                "generator_model": model_key,
                "generator_repo_id": model.repo_id,
                "generator_revision": model.revision,
                "prompt_revision": PROMPT_REVISION,
                "reasoning_effort": config.generation.reasoning_effort,
                "renderer_provenance": renderer_provenance,
                "raw_question_request": {
                    "messages": q_request.messages,
                    "temperature": q_request.temperature,
                    "top_p": q_request.top_p,
                    "max_tokens": q_request.max_tokens,
                    "seed": q_request.seed,
                },
                "raw_question_response": q_response.text,
                "problem": q_response.text,
                "raw_solution_request": {
                    "messages": s_request.messages,
                    "temperature": s_request.temperature,
                    "top_p": s_request.top_p,
                    "max_tokens": s_request.max_tokens,
                    "seed": s_request.seed,
                },
                "raw_solution_response": s_response.text,
                "solution_cot": s_response.text,
                "expected_answer": answer,
                "finish_reason": f"question={q_response.finish_reason};solution={s_response.finish_reason}",
                "parse_status": "rejected" if reasons else "ok",
                "reject_reasons": reasons,
            }
            rows.append(RawGeneration(**payload, content_sha256=sha256_value(payload)))
    write_jsonl(output_path, rows)
    return rows


def validate_problems(
    config: ExperimentConfig,
    raw_paths: list[Path],
    output_path: Path,
    validator_key: Literal["generator_20b", "generator_120b"],
    backend_name: Literal["fixture", "vllm"],
) -> list[ValidationVote]:
    raw_rows = [row for path in raw_paths for row in read_jsonl(path, RawGeneration)]
    model = _model_ref(config, validator_key)
    backend = _backend(backend_name, model, config)
    with _managed_backend(backend):
        renderer_provenance = backend.provenance()
        requests = [
            Request(
                messages=[{"role": "user", "content": VALIDATOR_PROMPT.format(problem=row.problem)}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=config.generation.max_tokens,
                seed=row.generation_seed ^ (0x2020 if validator_key == "generator_20b" else 0x120120),
            )
            for row in raw_rows
        ]
        responses = backend.chat(requests, config.generation.reasoning_effort)
        votes = []
        for row, response in zip(raw_rows, responses, strict=True):
            parsed = extract_last_boxed(response.text)
            equivalent = bool(
                parsed
                and row.expected_answer
                and math_equivalent(parsed, row.expected_answer, strict=backend_name != "fixture")
            )
            payload = {
                "spec_id": row.spec_id,
                "generator_model": row.generator_model,
                "validator_model": validator_key,
                "validator_revision": model.revision,
                "raw_request": {
                    "messages": requests[len(votes)].messages,
                    "temperature": requests[len(votes)].temperature,
                    "top_p": requests[len(votes)].top_p,
                    "max_tokens": requests[len(votes)].max_tokens,
                    "seed": requests[len(votes)].seed,
                    "reasoning_effort": config.generation.reasoning_effort,
                },
                "renderer_provenance": renderer_provenance,
                "raw_response": response.text,
                "parsed_answer": parsed,
                "equivalent_to_reference": equivalent,
                "parse_status": "ok" if parsed else "rejected",
            }
            votes.append(ValidationVote(**payload, content_sha256=sha256_value(payload)))
    write_jsonl(output_path, votes)
    return votes

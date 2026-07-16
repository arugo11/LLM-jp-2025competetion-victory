import os
import sys
from pathlib import Path

import pytest

from tv_gptoss120b import generation
from tv_gptoss120b.config import load_config
from tv_gptoss120b.generation import Request, Response, generate_problems
from tv_gptoss120b.jsonl import write_jsonl
from tv_gptoss120b.specs import build_specs

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def _requests(count: int) -> list[Request]:
    return [
        Request(
            messages=[{"role": "user", "content": f"prompt-{index}"}],
            temperature=index / 10,
            top_p=1.0 - index / 100,
            max_tokens=100 + index,
            seed=1000 + index,
        )
        for index in range(count)
    ]


def test_explicit_dp_sharding_is_balanced_and_preserves_request_objects() -> None:
    requests = _requests(11)
    shards = generation._shard_requests(requests, 8)

    assert [len(shard) for shard in shards] == [2, 2, 2, 1, 1, 1, 1, 1]
    flattened = [item for shard in shards for item in shard]
    assert [index for index, _ in flattened] == list(range(11))
    assert [request for _, request in flattened] == requests


def test_explicit_dp_reassembly_restores_input_order() -> None:
    shards = generation._shard_requests(_requests(11), 8)
    payloads = {
        rank: [(index, f"answer-{index}", "stop") for index, _ in shard]
        for rank, shard in enumerate(shards)
    }

    responses = generation._merge_rank_responses(
        payloads,
        expected_indices_by_rank={rank: {index for index, _ in shard} for rank, shard in enumerate(shards)},
        expected_count=11,
        data_parallel_size=8,
    )

    assert responses == [Response(text=f"answer-{index}", finish_reason="stop") for index in range(11)]


def test_explicit_dp_reassembly_rejects_missing_rank() -> None:
    with pytest.raises(RuntimeError, match="missing one or more ranks"):
        generation._merge_rank_responses(
            {rank: [] for rank in range(7)},
            expected_indices_by_rank={rank: set() for rank in range(8)},
            expected_count=0,
            data_parallel_size=8,
        )


def test_explicit_dp_reassembly_rejects_duplicate_and_wrong_rank_ownership() -> None:
    payloads = {rank: [] for rank in range(8)}
    payloads[0] = [(0, "a", "stop")]
    payloads[1] = [(0, "b", "stop")]
    ownership = {rank: set() for rank in range(8)}
    ownership[0] = {0}
    ownership[1] = {0}
    with pytest.raises(RuntimeError, match="duplicate"):
        generation._merge_rank_responses(
            payloads,
            expected_indices_by_rank=ownership,
            expected_count=1,
            data_parallel_size=8,
        )

    payloads[0], payloads[1] = payloads[1], payloads[0]
    ownership[0], ownership[1] = {0}, set()
    with pytest.raises(RuntimeError, match="rank ownership mismatch"):
        generation._merge_rank_responses(
            payloads,
            expected_indices_by_rank=ownership,
            expected_count=1,
            data_parallel_size=8,
        )


def test_worker_rejects_single_process_dp_argument_before_importing_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    imported_vllm = False

    class FakeConnection:
        def send(self, message: dict) -> None:
            assert message["kind"] == "error"

        def close(self) -> None:
            return None

    original_import = __import__

    def guarded_import(name: str, *args, **kwargs):
        nonlocal imported_vllm
        if name == "vllm":
            imported_vllm = True
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    with pytest.raises(ValueError, match="must not receive data_parallel_size"):
        generation._vllm_dp_worker(
            FakeConnection(),
            rank=0,
            local_rank=0,
            data_parallel_size=8,
            master_ip="127.0.0.1",
            master_port=12345,
            engine_kwargs={"data_parallel_size": 8},
            isolate_process_group=False,
        )
    assert imported_vllm is False


def test_generation_failure_closes_backend_and_writes_no_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG)
    specs_path = tmp_path / "specs.jsonl"
    output_path = tmp_path / "raw.jsonl"
    write_jsonl(specs_path, build_specs(config.data)[:1])

    class FailingBackend:
        closed = False
        calls = 0

        def provenance(self) -> dict[str, str]:
            return {"backend": "test"}

        def chat(self, requests: list[Request], reasoning_effort: str) -> list[Response]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("rank 3 failed")
            return [Response(text="problem", finish_reason="stop") for _ in requests]

        def close(self) -> None:
            self.closed = True

    backend = FailingBackend()
    monkeypatch.setattr(generation, "_backend", lambda *args, **kwargs: backend)

    with pytest.raises(RuntimeError, match="rank 3 failed"):
        generate_problems(config, specs_path, output_path, "generator_20b", "vllm")

    assert backend.closed is True
    assert not output_path.exists()


def test_generation_preserves_primary_and_cleanup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG)
    specs_path = tmp_path / "specs.jsonl"
    output_path = tmp_path / "raw.jsonl"
    write_jsonl(specs_path, build_specs(config.data)[:1])

    class DoublyFailingBackend:
        def provenance(self) -> dict[str, str]:
            return {"backend": "test"}

        def chat(self, requests: list[Request], reasoning_effort: str) -> list[Response]:
            raise RuntimeError("primary inference failure")

        def close(self) -> None:
            raise RuntimeError("cleanup failure")

    monkeypatch.setattr(generation, "_backend", lambda *args, **kwargs: DoublyFailingBackend())

    with pytest.raises(BaseExceptionGroup) as captured:
        generate_problems(config, specs_path, output_path, "generator_20b", "vllm")

    assert [str(error) for error in captured.value.exceptions] == [
        "primary inference failure",
        "cleanup failure",
    ]
    assert not output_path.exists()


def test_each_model_backend_is_closed_before_next_model_initializes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG)
    specs_path = tmp_path / "specs.jsonl"
    write_jsonl(specs_path, build_specs(config.data)[:1])
    backends = []

    class TrackingBackend:
        def __init__(self) -> None:
            if backends:
                assert backends[-1].closed is True
            self.closed = False
            backends.append(self)

        def provenance(self) -> dict[str, str]:
            return {"backend": "test"}

        def chat(self, requests: list[Request], reasoning_effort: str) -> list[Response]:
            if "入試テスト問題を一つ" in requests[0].messages[0]["content"]:
                text = "整数 x について x=1 を求めよ。"
            else:
                text = "したがって \\boxed{1}。"
            return [Response(text=text, finish_reason="stop") for _ in requests]

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(generation, "_backend", lambda *args, **kwargs: TrackingBackend())

    generate_problems(config, specs_path, tmp_path / "raw-20b.jsonl", "generator_20b", "vllm")
    generate_problems(config, specs_path, tmp_path / "raw-120b.jsonl", "generator_120b", "vllm")

    assert len(backends) == 2
    assert all(backend.closed for backend in backends)


def test_generation_topology_drift_is_rejected_by_yaml_model() -> None:
    config = load_config(CONFIG)
    payload = config.generation.model_dump()
    payload["data_parallel_size"] = 1
    with pytest.raises(ValueError, match="TP=1 x explicit-process DP=8"):
        type(config.generation).model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question_temperature", -0.1),
        ("solution_temperature", -0.1),
        ("top_p", 0),
        ("top_p", 1.1),
        ("max_tokens", 0),
        ("max_model_len", 0),
        ("gpu_memory_utilization", 0),
        ("gpu_memory_utilization", 1.1),
    ],
)
def test_generation_config_rejects_invalid_runtime_values(field: str, value: float) -> None:
    config = load_config(CONFIG)
    payload = config.generation.model_dump()
    payload[field] = value
    with pytest.raises(ValueError):
        type(config.generation).model_validate(payload)


FAKE_VLLM = '''
import os


class SamplingParams:
    def __init__(self, *, temperature, top_p, max_tokens, seed):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed


class _Candidate:
    def __init__(self, text):
        self.text = text
        self.finish_reason = "stop"


class _Output:
    def __init__(self, text):
        self.outputs = [_Candidate(text)]


class _Tokenizer:
    chat_template = "fake-template-v1"


class LLM:
    def __init__(self, **kwargs):
        if "data_parallel_size" in kwargs:
            raise AssertionError("single-process DP argument leaked")
        self.rank = int(os.environ["VLLM_DP_RANK"])

    def get_tokenizer(self):
        return _Tokenizer()

    def chat(self, messages, *, sampling_params, chat_template_kwargs):
        outputs = []
        for item, params in zip(messages, sampling_params, strict=True):
            content = item[-1]["content"]
            if content == "FAIL" and self.rank == 3:
                raise RuntimeError("injected rank failure")
            outputs.append(_Output(
                f"{content}|rank={self.rank}|seed={params.seed}|effort={chat_template_kwargs['reasoning_effort']}"
            ))
        return outputs
'''


def _install_fake_vllm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text(FAKE_VLLM, encoding="utf-8")
    metadata = tmp_path / "vllm-0.18.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: vllm\nVersion: 0.18.0\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) if not existing else f"{tmp_path}{os.pathsep}{existing}")


@pytest.mark.skipif(sys.platform == "win32", reason="ABCI process-group contract is POSIX-only")
def test_real_explicit_pool_protocol_with_spawned_fake_vllm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vllm(tmp_path, monkeypatch)
    config = load_config(CONFIG)
    pool = generation._ExplicitDataParallelPool(config.models.generator_20b, config)

    assert pool.chat([], "high") == []
    for count in (1, 7, 8, 9, 11):
        requests = _requests(count)
        responses = pool.chat(requests, "high")
        expected_ranks = [
            rank
            for rank, shard in enumerate(generation._shard_requests(requests, 8))
            for _ in shard
        ]
        assert [response.text for response in responses] == [
            f"prompt-{index}|rank={rank}|seed={1000 + index}|effort=high"
            for index, rank in enumerate(expected_ranks)
        ]
    process_groups = list(pool._process_groups.values())
    pool.close()
    assert all(not generation._process_group_exists(group) for group in process_groups)


@pytest.mark.skipif(sys.platform == "win32", reason="ABCI process-group contract is POSIX-only")
def test_real_explicit_pool_rank_failure_stops_every_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vllm(tmp_path, monkeypatch)
    config = load_config(CONFIG)
    pool = generation._ExplicitDataParallelPool(config.models.generator_20b, config)
    requests = _requests(32)
    requests[12] = Request(
        messages=[{"role": "user", "content": "FAIL"}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        seed=12,
    )
    process_groups = list(pool._process_groups.values())

    with pytest.raises(RuntimeError, match="injected rank failure"):
        pool.chat(requests, "high")

    assert pool._closed is True
    assert all(not generation._process_group_exists(group) for group in process_groups)


def test_vllm_version_mismatch_fails_before_process_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG)
    monkeypatch.setattr(generation, "_package_version", lambda name: "0.17.0")
    monkeypatch.setattr(
        generation.multiprocessing,
        "get_context",
        lambda *args, **kwargs: pytest.fail("process context must not be created"),
    )
    with pytest.raises(RuntimeError, match="requires vLLM 0.18.0"):
        generation._ExplicitDataParallelPool(config.models.generator_20b, config)

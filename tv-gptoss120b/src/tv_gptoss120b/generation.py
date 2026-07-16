from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, Protocol

from .config import ExperimentConfig, ModelRef
from .hashing import sha256_value
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


class VllmBackend:
    def __init__(self, model: ModelRef, config: ExperimentConfig, *, max_model_len: int | None = None) -> None:
        from vllm import LLM, SamplingParams

        self._sampling_params_cls = SamplingParams
        self._llm = LLM(
            model=model.repo_id,
            revision=model.revision,
            max_model_len=max_model_len or config.generation.max_model_len,
            gpu_memory_utilization=config.generation.gpu_memory_utilization,
            tensor_parallel_size=config.generation.tensor_parallel_size,
            data_parallel_size=config.generation.data_parallel_size,
            trust_remote_code=False,
        )

    def provenance(self) -> dict[str, str]:
        tokenizer = self._llm.get_tokenizer()

        def package_version(name: str) -> str:
            try:
                return version(name)
            except PackageNotFoundError:
                return "not-installed"

        return {
            "backend": "vllm",
            "vllm_version": package_version("vllm"),
            "transformers_version": package_version("transformers"),
            "openai_harmony_version": package_version("openai-harmony"),
            "tokenizer_class": tokenizer.__class__.__name__,
            "chat_template_sha256": sha256_value(tokenizer.chat_template),
        }

    def chat(self, requests: list[Request], reasoning_effort: str) -> list[Response]:
        params = [
            self._sampling_params_cls(
                temperature=item.temperature,
                top_p=item.top_p,
                max_tokens=item.max_tokens,
                seed=item.seed,
            )
            for item in requests
        ]
        outputs = self._llm.chat(
            [item.messages for item in requests],
            sampling_params=params,
            chat_template_kwargs={"reasoning_effort": reasoning_effort},
        )
        return [
            Response(text=item.outputs[0].text, finish_reason=str(item.outputs[0].finish_reason))
            for item in outputs
        ]


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

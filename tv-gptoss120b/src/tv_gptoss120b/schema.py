from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProblemSpec(StrictModel):
    spec_id: str
    category: str
    unit: str
    declared_difficulty: int = Field(ge=1, le=10)
    split: Literal["difficulty", "sft", "grpo"]
    generation_seed: int
    spec_sha256: str


class RawGeneration(StrictModel):
    spec_id: str
    generator_model: Literal["generator_20b", "generator_120b"]
    generator_repo_id: str
    generator_revision: str
    category: str
    unit: str
    declared_difficulty: int
    split: Literal["difficulty", "sft", "grpo"]
    generation_seed: int
    prompt_revision: str
    reasoning_effort: str
    renderer_provenance: dict[str, str]
    raw_question_request: dict
    raw_question_response: str
    problem: str | None
    raw_solution_request: dict | None = None
    raw_solution_response: str | None = None
    solution_cot: str | None = None
    expected_answer: str | None = None
    finish_reason: str | None = None
    parse_status: Literal["ok", "rejected"]
    reject_reasons: list[str] = Field(default_factory=list)
    content_sha256: str


class ValidationVote(StrictModel):
    spec_id: str
    generator_model: Literal["generator_20b", "generator_120b"]
    validator_model: Literal["generator_20b", "generator_120b"]
    validator_revision: str
    raw_request: dict
    renderer_provenance: dict[str, str]
    raw_response: str
    parsed_answer: str | None
    equivalent_to_reference: bool
    parse_status: Literal["ok", "rejected"]
    content_sha256: str


class CuratedRecord(StrictModel):
    spec_id: str
    generator_model: Literal["generator_20b", "generator_120b"]
    generator_revision: str
    category: str
    unit: str
    declared_difficulty: int
    problem: str
    solution_cot: str
    expected_answer: str
    generation_seed: int
    prompt_revision: str
    validation_status: Literal["accepted"]
    validation_votes: dict[str, bool]
    split: Literal["difficulty", "sft", "grpo"]
    content_sha256: str

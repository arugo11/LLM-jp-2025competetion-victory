from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .hashing import sha256_value

SHA40 = re.compile(r"^[0-9a-f]{40}$")
EXPERIMENT_ID = re.compile(r"^[0-9]{4}$")
PRODUCTION_CONTRACT_SHA256 = "4b06bab6c0bcfc260fca94a09d162eb068872aaf3d3b91f5289d022be0c9caeb"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IdentityConfig(StrictModel):
    experiment_id: str
    slug: Literal["tv-gptoss120b"]
    owner: str
    group: Literal["gcg51557"]
    upstream_repo: str
    upstream_revision: str
    fork_repo: str


class PathsConfig(StrictModel):
    experiment_root: str
    data_dir: str
    model_dir: str
    output_dir: str
    log_dir: str


class DecisionEvidence(StrictModel):
    status: Literal["user-fixed", "meeting-verified", "slack-confirmed", "unresolved"]
    source_id: str = Field(min_length=1)
    note: str = Field(min_length=1)


class CoordinationConfig(StrictModel):
    matched_evaluation: DecisionEvidence
    team_evaluation_harness: DecisionEvidence
    common_evaluation_procedure: DecisionEvidence
    aime_2025_split: DecisionEvidence

    @model_validator(mode="after")
    def preserve_known_authority_and_conflict(self) -> CoordinationConfig:
        expected = {
            "matched_evaluation": "meeting-verified",
            "team_evaluation_harness": "slack-confirmed",
            "common_evaluation_procedure": "unresolved",
            "aime_2025_split": "user-fixed",
        }
        for name, status in expected.items():
            if getattr(self, name).status != status:
                raise ValueError(f"coordination authority drift: {name} must remain {status}")
        return self


StageName = Literal[
    "audit",
    "generate",
    "curate",
    "difficulty",
    "preprocess",
    "sft",
    "grpo",
    "evaluate",
    "publish",
]


class RuntimeProfile(StrictModel):
    venue: Literal["local", "compute", "decision_required"]
    resource_class: Literal["local_cpu", "cpu_only", "h200"]
    stages: list[StageName]
    uv_extras: list[Literal["data", "generation", "training", "tracking", "dev"]]
    nodes: Literal[1]
    gpus_per_node: Literal[0, 8]
    cpu_workers: int = Field(gt=0)
    threads_per_worker: int = Field(gt=0)
    io_concurrency: int = Field(gt=0)
    environment: dict[str, str]
    approval_gate: Literal["none", "qsub", "publication"]
    hub_upload_allowed: bool

    @model_validator(mode="after")
    def resource_and_concurrency_are_consistent(self) -> RuntimeProfile:
        if self.resource_class == "h200" and self.gpus_per_node != 8:
            raise ValueError("H200 profile must request the approved single 8-GPU node")
        if self.resource_class != "h200" and self.gpus_per_node != 0:
            raise ValueError("CPU profiles must request zero GPUs")
        if self.venue in {"compute", "decision_required"} and self.approval_gate == "none":
            raise ValueError("compute profiles require an explicit approval gate")
        if self.resource_class == "h200" and self.hub_upload_allowed:
            raise ValueError("GPU profiles must not upload to Hugging Face")
        if self.cpu_workers * self.threads_per_worker > 8 and self.resource_class == "local_cpu":
            raise ValueError("local profile exceeds the bounded local CPU concurrency")
        required_thread_env = {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
        if self.cpu_workers > 1 and not required_thread_env <= self.environment.keys():
            raise ValueError("process-parallel profiles must pin native-library thread counts")
        return self


class RuntimeConfig(StrictModel):
    python: Literal["3.12"]
    package_manager: Literal["uv"]
    profiles: dict[str, RuntimeProfile]
    stage_profile: dict[StageName, str]
    transient_cluster_fields: list[
        Literal["queue", "billing_mode", "resource_type", "submit_account", "team_approval_ref"]
    ]

    @model_validator(mode="after")
    def profiles_cover_every_stage_once(self) -> RuntimeConfig:
        expected_profiles = {"local_audit", "cpu_pipeline", "h200_pipeline", "cpu_publication"}
        if set(self.profiles) != expected_profiles:
            raise ValueError(f"runtime profiles must be exactly {sorted(expected_profiles)}")
        expected_stages = set(StageName.__args__)
        if set(self.stage_profile) != expected_stages:
            raise ValueError("runtime stage_profile must cover every stage exactly once")
        for stage, profile_name in self.stage_profile.items():
            profile = self.profiles.get(profile_name)
            if profile is None or stage not in profile.stages:
                raise ValueError(f"runtime stage/profile mismatch: {stage} -> {profile_name}")
        if set(self.transient_cluster_fields) != {
            "queue", "billing_mode", "resource_type", "submit_account", "team_approval_ref"
        }:
            raise ValueError("all transient cluster fields must be resolved from a fresh policy snapshot")
        return self


class ModelRef(StrictModel):
    repo_id: str
    revision: str

    @model_validator(mode="after")
    def revision_is_immutable(self) -> ModelRef:
        if not SHA40.fullmatch(self.revision):
            raise ValueError(f"model revision must be a 40-character commit SHA: {self.repo_id}")
        return self


class ModelsConfig(StrictModel):
    generator_20b: ModelRef
    generator_120b: ModelRef
    thinking: ModelRef
    instruct: ModelRef


class SplitCounts(StrictModel):
    difficulty: int
    sft: int
    grpo: int


class DataConfig(StrictModel):
    spec_count: int
    split_counts: SplitCounts
    difficulty_min: int
    difficulty_max: int
    seed: int

    @model_validator(mode="after")
    def split_total_matches(self) -> DataConfig:
        total = self.split_counts.difficulty + self.split_counts.sft + self.split_counts.grpo
        if total != self.spec_count:
            raise ValueError(f"split counts total {total} != spec_count {self.spec_count}")
        if not 1 <= self.difficulty_min <= self.difficulty_max <= 10:
            raise ValueError("difficulty range must be within 1..10")
        return self


class GenerationConfig(StrictModel):
    reasoning_effort: Literal["low", "medium", "high"]
    question_temperature: float
    solution_temperature: float
    top_p: float
    max_tokens: int
    max_model_len: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    data_parallel_size: int


class ValidationConfig(StrictModel):
    parse_rate_min: float
    validity_noninferiority_margin: float
    valid_pair_min: SplitCounts
    ngram_size: int
    similarity_flag_threshold: float
    internal_similarity_threshold: float
    manual_review_pairs: int


class DifficultyConfig(StrictModel):
    num_samples: int
    reasoning_effort: Literal["medium"]
    temperature: float
    top_p: float
    max_new_tokens: int
    max_model_len: int
    seed: int
    bootstrap_samples: int
    pooled_delta_min: float
    arm_delta_min: float


class SftConfig(StrictModel):
    max_length: int
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    max_grad_norm: float
    seed: int
    packing: Literal[False]
    assistant_only_loss: Literal[True]


class GrpoConfig(StrictModel):
    max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    temperature: float
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    beta: float
    epsilon: float
    epsilon_high: float
    loss_type: Literal["dapo"]
    scale_rewards: Literal[False]
    mask_truncated_completions: Literal[True]
    use_vllm: Literal[False]
    seed: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    parser_success_min: float
    informative_group_min: float
    truncation_rate_max: float


class DatasetRef(StrictModel):
    repo_id: str
    revision: str
    split: Literal["train"]

    @model_validator(mode="after")
    def revision_is_immutable(self) -> DatasetRef:
        if not SHA40.fullmatch(self.revision):
            raise ValueError(f"dataset revision must be a 40-character commit SHA: {self.repo_id}")
        return self


class EvaluationConfig(StrictModel):
    task: Literal["swallow|aime_N4|0|0"]
    parser: Literal["llmjp4"]
    reasoning_effort: Literal["medium"]
    temperature: float
    top_p: float
    max_new_tokens: int
    max_model_len: int
    seed: int
    aime_2024: DatasetRef
    aime_2025: DatasetRef
    swallow_revision: Literal["v202604"]
    swallow_commit: Literal["e8fbf6210eeae8a9cd65ed1d58dd0e4f34239f71"]
    llm_jp_eval_revision: Literal["v2.1.4"]
    llm_jp_vllm_revision: Literal["v0.0.2"]
    vllm_version: Literal["0.18.0"]


class ResourcesConfig(StrictModel):
    stage_node_hours: dict[str, float]
    h200_node_hours_planned: float
    h200_node_hours_reserve: float
    h200_node_hours_absolute_max: float
    cpu_node_hours_max: float
    storage_bytes_max: int

    @model_validator(mode="after")
    def budget_is_bounded(self) -> ResourcesConfig:
        if self.h200_node_hours_planned + self.h200_node_hours_reserve != self.h200_node_hours_absolute_max:
            raise ValueError("planned + reserve must equal absolute max")
        if self.h200_node_hours_absolute_max > 16:
            raise ValueError("H200 budget exceeds approved 16 node-hours")
        if self.storage_bytes_max > 250_000_000_000:
            raise ValueError("storage budget exceeds approved 250 GB")
        expected_stages = {
            "generation_validation": 2.0,
            "difficulty": 1.5,
            "sft": 3.0,
            "grpo": 4.0,
            "aime": 3.5,
        }
        if self.stage_node_hours != expected_stages:
            raise ValueError(f"stage node-hour budget must be exactly {expected_stages}")
        if sum(self.stage_node_hours.values()) != self.h200_node_hours_planned:
            raise ValueError("stage node-hour budgets must sum to the 14-hour planned budget")
        return self


class TrackingConfig(StrictModel):
    wandb_entity: Literal["argo-lab"]
    wandb_project: str


class PublicationConfig(StrictModel):
    hf_namespace: Literal["argo11"]
    dataset_repo: str
    thinking_model_repo: str
    instruct_model_repo: str
    public_after_validation: Literal[True]
    license_metadata: None = None


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    identity: IdentityConfig
    paths: PathsConfig
    coordination: CoordinationConfig
    runtime: RuntimeConfig
    models: ModelsConfig
    data: DataConfig
    generation: GenerationConfig
    validation: ValidationConfig
    difficulty: DifficultyConfig
    sft: SftConfig
    grpo: GrpoConfig
    evaluation: EvaluationConfig
    resources: ResourcesConfig
    tracking: TrackingConfig
    publication: PublicationConfig

    def runtime_profile_for_stage(self, stage: StageName) -> RuntimeProfile:
        return self.runtime.profiles[self.runtime.stage_profile[stage]]

    def runtime_stage_for_manifest(self, stage: str) -> StageName:
        if stage == "fixture" or stage.startswith("fixture-"):
            return "audit"
        for candidate in self.runtime.stage_profile:
            if (
                stage == candidate
                or stage.startswith(f"{candidate}-")
                or stage.endswith(f"-{candidate}")
            ):
                return candidate
        raise ValueError(f"stage is not covered by runtime.stage_profile: {stage}")

    def require_assigned_id(self) -> str:
        value = self.identity.experiment_id
        if not EXPERIMENT_ID.fullmatch(value):
            raise ValueError("formal 4-digit experiment ID is required; UNASSIGNED is fixture-only")
        return value

    def require_production_contract(self) -> None:
        expected_models = {
            "generator_20b": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
            "generator_120b": ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"),
            "thinking": ("llm-jp/llm-jp-4-8b-thinking", "fc7c15c262710016c19bcda4372a96ff68875846"),
            "instruct": ("llm-jp/llm-jp-4-8b-instruct", "098f2b2cf33021eba19a6d3582aa3d071ccc0aff"),
        }
        failures = []
        if self.identity.upstream_revision != "229e52168ad3962d3215639e2d97603f28ad103d":
            failures.append("upstream revision drift")
        for name, (repo_id, revision) in expected_models.items():
            actual = getattr(self.models, name)
            if (actual.repo_id, actual.revision) != (repo_id, revision):
                failures.append(f"pinned model drift: {name}")
        expected = {
            "data": {
                "spec_count": 512,
                "split_counts": {"difficulty": 128, "sft": 256, "grpo": 128},
                "difficulty_min": 6,
                "difficulty_max": 10,
                "seed": 37,
            },
            "sft": {
                "max_length": 4096,
                "num_train_epochs": 1.0,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "learning_rate": 5e-6,
                "seed": 37,
                "packing": False,
                "assistant_only_loss": True,
            },
            "difficulty": {
                "pooled_delta_min": 0.10,
                "arm_delta_min": 0.0,
            },
            "grpo": {
                "max_steps": 32,
                "num_generations": 8,
                "max_prompt_length": 1024,
                "max_completion_length": 2048,
                "learning_rate": 5e-6,
                "loss_type": "dapo",
                "scale_rewards": False,
                "mask_truncated_completions": True,
                "use_vllm": False,
                "seed": 37,
            },
            "resources": {
                "stage_node_hours": {
                    "generation_validation": 2.0,
                    "difficulty": 1.5,
                    "sft": 3.0,
                    "grpo": 4.0,
                    "aime": 3.5,
                },
                "h200_node_hours_planned": 14.0,
                "h200_node_hours_reserve": 2.0,
                "h200_node_hours_absolute_max": 16.0,
                "cpu_node_hours_max": 4.0,
                "storage_bytes_max": 250_000_000_000,
            },
        }
        payload = self.model_dump(mode="json")
        for section, values in expected.items():
            for key, value in values.items():
                if payload[section][key] != value:
                    failures.append(f"production setting drift: {section}.{key}")
        expected_datasets = {
            "aime_2024": ("HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"),
            "aime_2025": ("yentinglin/aime_2025", "6f71d77b0b89b9dabe07ab466c51df33f514df7f"),
        }
        for name, (repo_id, revision) in expected_datasets.items():
            actual = getattr(self.evaluation, name)
            if (actual.repo_id, actual.revision, actual.split) != (repo_id, revision, "train"):
                failures.append(f"pinned evaluation dataset drift: {name}")
        normalized = self.model_dump(mode="json")
        normalized["identity"]["experiment_id"] = "UNASSIGNED"
        if sha256_value(normalized) != PRODUCTION_CONTRACT_SHA256:
            failures.append("unapproved production setting drift outside the explicit checks")
        if failures:
            raise ValueError("production contract failed: " + "; ".join(failures))

    def render(self, value: str, require_id: bool = True) -> str:
        experiment_id = self.require_assigned_id() if require_id else self.identity.experiment_id
        return value.replace("{ID}", experiment_id)

    def experiment_root(self, require_id: bool = True) -> Path:
        return Path(self.render(self.paths.experiment_root, require_id=require_id))


def load_config(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(payload)

from __future__ import annotations

import json
import math
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .config import ExperimentConfig
from .hashing import sha256_file
from .math_utils import extract_last_boxed, math_equivalent
from .preprocessing import (
    PREPROCESSING_MANIFEST,
    bind_pretokenized_grpo_prompts,
    load_verified_preprocessed_dataset,
)


def zero3_config(max_grad_norm: float) -> dict[str, Any]:
    return {
        "bf16": {"enabled": True},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": max_grad_norm,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_gather_16bit_weights_on_model_save": True,
        },
    }


class PretokenizedCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_length = max(len(item["input_ids"]) for item in features)

        def pad(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (max_length - len(values))

        return {
            "input_ids": torch.tensor([pad(item["input_ids"], self.pad_token_id) for item in features]),
            "attention_mask": torch.tensor([pad(item["attention_mask"], 0) for item in features]),
            "labels": torch.tensor([pad(item["labels"], -100) for item in features]),
        }


def run_sft(
    config: ExperimentConfig,
    arm: Literal["thinking", "instruct"],
    tokenized_dir: Path,
    output_dir: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    class FiniteCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "grad_norm"):
                value = (logs or {}).get(key)
                if value is not None and not math.isfinite(float(value)):
                    raise RuntimeError(f"non-finite SFT metric: {key}={value}")

    config.require_assigned_id()
    model_ref = getattr(config.models, arm)
    tokenizer = AutoTokenizer.from_pretrained(model_ref.repo_id, revision=model_ref.revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset, preprocessing = load_verified_preprocessed_dataset(
        config,
        kind="sft",
        arm=arm,
        dataset_dir=tokenized_dir,
        tokenizer=tokenizer,
        smoke=smoke,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_ref.repo_id,
        revision=model_ref.revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("full-parameter SFT requires every model parameter to be trainable")
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters != total_parameters:
        raise RuntimeError("full-parameter SFT trainable parameter count does not equal total parameter count")
    output_dir.mkdir(parents=True, exist_ok=False)
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.sft.num_train_epochs,
        max_steps=16 if smoke else -1,
        per_device_train_batch_size=config.sft.per_device_train_batch_size,
        gradient_accumulation_steps=config.sft.gradient_accumulation_steps,
        learning_rate=config.sft.learning_rate,
        warmup_ratio=config.sft.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=config.sft.weight_decay,
        adam_beta1=config.sft.adam_beta1,
        adam_beta2=config.sft.adam_beta2,
        max_grad_norm=config.sft.max_grad_norm,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=zero3_config(config.sft.max_grad_norm),
        save_strategy="no",
        logging_steps=1,
        report_to=["wandb"],
        run_name=f"{config.identity.experiment_id}-{arm}-sft{'-smoke' if smoke else ''}",
        seed=config.sft.seed,
        data_seed=config.sft.seed,
        remove_unused_columns=False,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=PretokenizedCollator(tokenizer.pad_token_id),
        callbacks=[FiniteCallback()],
    )
    result = trainer.train()
    expected_steps = 16 if smoke else math.ceil(
        len(dataset) / (config.sft.per_device_train_batch_size * config.sft.gradient_accumulation_steps * 8)
    )
    if trainer.state.global_step < expected_steps:
        raise RuntimeError(f"optimizer steps below gate: {trainer.state.global_step} < {expected_steps}")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    base_snapshot = Path(snapshot_download(
        repo_id=model_ref.repo_id,
        revision=model_ref.revision,
        allow_patterns=["*.safetensors", "*.safetensors.index.json"],
    ))
    from .weight_guard import compare_weight_update

    weight_update = compare_weight_update(base_snapshot, final_dir)
    if weight_update["changed_keys"] == 0:
        raise RuntimeError("SFT completed without a proven weight update")
    metrics = {
        **result.metrics,
        "global_step": trainer.state.global_step,
        "expected_steps": expected_steps,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "weight_update": weight_update,
        "preprocessing_manifest_sha256": sha256_file(tokenized_dir / PREPROCESSING_MANIFEST),
        "preprocessing": preprocessing.model_dump(mode="json"),
    }
    (output_dir / "training-metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


@dataclass
class RewardHealth:
    parsed: int = 0
    total: int = 0

    @property
    def parse_rate(self) -> float:
        return self.parsed / self.total if self.total else 0.0


REWARD_HEALTH = RewardHealth()


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def math_accuracy_reward(completions, expected_answer=None, **kwargs) -> list[float]:
    references = expected_answer or kwargs.get("answer")
    if references is None:
        raise ValueError("expected_answer column is required by the outcome reward")
    rewards = []
    for completion, reference in zip(completions, references, strict=True):
        parsed = extract_last_boxed(_completion_text(completion))
        REWARD_HEALTH.total += 1
        if parsed is not None:
            REWARD_HEALTH.parsed += 1
        rewards.append(float(bool(parsed and math_equivalent(parsed, str(reference), strict=True))))
    return rewards


class GrpoHealthCallback:
    def __new__(cls, config: ExperimentConfig, failure_path: Path):
        from transformers import TrainerCallback

        class Callback(TrainerCallback):
            def __init__(self) -> None:
                self.zero_std = deque(maxlen=10)

            def on_train_begin(self, args, state, control, **kwargs):
                REWARD_HEALTH.parsed = 0
                REWARD_HEALTH.total = 0

            def fail(self, reason: str, logs: dict) -> None:
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(
                    json.dumps({"reason": reason, "logs": logs}, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
                raise RuntimeError(reason)

            def on_log(self, args, state, control, logs=None, **kwargs):
                logs = dict(logs or {})
                for key, value in logs.items():
                    if isinstance(value, (float, int)) and not math.isfinite(float(value)):
                        self.fail(f"non-finite GRPO metric: {key}={value}", logs)
                zero_std = logs.get("frac_reward_zero_std")
                clipped = logs.get("completions/clipped_ratio")
                if zero_std is not None:
                    self.zero_std.append(float(zero_std))
                if state.global_step >= 2:
                    if REWARD_HEALTH.parse_rate < config.grpo.parser_success_min:
                        self.fail(f"parser success gate failed: {REWARD_HEALTH.parse_rate:.4f}", logs)
                    if zero_std is None or clipped is None:
                        self.fail("required GRPO health metrics were not logged", logs)
                    if 1.0 - float(zero_std) < config.grpo.informative_group_min:
                        self.fail(f"informative group gate failed: {1.0 - float(zero_std):.4f}", logs)
                    if float(clipped) > config.grpo.truncation_rate_max:
                        self.fail(f"completion truncation gate failed: {float(clipped):.4f}", logs)
                if len(self.zero_std) == 10 and sum(self.zero_std) / 10 > 0.5:
                    self.fail("rolling zero-variance group rate exceeded 0.5", logs)

        return Callback()


def run_grpo(
    config: ExperimentConfig,
    arm: Literal["thinking", "instruct"],
    sft_model_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    config.require_assigned_id()
    tokenizer = AutoTokenizer.from_pretrained(sft_model_dir, trust_remote_code=False)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("SFT tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    dataset, preprocessing = load_verified_preprocessed_dataset(
        config,
        kind="grpo",
        arm=arm,
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        smoke=smoke,
    )
    bind_pretokenized_grpo_prompts(tokenizer, dataset)
    output_dir.mkdir(parents=True, exist_ok=False)
    grpo = config.grpo
    args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=2 if smoke else grpo.max_steps,
        per_device_train_batch_size=grpo.per_device_train_batch_size,
        gradient_accumulation_steps=grpo.gradient_accumulation_steps,
        num_generations=grpo.num_generations,
        max_prompt_length=grpo.max_prompt_length,
        max_completion_length=grpo.max_completion_length,
        temperature=grpo.temperature,
        learning_rate=grpo.learning_rate,
        warmup_steps=grpo.warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=grpo.weight_decay,
        max_grad_norm=grpo.max_grad_norm,
        beta=grpo.beta,
        epsilon=grpo.epsilon,
        epsilon_high=grpo.epsilon_high,
        loss_type=grpo.loss_type,
        scale_rewards=grpo.scale_rewards,
        mask_truncated_completions=grpo.mask_truncated_completions,
        use_vllm=grpo.use_vllm,
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="no",
        logging_steps=1,
        report_to=["wandb"],
        run_name=f"{config.identity.experiment_id}-{arm}-grpo{'-smoke' if smoke else ''}",
        seed=grpo.seed,
        data_seed=grpo.seed,
    )
    peft_config = LoraConfig(
        r=grpo.lora_rank,
        lora_alpha=grpo.lora_alpha,
        lora_dropout=grpo.lora_dropout,
        target_modules=grpo.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=str(sft_model_dir),
        reward_funcs=[math_accuracy_reward],
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=[GrpoHealthCallback(config, output_dir / "gate-failure.json")],
        processing_class=tokenizer,
    )
    initial_adapter_dir = output_dir / "adapter-initial"
    trainer.save_model(str(initial_adapter_dir))
    trainer.accelerator.wait_for_everyone()
    result = trainer.train()
    expected_steps = 2 if smoke else grpo.max_steps
    if trainer.state.global_step != expected_steps:
        raise RuntimeError(f"GRPO optimizer step mismatch: {trainer.state.global_step} != {expected_steps}")
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    trainer.accelerator.wait_for_everyone()
    from .weight_guard import compare_adapter_update

    adapter_weight_update = compare_adapter_update(initial_adapter_dir, adapter_dir)
    adapter_weight_update_path = adapter_dir / "tv-gptoss120b-weight-update.json"
    adapter_weight_update_path.write_text(
        json.dumps(adapter_weight_update, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copytree(initial_adapter_dir, adapter_dir / "initial-snapshot")
    metrics = {
        **result.metrics,
        "global_step": trainer.state.global_step,
        "expected_steps": expected_steps,
        "parser_success": REWARD_HEALTH.parse_rate,
        "adapter_weight_update": adapter_weight_update,
        "preprocessing_manifest_sha256": sha256_file(dataset_dir / PREPROCESSING_MANIFEST),
        "preprocessing": preprocessing.model_dump(mode="json"),
    }
    (output_dir / "training-metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def merge_adapter(sft_model_dir: Path, adapter_dir: Path, output_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if output_dir.exists():
        raise FileExistsError(f"merged model output is immutable: {output_dir}")
    base = AutoModelForCausalLM.from_pretrained(sft_model_dir, torch_dtype=torch.bfloat16, trust_remote_code=False)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(sft_model_dir, trust_remote_code=False).save_pretrained(output_dir)


def verify_merge_parity(
    sft_model_dir: Path,
    adapter_dir: Path,
    merged_dir: Path,
    prompts: list[str],
    report_path: Path,
    tolerance: float = 5e-3,
) -> dict[str, Any]:
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if len(prompts) != 8:
        raise ValueError("merge parity requires exactly 8 fixed prompts")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_dir, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    def evaluate(model) -> tuple[list, list[list[int]]]:
        model.eval().cuda()
        batch = {key: value.cuda() for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, :].float().cpu()
            generated = model.generate(**batch, do_sample=False, max_new_tokens=32).cpu().tolist()
        return logits, generated

    base = AutoModelForCausalLM.from_pretrained(sft_model_dir, torch_dtype=torch.bfloat16, trust_remote_code=False)
    adapter_model = PeftModel.from_pretrained(base, adapter_dir)
    adapter_logits, adapter_generated = evaluate(adapter_model)
    del adapter_model, base
    gc.collect()
    torch.cuda.empty_cache()

    merged = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch.bfloat16, trust_remote_code=False)
    merged_logits, merged_generated = evaluate(merged)
    max_abs_diff = float((adapter_logits - merged_logits).abs().max().item())
    greedy_match = adapter_generated == merged_generated
    report = {
        "prompt_count": len(prompts),
        "max_abs_logit_diff": max_abs_diff,
        "tolerance": tolerance,
        "greedy_outputs_match": greedy_match,
    }
    if max_abs_diff > tolerance or not greedy_match:
        raise RuntimeError(f"adapter/merged parity gate failed: {report}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report

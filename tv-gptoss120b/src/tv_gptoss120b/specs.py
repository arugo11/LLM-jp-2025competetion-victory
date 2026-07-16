from __future__ import annotations

from pathlib import Path

from .config import DataConfig
from .hashing import sha256_value
from .jsonl import read_jsonl
from .schema import ProblemSpec

CATEGORY_UNITS = [
    ("中1", "一次方程式"), ("中1", "文字式"), ("中1", "正負の数"), ("中1", "比例反比例"),
    ("中2", "一次関数"), ("中2", "文字式"), ("中2", "確率"), ("中2", "連立方程式"),
    ("中3", "二次方程式"), ("中3", "二次関数"), ("中3", "展開と因数分解"), ("中3", "平方根"),
    ("IA", "2次関数"), ("IA", "場合の数と確率"), ("IA", "数と式"),
    ("IA", "整数の性質（数学と人間活動）"), ("IIB", "いろいろな式"), ("IIB", "三角関数"),
    ("IIB", "微分法・積分法"), ("IIB", "指数・対数"), ("IIB", "数列"), ("IIB", "統計的な推測"),
    ("IIIC", "微分"), ("IIIC", "極限"), ("IIIC", "積分"), ("IIIC", "ベクトル"),
    ("IIIC", "平面上の曲線と複素数平面"),
]


def build_specs(config: DataConfig) -> list[ProblemSpec]:
    base: list[dict] = []
    difficulty_span = config.difficulty_max - config.difficulty_min + 1
    for index in range(config.spec_count):
        category, unit = CATEGORY_UNITS[index % len(CATEGORY_UNITS)]
        difficulty = config.difficulty_min + ((index // len(CATEGORY_UNITS)) % difficulty_span)
        spec_id = f"spec-{index:04d}"
        seed_digest = sha256_value({"seed": config.seed, "spec_id": spec_id})
        generation_seed = int(seed_digest[:8], 16)
        base.append({
            "spec_id": spec_id,
            "category": category,
            "unit": unit,
            "declared_difficulty": difficulty,
            "generation_seed": generation_seed,
        })

    ordered_ids = sorted((item["spec_id"] for item in base), key=lambda value: sha256_value({"split": value}))
    d_end = config.split_counts.difficulty
    s_end = d_end + config.split_counts.sft
    split_by_id = {
        **{value: "difficulty" for value in ordered_ids[:d_end]},
        **{value: "sft" for value in ordered_ids[d_end:s_end]},
        **{value: "grpo" for value in ordered_ids[s_end:]},
    }

    result = []
    for item in base:
        payload = {**item, "split": split_by_id[item["spec_id"]]}
        result.append(ProblemSpec(**payload, spec_sha256=sha256_value(payload)))
    return result


def load_verified_specs(config: DataConfig, path: Path) -> list[ProblemSpec]:
    """Load the immutable generation plan and reject any config or file drift."""
    actual = read_jsonl(path, ProblemSpec)
    expected = build_specs(config)
    if actual != expected:
        actual_by_id = {item.spec_id: item for item in actual}
        expected_by_id = {item.spec_id: item for item in expected}
        duplicate_count = len(actual) - len(actual_by_id)
        missing = sorted(expected_by_id.keys() - actual_by_id.keys())
        extra = sorted(actual_by_id.keys() - expected_by_id.keys())
        changed = sorted(
            spec_id
            for spec_id in expected_by_id.keys() & actual_by_id.keys()
            if actual_by_id[spec_id] != expected_by_id[spec_id]
        )
        raise ValueError(
            "generation spec does not match the canonical config-derived plan: "
            f"rows={len(actual)}, duplicates={duplicate_count}, "
            f"missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}"
        )
    return actual

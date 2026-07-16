from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .jsonl import write_jsonl
from .math_utils import extract_last_boxed, math_equivalent, normalize_text
from .statistics import holm_adjust


def normalize_detail_rows(
    label: str,
    year: int,
    rows: list[dict[str, Any]],
    *,
    expected_problem_count: int = 30,
    strict_math_verify: bool = True,
) -> list[dict[str, Any]]:
    if year not in {2024, 2025}:
        raise ValueError(f"unsupported AIME year: {year}")
    if len(rows) != expected_problem_count:
        raise ValueError(
            f"AIME{year} details must contain exactly {expected_problem_count} problem rows, got {len(rows)}"
        )
    normalized = []
    for problem_index, row in enumerate(rows, start=1):
        predictions = list(row.get("predictions") or [])
        golds = list(row.get("gold") or [])
        if len(predictions) != 4 or not golds:
            raise ValueError(f"AIME{year} problem {problem_index} requires four predictions and at least one gold")
        continuation_tokens = list(row.get("cont_tokens") or [[] for _ in predictions])
        truncated = list(row.get("truncated") or [0 for _ in predictions])
        if not (len(continuation_tokens) == len(truncated) == len(predictions)):
            raise ValueError(f"AIME{year} problem {problem_index} detail arrays disagree")
        expected = str(golds[0])
        for sample_index, prediction in enumerate(predictions):
            text = str(prediction)
            parsed = extract_last_boxed(text)
            normalized.append({
                "model": label,
                "year": year,
                "problem_id": str(problem_index),
                "sample_index": sample_index,
                "raw_response": text,
                "parsed_answer": parsed,
                "expected_answer": expected,
                "parsed": parsed is not None,
                "correct": bool(parsed and math_equivalent(parsed, expected, strict=strict_math_verify)),
                "truncated": bool(truncated[sample_index]),
                "completion_tokens": len(continuation_tokens[sample_index]),
                "full_prompt": row.get("full_prompt", ""),
            })
    return normalized


def normalize_lighteval_details(
    label: str,
    detail_paths: list[Path],
    output_path: Path,
    *,
    expected_problem_count: int = 30,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from datasets import Dataset

    by_year: dict[int, list[dict[str, Any]]] = {}
    for path in detail_paths:
        match = re.search(r"aime[_|:-]?n4[:_|-]?(24|25)(?:[|_.-]|$)", path.name.lower())
        if match is None:
            raise ValueError(f"cannot identify one AIME year from detail filename: {path}")
        year = 2000 + int(match.group(1))
        if year in by_year:
            raise ValueError(f"multiple detail files found for AIME{year}")
        by_year[year] = list(Dataset.from_parquet(str(path)))
    if set(by_year) != {2024, 2025}:
        raise ValueError(f"both AIME2024 and AIME2025 detail files are required, found {sorted(by_year)}")
    records = [
        {**row, **(metadata or {})}
        for year in (2024, 2025)
        for row in normalize_detail_rows(
            label,
            year,
            by_year[year],
            expected_problem_count=expected_problem_count,
        )
    ]
    write_jsonl(output_path, records)
    return records


def verify_matched_fingerprint(records: list[dict[str, Any]], expected_fingerprint: str) -> None:
    fingerprints = {row.get("matched_config_sha256") for row in records}
    if fingerprints != {expected_fingerprint}:
        raise ValueError(
            f"base/post evaluation settings are not matched: expected={expected_fingerprint}, found={fingerprints}"
        )
    indexed = {
        (str(row.get("model")), int(row["year"]), str(row["problem_id"]), int(row["sample_index"])): row
        for row in records
        if row.get("model") and all(key in row for key in ("year", "problem_id", "sample_index"))
    }
    for arm in ("thinking", "instruct"):
        base_keys = {key[1:] for key in indexed if key[0] == f"{arm}-base"}
        post_keys = {key[1:] for key in indexed if key[0] == f"{arm}-post"}
        if base_keys or post_keys:
            if base_keys != post_keys:
                raise ValueError(f"base/post evaluation records are not paired for {arm}")
            for key in base_keys:
                base = indexed[(f"{arm}-base", *key)]
                post = indexed[(f"{arm}-post", *key)]
                for field in ("full_prompt", "expected_answer"):
                    if base.get(field) != post.get(field):
                        raise ValueError(f"base/post evaluation {field} mismatch for {arm}, key={key}")


def _maj_at_4(rows: list[dict[str, Any]]) -> float:
    normalized = [normalize_text(str(row.get("parsed_answer") or "EMPTY")) for row in rows]
    counts = Counter(normalized)
    highest = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == highest]
    correct_winners = 0
    for answer in winners:
        representative = next(
            row
            for row, normalized_answer in zip(rows, normalized, strict=True)
            if normalized_answer == answer
        )
        correct_winners += int(bool(representative["correct"]))
    return correct_winners / len(winners)


def summarize_aime(records: list[dict[str, Any]], *, bootstrap_samples: int = 10_000, seed: int = 37) -> dict:
    required_models = {"thinking-base", "thinking-post", "instruct-base", "instruct-post"}
    keys = {(str(row["model"]), int(row["year"]), str(row["problem_id"]), int(row["sample_index"])) for row in records}
    expected = {
        (model, year, str(problem), sample)
        for model in required_models
        for year in (2024, 2025)
        for problem in range(1, 31)
        for sample in range(4)
    }
    if keys != expected or len(records) != len(expected):
        missing = len(expected - keys)
        extra = len(keys - expected)
        duplicates = len(records) - len(keys)
        raise ValueError(
            f"AIME evaluation must contain the matched 960 records; "
            f"missing={missing}, extra={extra}, duplicates={duplicates}"
        )

    by_problem: dict[tuple[str, int, str], list[bool]] = defaultdict(list)
    parse_by_model: dict[str, list[bool]] = defaultdict(list)
    for row in records:
        by_problem[(str(row["model"]), int(row["year"]), str(row["problem_id"]))].append(bool(row["correct"]))
        parse_by_model[str(row["model"])].append(bool(row["parsed"]))

    def problem_scores(model: str) -> list[float]:
        return [
            float(np.mean(by_problem[(model, year, str(problem))]))
            for year in (2024, 2025)
            for problem in range(1, 31)
        ]

    pass1 = {model: float(np.mean(problem_scores(model))) for model in required_models}
    secondary = {}
    for model in required_models:
        model_summary = {}
        for year in (2024, 2025):
            problems = [by_problem[(model, year, str(problem))] for problem in range(1, 31)]
            detail_problems = [
                sorted(
                    [
                        row for row in records
                        if row["model"] == model and int(row["year"]) == year and str(row["problem_id"]) == str(problem)
                    ],
                    key=lambda row: int(row["sample_index"]),
                )
                for problem in range(1, 31)
            ]
            model_summary[str(year)] = {
                "pass_at_1_4": float(np.mean([np.mean(values) for values in problems])),
                "pass_at_4_4": float(np.mean([any(values) for values in problems])),
                "maj_at_4_4": float(np.mean([_maj_at_4(values) for values in detail_problems])),
            }
        secondary[model] = model_summary
    rng = np.random.default_rng(seed)
    arm_summaries: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float] = {}
    for arm in ("thinking", "instruct"):
        deltas = []
        base = np.array(problem_scores(f"{arm}-base"))
        post = np.array(problem_scores(f"{arm}-post"))
        for _ in range(bootstrap_samples):
            indices = rng.integers(0, 60, size=60)
            deltas.append(float(np.mean(post[indices] - base[indices])))
        observed = float(np.mean(post - base))
        p_value = min(1.0, 2 * min(np.mean(np.array(deltas) <= 0), np.mean(np.array(deltas) >= 0)))
        raw_p[arm] = float(p_value)
        arm_summaries[arm] = {
            "delta_pass_at_1_4": observed,
            "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
            "p_value_exploratory": float(p_value),
            "improved": observed > 0,
        }
    arm_order = ["thinking", "instruct"]
    adjusted = holm_adjust([raw_p[arm] for arm in arm_order])
    for arm, value in zip(arm_order, adjusted, strict=True):
        arm_summaries[arm]["p_value_holm"] = value
    return {
        "primary_pass_at_1_4": pass1,
        "parse_rate": {model: float(np.mean(values)) for model, values in parse_by_model.items()},
        "secondary_by_year": secondary,
        "truncation_rate": {
            model: float(np.mean([bool(row.get("truncated", False)) for row in records if row["model"] == model]))
            for model in required_models
        },
        "mean_completion_tokens": {
            model: float(np.mean([int(row.get("completion_tokens", 0)) for row in records if row["model"] == model]))
            for model in required_models
        },
        "arms": arm_summaries,
        "scientific_aime_gate": any(summary["improved"] for summary in arm_summaries.values()),
    }

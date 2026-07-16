from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .aime_reference import load_verified_aime_reference
from .config import ExperimentConfig
from .hashing import sha256_file, sha256_value
from .jsonl import read_jsonl, write_jsonl
from .math_utils import jaccard, minhash_candidate_pairs, normalize_text, token_ngrams
from .schema import CuratedRecord, ProblemSpec, RawGeneration, ValidationVote
from .specs import load_verified_specs


class ReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec_id: str
    reason: str
    generator_model: str | None = None
    reference_id: str | None = None
    similarity: float | None = None
    problem: str | None = None


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec_id: str
    decision: Literal["accept", "reject"]
    critical_defect: bool
    reviewer: str
    reviewed_at: str
    notes: str


def _vote_index(votes: list[ValidationVote]) -> dict[tuple[str, str, str], ValidationVote]:
    return {(vote.spec_id, vote.generator_model, vote.validator_model): vote for vote in votes}


def _verify_raw_against_specs(
    config: ExperimentConfig,
    generation_spec_path: Path,
    rows: list[RawGeneration],
) -> dict[str, ProblemSpec]:
    specs = load_verified_specs(config.data, generation_spec_path)
    by_id = {item.spec_id: item for item in specs}
    expected_models = {
        "generator_20b": config.models.generator_20b,
        "generator_120b": config.models.generator_120b,
    }
    spec_fields = ("category", "unit", "declared_difficulty", "split", "generation_seed")
    for row in rows:
        spec = by_id.get(row.spec_id)
        if spec is None:
            raise ValueError(f"raw generation references unknown spec_id: {row.spec_id}")
        drift = [field for field in spec_fields if getattr(row, field) != getattr(spec, field)]
        if drift:
            raise ValueError(f"raw generation/spec mismatch for {row.spec_id}: {drift}")
        expected_model = expected_models[row.generator_model]
        if (row.generator_repo_id, row.generator_revision) != (
            expected_model.repo_id,
            expected_model.revision,
        ):
            raise ValueError(f"generator model pin mismatch for {row.spec_id}:{row.generator_model}")
        payload = row.model_dump(exclude={"content_sha256"})
        if sha256_value(payload) != row.content_sha256:
            raise ValueError(f"raw generation content hash mismatch for {row.spec_id}:{row.generator_model}")
    expected_keys = {
        (spec.spec_id, generator)
        for spec in specs
        for generator in ("generator_20b", "generator_120b")
    }
    actual_keys = [(row.spec_id, row.generator_model) for row in rows]
    if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
        raise ValueError(
            "raw generation set must contain exactly one row per spec and generator: "
            f"rows={len(actual_keys)}, unique={len(set(actual_keys))}, expected={len(expected_keys)}"
        )
    return by_id


def _verify_votes(config: ExperimentConfig, votes: list[ValidationVote]) -> None:
    expected_revisions = {
        "generator_20b": config.models.generator_20b.revision,
        "generator_120b": config.models.generator_120b.revision,
    }
    for vote in votes:
        if vote.validator_revision != expected_revisions[vote.validator_model]:
            raise ValueError(f"validator model pin mismatch for {vote.spec_id}:{vote.validator_model}")
        payload = vote.model_dump(exclude={"content_sha256"})
        if sha256_value(payload) != vote.content_sha256:
            raise ValueError(
                f"validation vote content hash mismatch for "
                f"{vote.spec_id}:{vote.generator_model}:{vote.validator_model}"
            )


def verify_generation_inputs(
    config: ExperimentConfig,
    generation_spec_path: Path,
    raw_paths: list[Path],
) -> dict[str, Any]:
    raw_rows = [row for path in raw_paths for row in read_jsonl(path, RawGeneration)]
    _verify_raw_against_specs(config, generation_spec_path, raw_rows)
    return {
        "generation_spec_sha256": sha256_file(generation_spec_path),
        "raw_generation_files": [
            {"name": path.name, "sha256": sha256_file(path)} for path in raw_paths
        ],
    }


def _similarity_flags(
    rows: list[RawGeneration], references: list[dict[str, Any]], ngram_size: int, threshold: float
) -> list[ReviewItem]:
    flags = []
    reference_grams = [
        (str(item.get("id", index)), str(item["problem"]), token_ngrams(str(item["problem"]), ngram_size))
        for index, item in enumerate(references)
    ]
    row_by_key = {f"row:{index}": row for index, row in enumerate(rows)}
    ref_by_key = {f"ref:{index}": item for index, item in enumerate(reference_grams)}
    grams_by_key = {
        **{key: token_ngrams(row.problem or "", ngram_size) for key, row in row_by_key.items()},
        **{key: item[2] for key, item in ref_by_key.items()},
    }
    candidates = minhash_candidate_pairs(grams_by_key, threshold)
    candidate_cross = {
        (left, right) if left.startswith("row:") else (right, left)
        for left, right in candidates
        if left.startswith("row:") != right.startswith("row:")
    }
    candidate_cross |= {
        (row_key, ref_key)
        for row_key in row_by_key
        for ref_key in ref_by_key
        if jaccard(grams_by_key[row_key], grams_by_key[ref_key]) >= threshold
    }
    exact_cross = {
        (row_key, ref_key)
        for row_key, row in row_by_key.items()
        for ref_key, (_, reference_problem, _) in ref_by_key.items()
        if normalize_text(row.problem or "") == normalize_text(reference_problem)
    }
    for row_key, ref_key in sorted(candidate_cross | exact_cross):
        row = row_by_key[row_key]
        reference_id, reference_problem, other_grams = ref_by_key[ref_key]
        if not row.problem:
            continue
        normalized = normalize_text(row.problem)
        grams = token_ngrams(row.problem, ngram_size)
        similarity = jaccard(grams, other_grams)
        if normalized == normalize_text(reference_problem) or similarity >= threshold:
            flags.append(ReviewItem(
                spec_id=row.spec_id,
                reason="aime_similarity",
                generator_model=row.generator_model,
                reference_id=reference_id,
                similarity=similarity,
                problem=row.problem,
            ))
    return flags


def _internal_duplicate_flags(rows: list[RawGeneration], ngram_size: int, threshold: float) -> list[ReviewItem]:
    prepared = [
        (row, normalize_text(row.problem or ""), token_ngrams(row.problem or "", ngram_size)) for row in rows
    ]
    flags: list[ReviewItem] = []
    seen: set[tuple[str, str]] = set()
    by_key = {f"row:{index}": value for index, value in enumerate(prepared)}
    candidates = minhash_candidate_pairs({key: value[2] for key, value in by_key.items()}, threshold)
    normalized_groups: dict[str, list[str]] = defaultdict(list)
    for key, (_, normalized, _) in by_key.items():
        normalized_groups[normalized].append(key)
    for group in normalized_groups.values():
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                candidates.add(tuple(sorted((left, right))))
    keys = sorted(by_key)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            if jaccard(by_key[left][2], by_key[right][2]) >= threshold:
                candidates.add((left, right))
    for left_key, right_key in sorted(candidates):
        left, left_normalized, left_grams = by_key[left_key]
        right, right_normalized, right_grams = by_key[right_key]
        similarity = jaccard(left_grams, right_grams)
        if left_normalized == right_normalized or similarity >= threshold:
            pair = tuple(sorted((f"{left.spec_id}:{left.generator_model}", f"{right.spec_id}:{right.generator_model}")))
            if pair in seen:
                continue
            seen.add(pair)
            flags.append(ReviewItem(
                spec_id=left.spec_id,
                reason="internal_near_duplicate",
                generator_model=left.generator_model,
                reference_id=right.spec_id,
                similarity=similarity,
                problem=left.problem,
            ))
            flags.append(ReviewItem(
                spec_id=right.spec_id,
                reason="internal_near_duplicate",
                generator_model=right.generator_model,
                reference_id=left.spec_id,
                similarity=similarity,
                problem=right.problem,
            ))
    return flags


def _balanced_manual_sample(rows: list[RawGeneration], paired_ids: list[str], count: int) -> list[str]:
    category_by_spec = {row.spec_id: row.category for row in rows}
    grouped: dict[str, list[str]] = defaultdict(list)
    for spec_id in paired_ids:
        grouped[category_by_spec[spec_id]].append(spec_id)
    for category in grouped:
        grouped[category].sort(key=lambda value: sha256_value({"manual_review": value}))
    category_order = sorted(grouped, key=lambda value: sha256_value({"manual_review_category": value}))
    selected = []
    round_index = 0
    while len(selected) < min(count, len(paired_ids)):
        added = False
        for category in category_order:
            if round_index < len(grouped[category]):
                selected.append(grouped[category][round_index])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        round_index += 1
    return selected


def prepare_review(
    config: ExperimentConfig,
    generation_spec_path: Path,
    raw_paths: list[Path],
    vote_paths: list[Path],
    aime_reference_path: Path,
    review_path: Path,
) -> dict[str, Any]:
    raw_rows = [row for path in raw_paths for row in read_jsonl(path, RawGeneration)]
    _verify_raw_against_specs(config, generation_spec_path, raw_rows)
    votes = [row for path in vote_paths for row in read_jsonl(path, ValidationVote)]
    _verify_votes(config, votes)
    references = [
        record.model_dump(mode="json")
        for record in load_verified_aime_reference(config, aime_reference_path)
    ]
    vote_index = _vote_index(votes)
    accepted_candidates = []
    for row in raw_rows:
        if row.parse_status != "ok":
            continue
        keys = [(row.spec_id, row.generator_model, validator) for validator in ("generator_20b", "generator_120b")]
        if all(key in vote_index and vote_index[key].equivalent_to_reference for key in keys):
            accepted_candidates.append(row)

    flags = _similarity_flags(
        accepted_candidates,
        references,
        config.validation.ngram_size,
        config.validation.similarity_flag_threshold,
    )
    flags.extend(_internal_duplicate_flags(
        accepted_candidates,
        config.validation.ngram_size,
        config.validation.internal_similarity_threshold,
    ))
    paired_ids = sorted(
        spec_id for spec_id, count in Counter(row.spec_id for row in accepted_candidates).items() if count == 2
    )
    sampled = _balanced_manual_sample(accepted_candidates, paired_ids, config.validation.manual_review_pairs)
    sampled_set = set(sampled)
    flags.extend(
        ReviewItem(
            spec_id=row.spec_id,
            reason="manual_quality_sample",
            generator_model=row.generator_model,
            problem=row.problem,
        )
        for row in accepted_candidates
        if row.spec_id in sampled_set
    )
    unique = {(item.spec_id, item.reason, item.generator_model, item.reference_id): item for item in flags}
    write_jsonl(review_path, unique.values())
    return {
        "candidate_rows": len(accepted_candidates),
        "paired_ids": len(paired_ids),
        "review_items": len(unique),
        "generation_spec_sha256": sha256_file(generation_spec_path),
        "raw_generation_files": [
            {"name": path.name, "sha256": sha256_file(path)} for path in raw_paths
        ],
        "validation_vote_files": [
            {"name": path.name, "sha256": sha256_file(path)} for path in vote_paths
        ],
        "aime_reference_sha256": sha256_file(aime_reference_path),
    }


def finalize_curation(
    config: ExperimentConfig,
    generation_spec_path: Path,
    raw_paths: list[Path],
    vote_paths: list[Path],
    review_path: Path,
    decisions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    raw_rows = [row for path in raw_paths for row in read_jsonl(path, RawGeneration)]
    _verify_raw_against_specs(config, generation_spec_path, raw_rows)
    votes = [row for path in vote_paths for row in read_jsonl(path, ValidationVote)]
    _verify_votes(config, votes)
    review_items = read_jsonl(review_path, ReviewItem)
    decisions = read_jsonl(decisions_path, ReviewDecision)
    decision_by_spec = {item.spec_id: item for item in decisions}
    required_specs = {item.spec_id for item in review_items}
    missing = required_specs - decision_by_spec.keys()
    if missing:
        raise ValueError(f"manual review decisions missing for {len(missing)} specs")
    if any(decision_by_spec[spec_id].critical_defect for spec_id in required_specs):
        raise ValueError("critical defect found in required manual review; quality gate is closed")
    rejected_specs = {spec_id for spec_id in required_specs if decision_by_spec[spec_id].decision != "accept"}

    vote_index = _vote_index(votes)
    accepted_by_spec: dict[str, list[RawGeneration]] = defaultdict(list)
    valid_by_generator = Counter()
    parse_by_generator = Counter(row.generator_model for row in raw_rows if row.parse_status == "ok")
    total_by_generator = Counter(row.generator_model for row in raw_rows)
    expected_spec_ids = {f"spec-{index:04d}" for index in range(config.data.spec_count)}
    raw_spec_ids = {
        generator: [row.spec_id for row in raw_rows if row.generator_model == generator]
        for generator in ("generator_20b", "generator_120b")
    }
    for row in raw_rows:
        keys = [(row.spec_id, row.generator_model, validator) for validator in ("generator_20b", "generator_120b")]
        if row.parse_status == "ok" and all(
            key in vote_index and vote_index[key].equivalent_to_reference for key in keys
        ):
            valid_by_generator[row.generator_model] += 1
            if row.spec_id not in rejected_specs:
                accepted_by_spec[row.spec_id].append(row)

    complete_pairs = {spec_id: rows for spec_id, rows in accepted_by_spec.items() if len(rows) == 2}
    curated = []
    for spec_id, rows in complete_pairs.items():
        for row in rows:
            assert row.problem and row.solution_cot and row.expected_answer
            validation_votes = {
                validator: vote_index[(spec_id, row.generator_model, validator)].equivalent_to_reference
                for validator in ("generator_20b", "generator_120b")
            }
            payload = {
                "spec_id": spec_id,
                "generator_model": row.generator_model,
                "generator_revision": row.generator_revision,
                "category": row.category,
                "unit": row.unit,
                "declared_difficulty": row.declared_difficulty,
                "problem": row.problem,
                "solution_cot": row.solution_cot,
                "expected_answer": row.expected_answer,
                "generation_seed": row.generation_seed,
                "prompt_revision": row.prompt_revision,
                "validation_status": "accepted",
                "validation_votes": validation_votes,
                "split": row.split,
            }
            curated.append(CuratedRecord(**payload, content_sha256=sha256_value(payload)))

    parse_rates = {key: parse_by_generator[key] / total_by_generator[key] for key in total_by_generator}
    validity_rates = {key: valid_by_generator[key] / total_by_generator[key] for key in total_by_generator}
    pair_counts = Counter(rows[0].split for rows in complete_pairs.values())
    failures = []
    for generator, spec_ids in raw_spec_ids.items():
        if len(spec_ids) != config.data.spec_count or set(spec_ids) != expected_spec_ids:
            failures.append(
                f"raw_generation_incomplete:{generator}=rows:{len(spec_ids)},unique:{len(set(spec_ids))}"
            )
    expected_vote_keys = {
        (spec_id, generator, validator)
        for spec_id in expected_spec_ids
        for generator in ("generator_20b", "generator_120b")
        for validator in ("generator_20b", "generator_120b")
    }
    actual_vote_keys = {(vote.spec_id, vote.generator_model, vote.validator_model) for vote in votes}
    if len(votes) != len(expected_vote_keys) or actual_vote_keys != expected_vote_keys:
        failures.append(f"symmetric_validation_incomplete:rows:{len(votes)},unique:{len(actual_vote_keys)}")
    for key, rate in parse_rates.items():
        if rate < config.validation.parse_rate_min:
            failures.append(f"parse_rate:{key}={rate:.4f}")
    if validity_rates["generator_120b"] < (
        validity_rates["generator_20b"] - config.validation.validity_noninferiority_margin
    ):
        failures.append("generator_120b validity noninferiority gate failed")
    for split in ("difficulty", "sft", "grpo"):
        required = getattr(config.validation.valid_pair_min, split)
        if pair_counts[split] < required:
            failures.append(f"valid_pairs:{split}={pair_counts[split]}<{required}")
    if failures:
        raise ValueError("quality gate failed: " + "; ".join(failures))
    write_jsonl(output_path, sorted(curated, key=lambda row: (row.spec_id, row.generator_model)))
    return {
        "raw_generation_counts": dict(total_by_generator),
        "validation_vote_count": len(votes),
        "parse_rates": parse_rates,
        "validity_rates": validity_rates,
        "valid_pair_counts": dict(pair_counts),
        "dataset_sha256": sha256_value([row.content_sha256 for row in curated]),
    }

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tv_gptoss120b.aime_reference import AimeReferenceRecord
from tv_gptoss120b.config import ExperimentConfig, load_config
from tv_gptoss120b.curation import ReviewItem, finalize_curation, prepare_review
from tv_gptoss120b.generation import generate_problems, validate_problems
from tv_gptoss120b.hashing import sha256_value
from tv_gptoss120b.jsonl import read_jsonl, write_jsonl
from tv_gptoss120b.specs import build_specs

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def fixture_config() -> ExperimentConfig:
    payload = load_config(CONFIG).model_dump()
    payload["data"]["spec_count"] = 8
    payload["data"]["split_counts"] = {"difficulty": 2, "sft": 4, "grpo": 2}
    payload["validation"]["valid_pair_min"] = {"difficulty": 2, "sft": 4, "grpo": 2}
    payload["validation"]["manual_review_pairs"] = 2
    payload["validation"]["internal_similarity_threshold"] = 1.0
    return ExperimentConfig.model_validate(payload)


def fixture_aime_reference(config: ExperimentConfig) -> list[AimeReferenceRecord]:
    records = []
    for year in (2024, 2025):
        source = getattr(config.evaluation, f"aime_{year}")
        for row_index in range(30):
            payload = {
                "year": year,
                "source_repo_id": source.repo_id,
                "source_revision": source.revision,
                "source_split": source.split,
                "source_row_index": row_index,
                "source_id": f"{year}-{row_index}",
                "problem": f"AIME{year} fixture problem {row_index}",
                "answer": str(row_index),
                "solution": f"fixture solution {row_index}",
                "url": f"https://example.invalid/aime/{year}/{row_index}",
            }
            records.append(AimeReferenceRecord(**payload, content_sha256=sha256_value(payload)))
    return records


def test_fixture_generation_validation_and_curation(tmp_path: Path) -> None:
    config = fixture_config()
    specs_path = tmp_path / "specs.jsonl"
    raw_20b = tmp_path / "raw-20b.jsonl"
    raw_120b = tmp_path / "raw-120b.jsonl"
    votes_20b = tmp_path / "votes-20b.jsonl"
    votes_120b = tmp_path / "votes-120b.jsonl"
    aime = tmp_path / "aime.jsonl"
    review = tmp_path / "review.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    curated = tmp_path / "curated.jsonl"

    write_jsonl(specs_path, build_specs(config.data))
    generate_problems(config, specs_path, raw_20b, "generator_20b", "fixture")
    generate_problems(config, specs_path, raw_120b, "generator_120b", "fixture")
    validate_problems(config, [raw_20b, raw_120b], votes_20b, "generator_20b", "fixture")
    validate_problems(config, [raw_20b, raw_120b], votes_120b, "generator_120b", "fixture")
    write_jsonl(aime, fixture_aime_reference(config))
    metrics = prepare_review(config, specs_path, [raw_20b, raw_120b], [votes_20b, votes_120b], aime, review)
    assert metrics["paired_ids"] == 8

    review_items = read_jsonl(review, ReviewItem)
    assert any(
        item.reason == "internal_near_duplicate" and item.spec_id == item.reference_id
        for item in review_items
    )
    manual_items = [item for item in review_items if item.reason == "manual_quality_sample"]
    assert len(manual_items) == 4
    assert all(item.generator_model and item.problem for item in manual_items)
    now = datetime.now(UTC).isoformat()
    write_jsonl(decisions, [
        {
            "spec_id": spec_id,
            "decision": "accept",
            "critical_defect": False,
            "reviewer": "fixture",
            "reviewed_at": now,
            "notes": "fixture review",
        }
        for spec_id in sorted({item.spec_id for item in review_items})
    ])
    result = finalize_curation(
        config,
        specs_path,
        [raw_20b, raw_120b],
        [votes_20b, votes_120b],
        review,
        decisions,
        curated,
    )
    assert result["valid_pair_counts"] == {"difficulty": 2, "sft": 4, "grpo": 2}
    assert len(read_jsonl(curated)) == 16


def test_curation_rejects_generation_spec_drift(tmp_path: Path) -> None:
    config = fixture_config()
    specs = build_specs(config.data)
    changed = specs[0].model_copy(update={"category": "tampered"})
    specs_path = tmp_path / "specs.jsonl"
    write_jsonl(specs_path, [changed, *specs[1:]])

    with pytest.raises(ValueError, match="canonical config-derived plan"):
        prepare_review(config, specs_path, [], [], tmp_path / "aime.jsonl", tmp_path / "review.jsonl")


def test_curation_rejects_raw_spec_field_drift_even_with_new_content_hash(tmp_path: Path) -> None:
    config = fixture_config()
    specs_path = tmp_path / "specs.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    tampered_path = tmp_path / "tampered.jsonl"
    write_jsonl(specs_path, build_specs(config.data))
    rows = generate_problems(config, specs_path, raw_path, "generator_120b", "fixture")
    payload = rows[0].model_dump(exclude={"content_sha256"})
    payload["category"] = "tampered"
    payload["content_sha256"] = sha256_value(payload)
    write_jsonl(tampered_path, [payload, *rows[1:]])

    with pytest.raises(ValueError, match="raw generation/spec mismatch"):
        prepare_review(
            config,
            specs_path,
            [tampered_path],
            [],
            tmp_path / "aime.jsonl",
            tmp_path / "review.jsonl",
        )


def test_curation_rejects_raw_content_hash_drift(tmp_path: Path) -> None:
    config = fixture_config()
    specs_path = tmp_path / "specs.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    tampered_path = tmp_path / "tampered.jsonl"
    write_jsonl(specs_path, build_specs(config.data))
    rows = generate_problems(config, specs_path, raw_path, "generator_120b", "fixture")
    changed = rows[0].model_copy(update={"problem": "tampered"})
    write_jsonl(tampered_path, [changed, *rows[1:]])

    with pytest.raises(ValueError, match="content hash mismatch"):
        prepare_review(
            config,
            specs_path,
            [tampered_path],
            [],
            tmp_path / "aime.jsonl",
            tmp_path / "review.jsonl",
        )

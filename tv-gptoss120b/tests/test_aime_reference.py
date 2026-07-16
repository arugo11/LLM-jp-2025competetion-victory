from pathlib import Path

import pytest

from tv_gptoss120b.aime_reference import AimeReferenceRecord, load_verified_aime_reference
from tv_gptoss120b.config import load_config
from tv_gptoss120b.hashing import sha256_value
from tv_gptoss120b.jsonl import write_jsonl

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def reference_rows() -> list[AimeReferenceRecord]:
    config = load_config(CONFIG)
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


def test_aime_reference_requires_exact_pinned_sixty_rows(tmp_path: Path) -> None:
    path = tmp_path / "aime-reference.jsonl"
    write_jsonl(path, reference_rows())

    records = load_verified_aime_reference(load_config(CONFIG), path)

    assert len(records) == 60
    assert {(record.year, record.source_row_index) for record in records} == {
        (year, row_index) for year in (2024, 2025) for row_index in range(30)
    }


def test_aime_reference_rejects_revision_drift_even_with_new_content_hash(tmp_path: Path) -> None:
    records = reference_rows()
    payload = records[0].model_dump(exclude={"content_sha256"})
    payload["source_revision"] = "0" * 40
    payload["content_sha256"] = sha256_value({key: value for key, value in payload.items() if key != "content_sha256"})
    path = tmp_path / "aime-reference.jsonl"
    write_jsonl(path, [payload, *records[1:]])

    with pytest.raises(ValueError, match="source pin mismatch"):
        load_verified_aime_reference(load_config(CONFIG), path)


def test_aime_reference_rejects_content_drift(tmp_path: Path) -> None:
    records = reference_rows()
    records[0] = records[0].model_copy(update={"problem": "tampered"})
    path = tmp_path / "aime-reference.jsonl"
    write_jsonl(path, records)

    with pytest.raises(ValueError, match="content hash mismatch"):
        load_verified_aime_reference(load_config(CONFIG), path)

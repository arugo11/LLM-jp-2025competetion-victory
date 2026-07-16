from __future__ import annotations

from pathlib import Path

from pydantic import Field

from .config import ExperimentConfig
from .hashing import sha256_value
from .jsonl import read_jsonl, write_jsonl
from .schema import StrictModel


class AimeReferenceRecord(StrictModel):
    year: int = Field(ge=2024, le=2025)
    source_repo_id: str
    source_revision: str
    source_split: str
    source_row_index: int = Field(ge=0, lt=30)
    source_id: str
    problem: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    solution: str = Field(min_length=1)
    url: str = Field(min_length=1)
    content_sha256: str


def _payload(record: AimeReferenceRecord) -> dict:
    return record.model_dump(mode="json", exclude={"content_sha256"})


def build_aime_reference(config: ExperimentConfig, output_path: Path) -> list[AimeReferenceRecord]:
    """Materialize the exact pinned AIME24/25 train rows for contamination checks."""
    from datasets import load_dataset

    records = []
    for year in (2024, 2025):
        source = getattr(config.evaluation, f"aime_{year}")
        dataset = load_dataset(
            source.repo_id,
            revision=source.revision,
            split=source.split,
        )
        if len(dataset) != 30:
            raise ValueError(f"pinned AIME{year} train split must contain exactly 30 rows, got {len(dataset)}")
        required_columns = {"id", "problem", "answer", "solution", "url", "year"}
        if not required_columns.issubset(dataset.column_names):
            raise ValueError(
                f"pinned AIME{year} schema is missing columns: {sorted(required_columns - set(dataset.column_names))}"
            )
        for row_index, row in enumerate(dataset):
            if int(row["year"]) != year:
                raise ValueError(f"pinned AIME{year} row {row_index} has year={row['year']}")
            payload = {
                "year": year,
                "source_repo_id": source.repo_id,
                "source_revision": source.revision,
                "source_split": source.split,
                "source_row_index": row_index,
                "source_id": str(row["id"]),
                "problem": str(row["problem"]),
                "answer": str(row["answer"]),
                "solution": str(row["solution"]),
                "url": str(row["url"]),
            }
            records.append(AimeReferenceRecord(**payload, content_sha256=sha256_value(payload)))
    write_jsonl(output_path, records)
    return records


def load_verified_aime_reference(
    config: ExperimentConfig,
    path: Path,
) -> list[AimeReferenceRecord]:
    records = read_jsonl(path, AimeReferenceRecord)
    expected_order = [(year, row_index) for year in (2024, 2025) for row_index in range(30)]
    expected_keys = set(expected_order)
    actual_keys = [(record.year, record.source_row_index) for record in records]
    if actual_keys != expected_order or set(actual_keys) != expected_keys:
        raise ValueError(
            "AIME reference must contain exactly the pinned 60 rows in canonical order: "
            f"rows={len(actual_keys)}, unique={len(set(actual_keys))}"
        )
    for record in records:
        expected_source = getattr(config.evaluation, f"aime_{record.year}")
        actual_source = (record.source_repo_id, record.source_revision, record.source_split)
        if actual_source != (
            expected_source.repo_id,
            expected_source.revision,
            expected_source.split,
        ):
            raise ValueError(f"AIME{record.year} reference source pin mismatch at row {record.source_row_index}")
        if sha256_value(_payload(record)) != record.content_sha256:
            raise ValueError(f"AIME{record.year} reference content hash mismatch at row {record.source_row_index}")
    for year in (2024, 2025):
        source_ids = [record.source_id for record in records if record.year == year]
        if len(set(source_ids)) != 30:
            raise ValueError(f"AIME{year} reference source IDs are not unique")
    return records

import pytest

from tv_gptoss120b.statistics import difficulty_summary


def test_difficulty_summary_rejects_unmatched_samples() -> None:
    rows = [{
        "spec_id": "spec-0001",
        "arm": "thinking",
        "generator_model": "generator_20b",
        "sample_index": 0,
        "correct": True,
    }]
    with pytest.raises(ValueError, match="incomplete"):
        difficulty_summary(rows, 10, 37)

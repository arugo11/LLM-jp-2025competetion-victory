from pathlib import Path

import pytest

from tv_gptoss120b.aime import normalize_detail_rows, summarize_aime, verify_matched_fingerprint
from tv_gptoss120b.config import load_config
from tv_gptoss120b.evaluation import (
    HARNESS_MANIFEST,
    _harness_tree_digest,
    matched_evaluation_fingerprint,
)


def test_harness_tree_digest_detects_untracked_source_but_ignores_runtime_cache(tmp_path: Path) -> None:
    (tmp_path / "task.py").write_text("VALUE = 1\n", encoding="utf-8")
    first = _harness_tree_digest(tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "task.pyc").write_bytes(b"runtime-cache")
    (tmp_path / HARNESS_MANIFEST).write_text("{}\n", encoding="utf-8")
    assert _harness_tree_digest(tmp_path) == first

    (tmp_path / "untracked_override.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert _harness_tree_digest(tmp_path) != first


def test_matched_evaluation_fingerprint_is_fail_closed() -> None:
    verify_matched_fingerprint([{"matched_config_sha256": "same"}], "same")
    with pytest.raises(ValueError, match="not matched"):
        verify_matched_fingerprint(
            [{"matched_config_sha256": "base"}, {"matched_config_sha256": "post"}],
            "base",
        )


def test_matched_evaluation_fingerprint_includes_evaluator_revisions() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "experiment.yaml"
    config = load_config(config_path)
    original = config.evaluation
    fingerprint = matched_evaluation_fingerprint(config)
    changed = config.model_copy(update={
        "evaluation": original.model_copy(update={"max_model_len": original.max_model_len + 1})
    })
    assert matched_evaluation_fingerprint(changed) != fingerprint


def test_matched_evaluation_rejects_prompt_drift() -> None:
    rows = [
        {
            "matched_config_sha256": "same",
            "model": "thinking-base",
            "year": 2024,
            "problem_id": "1",
            "sample_index": 0,
            "full_prompt": "prompt-a",
            "expected_answer": "1",
        },
        {
            "matched_config_sha256": "same",
            "model": "thinking-post",
            "year": 2024,
            "problem_id": "1",
            "sample_index": 0,
            "full_prompt": "prompt-b",
            "expected_answer": "1",
        },
    ]
    with pytest.raises(ValueError, match="full_prompt mismatch"):
        verify_matched_fingerprint(rows, "same")


def test_normalize_lighteval_detail_rows_expands_four_samples() -> None:
    rows = [
        {
            "predictions": ["work \\boxed{1}", "\\boxed{2}", "no answer", "\\boxed{1}"],
            "gold": ["1"],
            "cont_tokens": [[1], [2, 3], [], [4]],
            "truncated": [0, 0, 1, 0],
            "full_prompt": "problem",
        }
    ]
    normalized = normalize_detail_rows(
        "thinking-base",
        2024,
        rows,
        expected_problem_count=1,
        strict_math_verify=False,
    )
    assert len(normalized) == 4
    assert [row["correct"] for row in normalized] == [True, False, False, True]
    assert normalized[2]["parsed"] is False
    assert normalized[2]["truncated"] is True


def test_aime_matched_summary_detects_positive_arm() -> None:
    rows = []
    for model in ("thinking-base", "thinking-post", "instruct-base", "instruct-post"):
        for year in (2024, 2025):
            for problem in range(1, 31):
                for sample in range(4):
                    correct = model == "thinking-post" and problem == 1
                    rows.append({
                        "model": model,
                        "year": year,
                        "problem_id": str(problem),
                        "sample_index": sample,
                        "correct": correct,
                        "parsed": True,
                    })
    summary = summarize_aime(rows, bootstrap_samples=200, seed=37)
    assert summary["arms"]["thinking"]["delta_pass_at_1_4"] > 0
    assert summary["arms"]["instruct"]["delta_pass_at_1_4"] == 0
    assert summary["scientific_aime_gate"] is True


def test_aime_rejects_unmatched_records() -> None:
    try:
        summarize_aime([])
    except ValueError as error:
        assert "960" in str(error)
    else:
        raise AssertionError("unmatched evaluation must fail")


def test_aime_rejects_duplicate_records() -> None:
    rows = []
    for model in ("thinking-base", "thinking-post", "instruct-base", "instruct-post"):
        for year in (2024, 2025):
            for problem in range(1, 31):
                for sample in range(4):
                    rows.append({
                        "model": model,
                        "year": year,
                        "problem_id": str(problem),
                        "sample_index": sample,
                        "correct": False,
                        "parsed": False,
                    })
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicates=1"):
        summarize_aime(rows)

from tv_gptoss120b.math_utils import (
    extract_last_boxed,
    jaccard,
    normalize_text,
    problem_rule_rejections,
    solution_rule_rejections,
    token_ngrams,
)


def test_extract_last_boxed_supports_nested_braces() -> None:
    assert extract_last_boxed(r"first \boxed{1}, final \boxed{\frac{2}{3}}") == r"\frac{2}{3}"
    assert extract_last_boxed(r"\boxed{broken") is None


def test_normalization_and_ngram_similarity() -> None:
    assert normalize_text(" $x = 1$。") == "$x=1$"
    grams = token_ngrams("abc 123", 2)
    assert jaccard(grams, grams) == 1.0


def test_structural_math_rules_reject_invalid_outputs() -> None:
    assert "problem_not_japanese" in problem_rule_rejections("Compute $x$.")
    assert "problem_unbalanced_latex" in problem_rule_rejections("次を求めよ。\\(x+1")
    assert "solution_box_count_not_one" in solution_rule_rejections(
        r"答えは\boxed{1}または\boxed{2}",
        "2",
    )
    assert "non_finite_answer" in solution_rule_rejections(r"\boxed{\infty}", r"\infty")

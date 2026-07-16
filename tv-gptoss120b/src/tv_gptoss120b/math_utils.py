from __future__ import annotations

import re
import unicodedata


def extract_last_boxed(text: str | None) -> str | None:
    if not text:
        return None
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    cursor = start + len(marker)
    depth = 1
    content_start = cursor
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:cursor].strip() or None
        cursor += 1
    return None


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    value = value.replace(r"\displaystyle", "")
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[。、，,.;；:：]", "", value)
    return value


def token_ngrams(text: str, size: int) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-zA-Z]+|\d+|[^\s]", unicodedata.normalize("NFKC", text).lower())
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def minhash_candidate_pairs(
    grams_by_id: dict[str, set[tuple[str, ...]]],
    threshold: float,
    *,
    num_perm: int = 128,
) -> set[tuple[str, str]]:
    from datasketch import MinHash, MinHashLSH

    if threshold >= 1.0:
        return set()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signatures = {}
    candidates: set[tuple[str, str]] = set()
    for item_id in sorted(grams_by_id):
        signature = MinHash(num_perm=num_perm, seed=37)
        grams = grams_by_id[item_id]
        for gram in sorted(grams):
            signature.update("\u241f".join(gram).encode("utf-8"))
        signatures[item_id] = signature
        for other_id in lsh.query(signature):
            candidates.add(tuple(sorted((item_id, other_id))))
        lsh.insert(item_id, signature)
    return candidates


def math_equivalent(prediction: str, reference: str, *, strict: bool = True) -> bool:
    try:
        from math_verify import parse, verify
    except ImportError:
        if strict:
            raise RuntimeError("math-verify is required for production validation") from None
        return normalize_text(prediction) == normalize_text(reference)
    try:
        return bool(verify(parse(prediction), parse(reference)))
    except Exception:
        return False


def problem_rule_rejections(problem: str | None) -> list[str]:
    if not problem or not problem.strip():
        return ["empty_problem"]
    reasons: list[str] = []
    if r"\displaystyle" in problem:
        reasons.append("forbidden_displaystyle")
    if "，" in problem:
        reasons.append("full_width_comma")
    if len(problem) > 8000:
        reasons.append("problem_too_long")
    if problem.count(r"\boxed{") > 0:
        reasons.append("problem_contains_answer_box")
    if not re.search(r"[ぁ-んァ-ヶ一-龯]", problem):
        reasons.append("problem_not_japanese")
    if not _balanced_latex(problem):
        reasons.append("problem_unbalanced_latex")
    return reasons


def _balanced_latex(text: str) -> bool:
    depth = 0
    for index, char in enumerate(text):
        if index > 0 and text[index - 1] == "\\":
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and text.count(r"\(") == text.count(r"\)") and text.count(r"\[") == text.count(r"\]")


def solution_rule_rejections(solution: str | None, answer: str | None) -> list[str]:
    if not solution or not solution.strip():
        return ["empty_solution"]
    reasons = []
    if solution.count(r"\boxed{") != 1:
        reasons.append("solution_box_count_not_one")
    if len(solution) > 32_000:
        reasons.append("solution_too_long")
    if not _balanced_latex(solution):
        reasons.append("solution_unbalanced_latex")
    if answer is None:
        reasons.append("missing_boxed_answer")
        return reasons
    normalized = normalize_text(answer)
    if re.search(r"(?:nan|inf(?:inity|ty)?|∞)", normalized):
        reasons.append("non_finite_answer")
    if any(marker in answer for marker in (r"\pm", "または", " or ", "、", ",", "，")):
        reasons.append("multiple_answer_candidate")
    return reasons

import textwrap
from typing import Optional

import answer_prompts as ap


def extract_answer(text: Optional[str]) -> Optional[str]:
    r"""Extract the content inside the last \\boxed{...} tag."""
    if not text:
        return None

    start_marker = "\\boxed{"
    start_indices = []
    idx = text.find(start_marker)
    while idx != -1:
        start_indices.append(idx)
        idx = text.find(start_marker, idx + 1)

    if not start_indices:
        return None

    for start_idx in reversed(start_indices):
        content_start = start_idx + len(start_marker)
        brace_count = 1
        current_idx = content_start

        while current_idx < len(text) and brace_count > 0:
            if text[current_idx] == "{":
                brace_count += 1
            elif text[current_idx] == "}":
                brace_count -= 1
            current_idx += 1

        if brace_count == 0:
            return text[content_start : current_idx - 1]

    return None


def _normalize_one_line(text: str) -> str:
    return " ".join((text or "").split())


def _last_non_empty_line(text: str) -> Optional[str]:
    for line in reversed((text or "").splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _tir_success(process_status: str, stdout: str, stderr: str) -> bool:
    if process_status != "completed":
        return False
    if not _last_non_empty_line(stdout):
        return False
    lower_stderr = (stderr or "").lower()
    lower_stdout = (stdout or "").lower()
    if "traceback" in lower_stderr or "syntaxerror" in lower_stderr:
        return False
    return not ("traceback" in lower_stdout or "syntaxerror" in lower_stdout)


def _extract_python_code(generation: str) -> Optional[str]:
    if not generation:
        return None
    match = ap.PYTHON_BLOCK_RE.search(generation)
    if not match:
        return None
    return textwrap.dedent(match.group(1)).strip()


def _format_tir_solution(
    problem_one_line: str,
    python_code: str,
    latex_value: str,
) -> str:
    latex_value = (latex_value or "").strip()
    return textwrap.dedent(
        f"""
        次の問題を計算する:
        {problem_one_line}

        Pythonを使って厳密に計算します。

        {ap.PYTHON_BEGIN}
        {python_code}
        {ap.PYTHON_END}

        {ap.PYTHON_OUTPUT_BEGIN}
        {latex_value}
        {ap.PYTHON_OUTPUT_END}

        Pythonの結果より、値は {latex_value} です。
        最終答\\boxed{{{latex_value}}}。
        """,
    ).strip()

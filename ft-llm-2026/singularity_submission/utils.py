"""テキスト処理とI/Oのユーティリティ関数."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompts import RESULT_BEGIN
from sympy import latex, sympify

# テキスト処理


def read_problems(path: Path) -> list[dict[str, Any]]:
    """JSONLファイルから問題定義を読み込む."""
    problems: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                problems.append(json.loads(stripped))
    return problems


def extract_blocks(text: str, begin: str, end: str) -> list[str]:
    """開始マーカーと終了マーカーの間のすべてのブロックを抽出."""
    if not text:
        return []
    pattern = re.compile(re.escape(begin) + r"(.*?)" + re.escape(end), re.DOTALL)
    return [match.strip() for match in pattern.findall(text)]


def last_non_empty_line(text: str) -> str:
    """テキストから最後の非空行を取得."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def extract_code_emergency(text: str) -> str:
    """<python>タグが欠落している場合の緊急コード抽出.

    リペアフェーズでタグが省略された場合に使用.
    <result>の前にあるコード風のブロック（import/fromで始まる）を抽出.
    """
    if not text:
        return ""

    # <result>の前のテキストのみ考慮
    result_idx = text.find(RESULT_BEGIN)
    if result_idx >= 0:
        text = text[:result_idx]

    lines = text.splitlines()
    code_lines: list[str] = []
    in_code = False
    consecutive_empty = 0

    for line in lines:
        stripped = line.strip()

        # コードブロックの開始を検出
        if not in_code and stripped.startswith(("from ", "import ")):
            in_code = True
            code_lines = [line]
            consecutive_empty = 0
            continue

        if in_code:
            if not stripped:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                code_lines.append(line)
            else:
                consecutive_empty = 0
                code_lines.append(line)

    return "\n".join(code_lines).strip()


def clip_text(text: str | None, max_chars: int) -> str:
    """テキストをmax_charsに切り詰め、省略符付きで先頭と末尾を表示."""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    half = max(0, max_chars // 2)
    return text[:half] + "\n...<clipped>...\n" + text[-half:]


def to_latex_scalar(text: str) -> str:
    """生の文字列を評価用のLaTeXスカラーに変換."""
    stripped = text.strip()
    if not stripped or stripped.lower().startswith("error"):
        return ""

    candidate = stripped
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        candidate = inner.split(",", maxsplit=1)[0].strip() if inner else ""
        if not candidate:
            return ""
    elif stripped.startswith("{") and stripped.endswith("}"):
        inner = stripped[1:-1]
        parts = [p for p in inner.split(",") if ":" in p]
        candidate = parts[0].split(":", maxsplit=1)[1].strip() if parts else ""
        if not candidate:
            return ""

    try:
        expr = sympify(candidate)
    except Exception:
        return stripped

    if isinstance(expr, (list, tuple)):
        expr = expr[0] if expr else None
    if hasattr(expr, "values"):
        values = list(expr.values())  # type: ignore[arg-type]
        expr = values[0] if values else None
    if expr is None:
        return ""
    return f"${latex(expr)}$"


def safe_exec_view(exec_dict: dict[str, Any] | None, max_chars: int) -> dict[str, Any]:
    """ログ記録用に実行結果の安全なビューを作成."""
    if not exec_dict:
        return {}
    return {
        "process_status": exec_dict.get("process_status"),
        "stdout": clip_text(str(exec_dict.get("stdout", "")), max_chars),
        "stderr": clip_text(str(exec_dict.get("stderr", "")), max_chars),
        "return_code": exec_dict.get("return_code"),
        "time": exec_dict.get("time"),
    }


# --- 結果データ構造 ---


@dataclass
class SolveResult:
    """単一の問題を解いた結果."""

    output: str
    session_id: str
    error: str = ""
    has_python_block: bool = False
    multiple_python_blocks: bool = False
    used_stdout: bool = False
    used_result_fallback: bool = False
    used_emergency_extract: bool = False
    parse_success: bool = False
    exec_success: bool = False
    result_vs_stdout_mismatch: bool = False
    temperature: float = 0.0
    python_block: str = ""
    result_block: str = ""
    stdout: str = ""
    stderr: str = ""
    repair_used: int = 0
    raw_output: str = ""

    def to_log_entry(self) -> dict[str, Any]:
        """ログエントリ辞書に変換（raw_outputは含まない）."""
        entry = {
            "id": self.session_id,
            "has_python_block": self.has_python_block,
            "multiple_python_blocks": self.multiple_python_blocks,
            "used_stdout": self.used_stdout,
            "used_result_fallback": self.used_result_fallback,
            "used_emergency_extract": self.used_emergency_extract,
            "parse_success": self.parse_success,
            "exec_success": self.exec_success,
            "result_vs_stdout_mismatch": self.result_vs_stdout_mismatch,
            "error": self.error,
            "final_output": self.output,
            "temperature": self.temperature,
            "python_block": self.python_block,
            "result_block": self.result_block,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "repair_used": self.repair_used,
        }
        return entry

    def to_log_entry_with_raw(self) -> dict[str, Any]:
        """ログエントリ辞書に変換（raw_outputを含む）."""
        entry = self.to_log_entry()
        entry["raw_output"] = self.raw_output
        return entry

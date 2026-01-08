"""Utility functions for text processing, logging, and I/O."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from prompts import RESULT_BEGIN

# =============================================================================
# Text Processing
# =============================================================================


def read_problems(path: Path) -> list[dict[str, Any]]:
    """Read problem definitions from a JSONL file."""
    problems: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                problems.append(json.loads(stripped))
    return problems


def extract_blocks(text: str, begin: str, end: str) -> list[str]:
    """Extract all blocks between begin and end markers."""
    if not text:
        return []
    pattern = re.compile(re.escape(begin) + r"(.*?)" + re.escape(end), re.DOTALL)
    return [match.strip() for match in pattern.findall(text)]


def last_non_empty_line(text: str) -> str:
    """Get the last non-empty line from text."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def extract_code_emergency(text: str) -> str:
    """Emergency code extraction when <python> tags are missing.

    This is used in repair phase when tags are omitted.
    Extracts code-like blocks (starting with import/from) before <result>.
    """
    if not text:
        return ""

    # Only consider text before <result>
    result_idx = text.find(RESULT_BEGIN)
    if result_idx >= 0:
        text = text[:result_idx]

    lines = text.splitlines()
    code_lines: list[str] = []
    in_code = False
    consecutive_empty = 0

    for line in lines:
        stripped = line.strip()

        # Detect code block start
        if not in_code and (
            stripped.startswith("from ") or stripped.startswith("import ")
        ):
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
    """Clip text to max_chars, showing head and tail with ellipsis."""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    half = max(0, max_chars // 2)
    return text[:half] + "\n...<clipped>...\n" + text[-half:]


# =============================================================================
# Logging
# =============================================================================


class TraceLogger:
    """Handles trace event logging to a file."""

    def __init__(self, trace_file: TextIO, max_chars: int = 8000):
        self.trace_file = trace_file
        self.max_chars = max_chars

    def write(self, event: dict[str, Any]) -> None:
        """Write a trace event to the log."""
        self.trace_file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def clip(self, text: str) -> str:
        """Clip text to max_chars."""
        return clip_text(text, self.max_chars)


def safe_exec_view(exec_dict: dict[str, Any] | None, max_chars: int) -> dict[str, Any]:
    """Create a safe view of execution results for logging."""
    if not exec_dict:
        return {}
    return {
        "process_status": exec_dict.get("process_status"),
        "stdout": clip_text(str(exec_dict.get("stdout", "")), max_chars),
        "stderr": clip_text(str(exec_dict.get("stderr", "")), max_chars),
        "return_code": exec_dict.get("return_code"),
        "time": exec_dict.get("time"),
    }


# =============================================================================
# Result Data Structures
# =============================================================================


@dataclass
class SolveResult:
    """Result of solving a single problem."""

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

    def to_log_entry(self, include_raw_output: bool = False) -> dict[str, Any]:
        """Convert to log entry dictionary."""
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
        if include_raw_output:
            entry["raw_output"] = self.raw_output
        return entry

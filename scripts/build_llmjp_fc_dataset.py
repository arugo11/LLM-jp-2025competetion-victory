#!/usr/bin/env python3
"""
Convert the tuning competition dev split into a Nemo-Skills dataset that
supports tool-calling style prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from copy import deepcopy
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "あなたは日本語の数学アシスタントです。"
    "与えられた問題文を丁寧に解析し、必要なら途中計算を示してから"
    "最後に `submit_answer` 関数で最終解を報告してください。"
    "回答は問題文と同じ形式（数値・式・組など）で表記し、"
    "複数解がある場合は半角カンマ区切りで順序を保ってください。"
)

DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "数学問題の最終解を1回だけ返す。",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "最終回答。分数は `\\frac{a}{b}`、集合や座標は `(x, y)` のようにTeX風の文字列で統一する。"
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "任意。途中の根拠や計算方針を短くまとめる。",
                    },
                },
                "required": ["answer"],
            },
        },
    }
]

NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
LATEX_ENV_PATTERN = re.compile(r"(\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{.*\})", re.DOTALL)


def normalize_solution(raw: str) -> str:
    """Strip LaTeX記号や余分な空白を取り除き、可能なら数値に変換する。"""
    if raw is None:
        return ""

    text = raw.strip().strip("$")
    text = re.sub(r"^\$+|\$+$", "", text)  # 両端の$や$$を除去
    text = text.replace("\n", " ").replace("\r", " ").replace("\u3000", " ")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    text = text.replace("\\quad", " ")
    text = re.sub(r"\\text\\{([^}]*)\\}", r"\\1", text)
    text = re.sub(r"\\mathrm\\{([^}]*)\\}", r"\\1", text)
    text = re.sub(r"\\phantom\\{[^}]*\\}", "", text)
    text = re.sub(r"\\begin\\{aligned\\}|\\end\\{aligned\\}", "", text)
    text = text.replace("\\cdot", "*")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,")

    numeric_candidate = text.replace(",", "")
    if NUMERIC_RE.fullmatch(numeric_candidate):
        value = float(numeric_candidate)
        return str(int(value)) if value.is_integer() else str(value)

    return text


def ensure_latex_environment(answer: str) -> str:
    stripped = answer.strip()
    if not stripped:
        return ""
    if LATEX_ENV_PATTERN.search(stripped):
        return stripped
    return f"${stripped}$"


def convert_entries(input_path: Path) -> list[dict]:
    input_path = input_path.resolve()

    processed: list[dict] = []
    with input_path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            cleaned_answer = normalize_solution(record.get("solution", ""))
            math_verify_gold = ensure_latex_environment(cleaned_answer) if cleaned_answer else ""

            system_and_user = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": record["problem"].strip()},
            ]

            processed.append(
                {
                    "id": f"llmjp_fc_dev-{int(record['id']):04d}",
                    "problem": record["problem"],
                    "reference_solution": record.get("solution", ""),
                    "expected_answer": math_verify_gold,
                    "expected_answer_plain": cleaned_answer,
                    "metadata": {"category": record.get("category"), "unit": record.get("unit")},
                    "question": [system_and_user],
                    "tools": deepcopy(DEFAULT_TOOLS),
                    "single_turn": True,
                }
            )
    return processed


def write_skills_jsonl(entries: list[dict], output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dst:
        for entry in entries:
            dst.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(entries)} samples to {output_path}")


def write_math_verify_csv(entries: list[dict], csv_path: Path) -> None:
    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "problem", "category", "unit", "gold"]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            metadata = entry.get("metadata", {})
            writer.writerow(
                {
                    "id": entry["id"],
                    "problem": entry["problem"],
                    "category": metadata.get("category", ""),
                    "unit": metadata.get("unit", ""),
                    "gold": entry["expected_answer"],
                }
            )
    print(f"Saved Math-Verify gold file to {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format llm-jp tuning dev split for Nemo-Skills tool-calling evals.")
    parser.add_argument("--input", type=Path, default=Path("data/dev.jsonl"), help="元データのjsonlパス")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Skills/nemo_skills/dataset/llmjp_fc_dev/test.jsonl"),
        help="整形後のjsonl出力パス",
    )
    parser.add_argument(
        "--math_verify_csv",
        type=Path,
        default=None,
        help="Math-Verifyのgold用CSVを書き出す場合に指定するパス",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = convert_entries(args.input)
    write_skills_jsonl(entries, args.output)
    if args.math_verify_csv:
        write_math_verify_csv(entries, args.math_verify_csv)


if __name__ == "__main__":
    main()

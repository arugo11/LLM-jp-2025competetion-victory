"""Singularity submission entry point for solving math problems via vLLM + NeMo-Skills sandbox."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import textwrap
from pathlib import Path
from typing import Any

from nemo_skills.code_execution.sandbox import get_sandbox  # type: ignore
from nemo_skills.inference.model import get_model  # type: ignore

PYTHON_BEGIN = "<python>"
PYTHON_END = "</python>"
RESULT_BEGIN = "<result>"
RESULT_END = "</result>"

PROMPT_TEMPLATE: str = textwrap.dedent(
    """
あなたは厳密な数学のsolverです。
以下のフォーマットに従ってPythonを使用して計算を行ってください。
他の文字・説明・markdown・フェンスは使用禁止です。

<python>
# ここにPythonコードを記述し、最終的な計算結果を求める
# 最後に print(...) で答えを出力すること
</python>
<result>answer</result>

ルール:
- <python>の前や</result>の後にテキストを出力するのは禁止です
- Markdownの ```python などを使用するのは禁止です
- 他のプログラミング言語を使用するのは禁止です
- 自然言語による説明や要約などを含めるのは禁止です
- 確実に 1つの <python> ブロックと <result> ブロックだけにしてください

問題:
{question}
    """
).strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Singularity Submission Example")
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Path to the input file",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to the output file",
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        default=Path("inference_log.jsonl"),
        help="Path to the inference log JSONL file",
    )

    parser.add_argument("--tir-sandbox-type", default="local")
    parser.add_argument("--tir-sandbox-host", default="127.0.0.1")
    parser.add_argument("--tir-sandbox-port", type=int, default=6000)

    parser.add_argument("--tir-llm-server-type", default="vllm")
    parser.add_argument("--tir-llm-host", default="127.0.0.1")
    parser.add_argument("--tir-llm-port", type=int, default=8000)
    parser.add_argument("--tir-model-name", type=str, default=None)

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retry-temperature", type=float, default=None)

    parser.add_argument("--code-language", type=str, default="python")
    parser.add_argument("--code-timeout", type=float, default=10.0)
    parser.add_argument("--max-output-chars", type=int, default=1000)

    return parser.parse_args()


def _build_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}]


def main() -> None:
    """推論システムを非同期で実行する関数"""
    args: argparse.Namespace = _parse_args()
    asyncio.run(_run_math_pipeline(args))


def _read_problems(path: Path) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            problems.append(json.loads(stripped))
    return problems


def _extract_blocks(text: str, begin: str, end: str) -> list[str]:
    if not text:
        return []
    pattern = re.compile(re.escape(begin) + r"(.*?)" + re.escape(end), re.DOTALL)
    return [match.strip() for match in pattern.findall(text)]


def _last_non_empty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _is_execution_error(execution_dict: dict[str, Any] | None) -> bool:
    if not execution_dict:
        return True
    if execution_dict.get("process_status") in {"error", "timeout"}:
        return True
    stdout = execution_dict.get("stdout", "")
    stderr = execution_dict.get("stderr", "")
    if "Traceback" in stdout or "Traceback" in stderr:
        return True
    if "SyntaxError" in stdout or "SyntaxError" in stderr:
        return True
    return False


async def _generate_once(
    llm,
    messages: list[dict[str, str]],
    temperature: float,
    max_new_tokens: int,
    min_tokens: int,
) -> str:
    extra_body = {"min_tokens": min_tokens} if min_tokens > 0 else None
    result = await llm.generate_async(
        prompt=messages,
        tokens_to_generate=max_new_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )
    return result.get("generation", "")


async def _run_math_pipeline(args: argparse.Namespace) -> None:
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input path not found: {args.input_path}")

    problems = _read_problems(args.input_path)
    model_name = args.tir_model_name or str(args.model_path.resolve())
    llm = get_model(
        server_type=args.tir_llm_server_type,
        host=args.tir_llm_host,
        port=args.tir_llm_port,
        model=model_name,
    )

    sandbox = get_sandbox(
        sandbox_type=args.tir_sandbox_type,
        host=args.tir_sandbox_host,
        port=args.tir_sandbox_port,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with args.output_path.open("w", encoding="utf-8") as out_f, args.log_path.open(
            "a", encoding="utf-8"
        ) as log_f:
            for idx, problem in enumerate(problems):
                question = str(problem.get("problem", ""))
                messages = _build_messages(question)
                session_id = str(problem.get("id", idx))

                temperatures = [args.temperature]
                if args.retry_temperature is not None:
                    temperatures.append(args.retry_temperature)

                raw_output = ""
                python_blocks: list[str] = []
                result_blocks: list[str] = []
                execution_dict: dict[str, Any] | None = None
                used_temperature = args.temperature

                for attempt, temperature in enumerate(temperatures, start=1):
                    used_temperature = temperature
                    raw_output = await _generate_once(
                        llm,
                        messages,
                        temperature=temperature,
                        max_new_tokens=args.max_new_tokens,
                        min_tokens=args.min_tokens,
                    )

                    python_blocks = _extract_blocks(raw_output, PYTHON_BEGIN, PYTHON_END)
                    result_blocks = _extract_blocks(raw_output, RESULT_BEGIN, RESULT_END)

                    if not python_blocks:
                        if attempt < len(temperatures):
                            continue
                        execution_dict = None
                        break

                    try:
                        execution_dict, _ = await sandbox.execute_code(
                            generated_code=python_blocks[0],
                            language=args.code_language,
                            timeout=args.code_timeout,
                            max_output_characters=args.max_output_chars,
                        )
                    except Exception:
                        execution_dict = None

                    if _is_execution_error(execution_dict):
                        if attempt < len(temperatures):
                            continue
                    break

                has_python_block = bool(python_blocks)
                multiple_python_blocks = len(python_blocks) > 1
                stdout = execution_dict.get("stdout", "") if execution_dict else ""
                stderr = execution_dict.get("stderr", "") if execution_dict else ""

                error = ""
                used_stdout = False
                if not has_python_block:
                    error = "no_python_block"
                    final_output = "no_python_block"
                elif _is_execution_error(execution_dict):
                    error = "execution_error"
                    final_output = "execution_error"
                else:
                    last_stdout_line = _last_non_empty_line(stdout)
                    if last_stdout_line:
                        final_output = last_stdout_line
                        used_stdout = True
                    elif result_blocks:
                        final_output = result_blocks[0].strip()
                    else:
                        final_output = raw_output.strip()

                problem["output"] = final_output
                out_f.write(json.dumps(problem, ensure_ascii=False) + "\n")

                log_entry = {
                    "id": session_id,
                    "has_python_block": has_python_block,
                    "multiple_python_blocks": multiple_python_blocks,
                    "used_stdout": used_stdout,
                    "error": error,
                    "final_output": final_output,
                    "temperature": used_temperature,
                    "raw_output": raw_output,
                    "python_block": python_blocks[0] if python_blocks else "",
                    "result_block": result_blocks[0] if result_blocks else "",
                    "stdout": stdout,
                    "stderr": stderr,
                }
                log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    finally:
        await sandbox.close()


if __name__ == "__main__":
    main()

import argparse
import asyncio
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.model import get_code_execution_model

PYTHON_BEGIN = "<python>"
PYTHON_END = "</python>"
RESULT_BEGIN = "<result>"
RESULT_END = "</result>"

SYSTEM_PROMPT = (
    "Environment: ipython\n\n"
    "You are a careful math assistant that MUST use Python for every task.\n"
    "Always respond with a single <python> block that prints the final answer,\n"
    "then echo the same value inside <result> tags on the next line."
)

USER_PROMPT_TEMPLATE = textwrap.dedent(
    """
    次の数学の問題をPythonだけで解き、printで最終解のみを出力してください。
    自然言語での解説は禁止です。<python>...</python> と <result>...</result> を必ず含めてください。

    # 問題
    {question}
    """,
).strip()

PYTHON_BLOCK_RE = re.compile(
    re.escape(PYTHON_BEGIN) + r"(.*?)" + re.escape(PYTHON_END),
    re.IGNORECASE | re.DOTALL,
)
RESULT_BLOCK_RE = re.compile(
    re.escape(RESULT_BEGIN) + r"(.*?)" + re.escape(RESULT_END),
    re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Singularity Submission with TIR")
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to the local model directory (still required by the harness)",
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)

    parser.add_argument("--tir-llm-host", default="127.0.0.1")
    parser.add_argument("--tir-llm-port", type=int, default=8000)
    parser.add_argument("--tir-llm-server-type", default="vllm")
    parser.add_argument(
        "--tir-model-name",
        default=None,
        help="Optional model identifier to pass to get_code_execution_model",
    )
    parser.add_argument("--tir-sandbox-type", default="local")
    parser.add_argument("--tir-sandbox-host", default="127.0.0.1")
    parser.add_argument("--tir-sandbox-port", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--log_path",
        type=Path,
        default=None,
        help="Optional path to append per-problem inference logs as JSONL",
    )
    return parser.parse_args()


def _read_problems(path: Path) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
    return problems


def _build_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser question:\n{question.strip()}\n"


def _extract_block(text: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return textwrap.dedent(match.group(1)).strip()


def _strip_result_wrappers(text: str) -> str:
    match = RESULT_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _last_non_empty_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _derive_answer(response_text: str, stdout: str) -> str:
    stdout_line = _last_non_empty_line(stdout)
    if stdout_line:
        return _strip_result_wrappers(stdout_line)

    result_block = _extract_block(response_text, RESULT_BLOCK_RE)
    if result_block:
        return result_block

    return response_text.strip()


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _append_log(log_path: Path | None, record: dict[str, Any]) -> None:
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_model_name(args: argparse.Namespace) -> str:
    if args.tir_model_name:
        return args.tir_model_name
    return args.model_path.name


async def run_inference(args: argparse.Namespace) -> None:
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    problems = _read_problems(args.input_path)
    if not problems:
        args.output_path.write_text("")
        return

    sandbox = get_sandbox(
        sandbox_type=args.tir_sandbox_type,
        host=args.tir_sandbox_host,
        port=args.tir_sandbox_port,
    )

    llm = get_code_execution_model(
        server_type=args.tir_llm_server_type,
        host=args.tir_llm_host,
        port=args.tir_llm_port,
        model=_resolve_model_name(args),
        sandbox=sandbox,
    )

    for idx, problem in enumerate(problems):
        question = problem.get("problem", "")
        session_id = str(problem.get("id", idx))
        prompt = _build_prompt(question)
        stdout = ""
        stderr = ""
        generation_text = ""
        error_message: str | None = None

        try:
            base_result = await llm.model.generate_async(
                prompt=prompt,
                tokens_to_generate=args.max_new_tokens,
                temperature=args.temperature,
            )
            generation_text = (base_result.get("generation") or "").strip()
        except Exception as exc:  # pragma: no cover - runtime guard in Singularity
            _log(f"[TIR] Generation failed for {session_id}: {exc}")
            problem["output"] = ""
            error_message = str(exc)
            _append_log(
                args.log_path,
                {
                    "id": session_id,
                    "prompt": prompt,
                    "generation": generation_text,
                    "stdout": stdout,
                    "stderr": stderr,
                    "error": error_message,
                },
            )
            continue

        output_text = generation_text
        if not output_text:
            _log(f"[TIR] Empty generation for {session_id}")
            problem["output"] = ""
            _append_log(
                args.log_path,
                {
                    "id": session_id,
                    "prompt": prompt,
                    "generation": generation_text,
                    "stdout": stdout,
                    "stderr": stderr,
                    "error": "empty generation",
                },
            )
            continue

        code_block = _extract_block(output_text, PYTHON_BLOCK_RE)

        if code_block:
            try:
                _, execution_dict, _ = await llm.execute_generated_code(
                    prompt,
                    PYTHON_BEGIN,
                    PYTHON_END,
                    output_text,
                    session_id=session_id,
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                _log(f"[TIR] Execution failed for {session_id}: {exc}")
                error_message = f"execution failed: {exc}"
            else:
                stdout = execution_dict.get("stdout", "") or ""
                stderr = execution_dict.get("stderr", "") or ""
                if stderr.strip():
                    _log(f"[TIR] stderr for {session_id}: {stderr.strip()}")
        else:
            _log(f"[TIR] Missing <python> block for {session_id}; using raw text.")

        problem["output"] = _derive_answer(output_text, stdout)

        _append_log(
            args.log_path,
            {
                "id": session_id,
                "prompt": prompt,
                "generation": generation_text,
                "stdout": stdout,
                "stderr": stderr,
                "error": error_message,
                "final_output": problem["output"],
            },
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()

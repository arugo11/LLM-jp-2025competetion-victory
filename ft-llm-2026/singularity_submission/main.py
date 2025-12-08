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

SYSTEM_PROMPT = textwrap.dedent(
    """
    Environment: ipython

    You are a strict math solver. You MUST use Python and you MUST obey the ONLY allowed format below.
    唯一許可された出力テンプレートは次のとおりです。他の文字・説明・Markdown・フェンスは禁止です。

    Output template (exactly, nothing before/after):
    <python>
    # write Python code that computes the final numeric answer
    print(answer)
    </python>
    <result>answer</result>

    Rules / ルール:
    - Do NOT emit any text before <python> or after </result>.
    - Do NOT use Markdown fences like ```python or any language tags.
    - No natural-language explanations or summaries; Python only.
    - Exactly one <python> block and one <result> block. Nothing else.
    - If you deviate from this format, the answer is invalid. Format first line must be '<python>'.
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
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--retry-temperature",
        type=float,
        default=0.2,
        help="Temperature to use when retrying empty generations.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=8,
        help="Min tokens to request via vLLM extra_body (avoids empty generations).",
    )
    parser.add_argument(
        "--empty-retry",
        type=int,
        default=2,
        help="Number of retries when generation is empty. 0 disables retries.",
    )
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
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "User question (solve only with Python and REQUIRED tags):\n"
        f"{question.strip()}\n"
        "Remember: output must be exactly the <python>...</python> then <result>...</result> lines, nothing else."
    )


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


def _derive_answer(
    response_text: str,
    stdout: str,
    result_block: str | None,
) -> tuple[str, bool, bool]:
    # Priority is explicitly stdout > <result> block > raw response to avoid future accidental changes.
    stdout_line = _last_non_empty_line(stdout)
    if stdout_line:
        return _strip_result_wrappers(stdout_line), True, False

    if result_block:
        return result_block.strip(), False, True

    return response_text.strip(), False, False


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

    # vLLM extra body: enforce minimum tokens to reduce empty generations
    extra_body = {"min_tokens": args.min_tokens} if args.min_tokens and args.min_tokens > 0 else None

    for idx, problem in enumerate(problems):
        question = problem.get("problem", "")
        session_id = str(problem.get("id", idx))
        prompt = _build_prompt(question)
        stdout = ""
        stderr = ""
        generation_text = ""
        error_message: str | None = None
        used_stdout = False
        used_result_block = False
        has_python_block = False
        has_result_block = False
        multiple_python_blocks = False
        multiple_result_blocks = False
        result_block_valid = False

        # Retry loop for empty generations
        try:
            for attempt in range(args.empty_retry + 1):
                temp = args.temperature if attempt == 0 else args.retry_temperature
                base_result = await llm.model.generate_async(
                    prompt=prompt,
                    tokens_to_generate=args.max_new_tokens,
                    temperature=temp,
                    extra_body=extra_body,
                )
                generation_text = (base_result.get("generation") or "").strip()
                if generation_text:
                    break
                if attempt < args.empty_retry:
                    _log(
                        f"[TIR] Empty generation for {session_id}; retry {attempt + 1}/{args.empty_retry} "
                        f"with temperature {temp}"
                    )
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
                    "final_output": "",
                    "has_python_block": False,
                    "has_result_block": False,
                    "multiple_python_blocks": False,
                    "multiple_result_blocks": False,
                    "result_block_valid": False,
                    "used_stdout": False,
                    "used_result_block": False,
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
                    "final_output": problem["output"],
                    "has_python_block": False,
                    "has_result_block": False,
                    "multiple_python_blocks": False,
                    "multiple_result_blocks": False,
                    "result_block_valid": False,
                    "used_stdout": False,
                    "used_result_block": False,
                },
            )
            continue

        python_blocks = PYTHON_BLOCK_RE.findall(output_text)
        result_blocks = RESULT_BLOCK_RE.findall(output_text)
        has_python_block = bool(python_blocks)
        has_result_block = bool(result_blocks)
        multiple_python_blocks = len(python_blocks) > 1
        multiple_result_blocks = len(result_blocks) > 1
        code_block = (
            textwrap.dedent(python_blocks[0]).strip() if has_python_block else None
        )
        result_block_raw = (
            textwrap.dedent(result_blocks[0]).strip() if has_result_block else None
        )
        if multiple_python_blocks:
            _log(f"[TIR] Multiple <python> blocks for {session_id}; using first only.")
        if multiple_result_blocks:
            _log(f"[TIR] Multiple <result> blocks for {session_id}; using first only.")
        result_block = None
        if (
            result_block_raw
            and "<python>" not in result_block_raw.lower()
            and "=" not in result_block_raw
        ):
            result_block_valid = True
            result_block = result_block_raw

        if not code_block:
            _log(f"[TIR] Missing <python> block for {session_id}; skipping execution.")
            problem["output"] = "no_python_block"
            _append_log(
                args.log_path,
                {
                    "id": session_id,
                    "prompt": prompt,
                    "generation": generation_text,
                    "stdout": stdout,
                    "stderr": stderr,
                    "error": "missing python block",
                    "final_output": problem["output"],
                    "has_python_block": has_python_block,
                    "has_result_block": has_result_block,
                    "multiple_python_blocks": multiple_python_blocks,
                    "multiple_result_blocks": multiple_result_blocks,
                    "result_block_valid": result_block_valid,
                    "used_stdout": used_stdout,
                    "used_result_block": used_result_block,
                },
            )
            continue

        generated_code_text = f"{PYTHON_BEGIN}\n{code_block}\n{PYTHON_END}"
        if code_block:
            try:
                _, execution_dict, _ = await llm.execute_generated_code(
                    prompt,
                    PYTHON_BEGIN,
                    PYTHON_END,
                    generated_code_text,
                    session_id=session_id,
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                _log(f"[TIR] Execution failed for {session_id}: {exc}")
                error_message = f"execution failed: {exc}"
                problem["output"] = "execution_error"
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
                        "has_python_block": has_python_block,
                        "has_result_block": has_result_block,
                        "multiple_python_blocks": multiple_python_blocks,
                        "multiple_result_blocks": multiple_result_blocks,
                        "result_block_valid": result_block_valid,
                        "used_stdout": used_stdout,
                        "used_result_block": used_result_block,
                    },
                )
                continue
            else:
                stdout = execution_dict.get("stdout", "") or ""
                stderr = execution_dict.get("stderr", "") or ""
                if stderr.strip():
                    _log(f"[TIR] stderr for {session_id}: {stderr.strip()}")

        if (
            "syntaxerror" in stderr.lower()
            or "traceback" in stderr.lower()
            or "syntaxerror" in stdout.lower()
            or "traceback" in stdout.lower()
        ):
            problem["output"] = "execution_error"
            error_message = (
                f"{error_message}; " if error_message else ""
            ) + "syntax error detected"
        else:
            answer, used_stdout, used_result_block = _derive_answer(
                output_text,
                stdout,
                result_block,
            )
            problem["output"] = answer

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
                "has_python_block": has_python_block,
                "has_result_block": has_result_block,
                "multiple_python_blocks": multiple_python_blocks,
                "multiple_result_blocks": multiple_result_blocks,
                "result_block_valid": result_block_valid,
                "used_stdout": used_stdout,
                "used_result_block": used_result_block,
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

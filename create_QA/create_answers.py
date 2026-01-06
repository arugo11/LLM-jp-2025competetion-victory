import argparse
import asyncio
import os
import re
import shlex
import signal
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_dataset
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.model import get_code_execution_model

PYTHON_BEGIN = "<PYTHON>"
PYTHON_END = "</PYTHON>"
PYTHON_OUTPUT_BEGIN = "<PYTHON_OUTPUT>"
PYTHON_OUTPUT_END = "</PYTHON_OUTPUT>"

PYTHON_BLOCK_RE = re.compile(
    re.escape(PYTHON_BEGIN) + r"(.*?)" + re.escape(PYTHON_END),
    re.DOTALL,
)

TIR_SYSTEM_PROMPT = textwrap.dedent(
    f"""
    Environment: ipython

    あなたは数学問題を解くための Python コードだけを生成します。
    出力は次の形式 **のみ** を厳守してください(他の文字・説明・Markdown・フェンスは禁止)

    {PYTHON_BEGIN}
    # sympy を使って厳密に計算し、最終解の LaTeX 文字列だけを 1 行で print する
    {PYTHON_END}

    ルール:
    - 出力の最初の行は必ず '{PYTHON_BEGIN}'、最後の行は必ず '{PYTHON_END}'。
    - print は 1 回だけ。最終解の LaTeX 文字列のみを出力する(余計なログ禁止)。
    - 可能な限り sympy の厳密計算(Rational など)を使う。
    - LaTeX は `latex = sympy.latex(expr).replace(\" \", \"\")` のように空白を除去してから出力する
    """,  # noqa: E501
).strip()

FALLBACK_PROMPT_TEMPLATE = """\
以下は数学の問題です。
解答を段階的に考え、最終的な答えとなる数値や解を\\boxedタグ内に記述してください。

# 制約事項
- 必ず最終的な解答を\\boxedタグ内に記述する。
- 最終的な解答は必ず一つの数値または数式で出力する。
- \\displaystyleを用いてはいけない。
- 最終的な解答では単位を出力してはならない。
- 数式は必ずlatex表記で出力する。

# 問題
{problem}

"""


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
    match = PYTHON_BLOCK_RE.search(generation)
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

        {PYTHON_BEGIN}
        {python_code}
        {PYTHON_END}

        {PYTHON_OUTPUT_BEGIN}
        {latex_value}
        {PYTHON_OUTPUT_END}

        Pythonの結果より、値は {latex_value} です。
        最終答\\boxed{{{latex_value}}}。
        """,
    ).strip()


def _http_get_status(url: str, timeout_sec: float = 1.0) -> Optional[int]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
            return getattr(resp, "status", 200)
    except urllib.error.URLError:
        return None


async def _wait_until_ready(
    url: str,
    timeout_sec: int,
    interval_sec: float = 1.0,
) -> bool:
    start = time.time()
    while time.time() - start < timeout_sec:
        status = await asyncio.to_thread(_http_get_status, url, 1.0)
        if status == 200:
            return True
        await asyncio.sleep(interval_sec)
    return False


@dataclass
class _StartedProcess:
    name: str
    process: subprocess.Popen[str]


def _terminate_process(proc: subprocess.Popen[str], *, name: str) -> None:
    if proc.poll() is not None:
        return

    # start_new_session=True で起動したプロセスは process group ごと止める
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return

    try:
        proc.wait(timeout=10)
        return
    except Exception:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            return

    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Math Answers (TIR)")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face input repository ID (dataset)",
    )
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token")
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Path to save the output dataset as JSONL",
    )

    # Sandbox
    parser.add_argument(
        "--tir-sandbox-type",
        type=str,
        default="local",
        help="TIR sandbox backend type",
    )
    parser.add_argument(
        "--tir-sandbox-host",
        type=str,
        default="127.0.0.1",
        help="TIR sandbox server host",
    )
    parser.add_argument(
        "--tir-sandbox-port",
        type=int,
        default=6000,
        help="TIR sandbox server port",
    )
    parser.add_argument(
        "--sandbox-block-network",
        action="store_true",
        help="Block network access inside the local sandbox server (recommended)",
    )

    # vLLM (OpenAI-compatible server)
    parser.add_argument(
        "--tir-llm-server-type",
        type=str,
        default="vllm",
        help="TIR LLM server backend type",
    )
    parser.add_argument(
        "--tir-llm-host",
        type=str,
        default="127.0.0.1",
        help="TIR LLM server host",
    )
    parser.add_argument(
        "--tir-llm-port",
        type=int,
        default=8000,
        help="TIR LLM server port",
    )
    parser.add_argument(
        "--tir-model-name",
        type=str,
        default="HayatoHongoEveryonesAI/llm-jp-4-8b-instruct",
        help="TIR served model name (OpenAI 'model' field)",
    )
    parser.add_argument(
        "--tir-max-retries",
        type=int,
        default=5,
        help="Max retries for TIR per problem",
    )
    parser.add_argument(
        "--tir-code-timeout",
        type=float,
        default=30.0,
        help="Sandbox code execution timeout (seconds)",
    )
    parser.add_argument(
        "--tir-max-output-chars",
        type=int,
        default=4000,
        help="Max sandbox stdout/stderr characters",
    )
    parser.add_argument(
        "--tir-temperature",
        type=float,
        default=0.2,
        help="Temperature for TIR code generation",
    )
    parser.add_argument(
        "--server-startup-timeout-sec",
        type=int,
        default=1800,
        help="Timeout seconds to wait for auto-started servers to become ready",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM --tensor-parallel-size for auto-started server",
    )
    parser.add_argument(
        "--vllm-extra-args",
        type=str,
        default="",
        help="Extra args passed to vLLM OpenAI server (shell-style string)",
    )

    return parser.parse_args()


async def _ensure_local_sandbox(
    args: argparse.Namespace,
    log_dir: Path,
) -> Optional[_StartedProcess]:
    health_url = f"http://{args.tir_sandbox_host}:{args.tir_sandbox_port}/health"
    if await _wait_until_ready(health_url, timeout_sec=2, interval_sec=0.2):
        return None

    log_dir.mkdir(parents=True, exist_ok=True)
    sandbox_log = (log_dir / "local_sandbox.log").open("a", encoding="utf-8")
    env = os.environ.copy()
    if args.sandbox_block_network:
        env["NEMO_SKILLS_SANDBOX_BLOCK_NETWORK"] = "1"

    proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "nemo_skills.code_execution.local_sandbox.local_sandbox_server",
        ],
        stdout=sandbox_log,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        start_new_session=True,
    )
    ok = await _wait_until_ready(
        health_url,
        timeout_sec=args.server_startup_timeout_sec,
        interval_sec=1.0,
    )
    if not ok:
        _terminate_process(proc, name="local_sandbox")
        raise RuntimeError(f"local sandbox did not become ready: {health_url}")
    return _StartedProcess(name="local_sandbox", process=proc)


async def _ensure_vllm_server(
    args: argparse.Namespace,
    log_dir: Path,
) -> Optional[_StartedProcess]:
    models_url = f"http://{args.tir_llm_host}:{args.tir_llm_port}/v1/models"
    if await _wait_until_ready(models_url, timeout_sec=2, interval_sec=0.2):
        return None

    log_dir.mkdir(parents=True, exist_ok=True)
    vllm_log = (log_dir / "vllm_server.log").open("a", encoding="utf-8")

    cmd: list[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model_path,
        "--host",
        args.tir_llm_host,
        "--port",
        str(args.tir_llm_port),
        "--served-model-name",
        args.tir_model_name,
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(args.vllm_tensor_parallel_size),
        "--disable-log-requests",
        "--disable-log-stats",
    ]
    if args.vllm_extra_args:
        cmd += shlex.split(args.vllm_extra_args)

    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=vllm_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    ok = await _wait_until_ready(
        models_url,
        timeout_sec=args.server_startup_timeout_sec,
        interval_sec=2.0,
    )
    if not ok:
        _terminate_process(proc, name="vllm_server")
        raise RuntimeError(f"vLLM server did not become ready: {models_url}")
    return _StartedProcess(name="vllm_server", process=proc)


async def _fallback_solve(args: argparse.Namespace, llm, problem: str) -> str:
    prompt = [
        {"role": "user", "content": FALLBACK_PROMPT_TEMPLATE.format(problem=problem)},
    ]
    try:
        result = await llm.model.generate_async(
            prompt=prompt,
            tokens_to_generate=args.max_tokens,
            temperature=0.0,
        )
    except Exception as exc:  # pragma: no cover
        return f"fallback_error: {exc}"
    return (result.get("generation") or "").strip()


async def _generate_tir_row(
    args: argparse.Namespace,
    llm,
    row: dict[str, Any],
) -> dict[str, Any]:
    new_row: dict[str, Any] = dict(row)
    problem = str(row.get("problem", "") or "")
    problem_one_line = _normalize_one_line(problem)

    new_row["tir_attempts"] = 0
    new_row["fallback_used"] = False
    new_row["tir_status"] = ""
    new_row["llm-code"] = None
    new_row["raw_generation"] = None
    new_row["output"] = None
    new_row["execution_output"] = None
    new_row["generated_solution"] = ""
    new_row["expected_answer"] = None

    last_code: Optional[str] = None
    last_exec_stderr: Optional[str] = None

    tir_prompt = [
        {"role": "system", "content": TIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"次の数学問題を解いてください。出力は <PYTHON>...</PYTHON> のみ。\n\n{problem}\n",
        },
    ]

    for attempt in range(1, max(args.tir_max_retries, 1) + 1):
        new_row["tir_attempts"] = attempt

        try:
            generation_result = await llm.generate_async(
                prompt=tir_prompt,
                code_begin=PYTHON_BEGIN,
                code_end=PYTHON_END,
                code_output_begin=PYTHON_OUTPUT_BEGIN,
                code_output_end=PYTHON_OUTPUT_END,
                code_output_format="qwen",
                tokens_to_generate=args.max_tokens,
                temperature=args.tir_temperature,
                stop_phrases=[PYTHON_END],
                max_code_executions=0,
                remove_stop_phrases=False,
            )
        except Exception:  # pragma: no cover
            continue

        raw_generation_text = str(generation_result.get("generation") or "")
        new_row["raw_generation"] = raw_generation_text
        generation_text = raw_generation_text.strip()
        if generation_text and generation_text.count(
            PYTHON_BEGIN,
        ) > generation_text.count(PYTHON_END):
            generation_text += f"\n{PYTHON_END}"

        code_block = _extract_python_code(generation_text)
        if not code_block:
            continue

        last_code = code_block
        new_row["llm-code"] = last_code

        generated_code_text = f"{PYTHON_BEGIN}\n{code_block}\n{PYTHON_END}"
        session_id = None
        try:
            _, execution_dict, session_id = await llm.execute_generated_code(
                tir_prompt,
                PYTHON_BEGIN,
                PYTHON_END,
                generated_code_text,
                session_id=None,
            )
        except Exception:  # pragma: no cover
            continue
        finally:
            try:
                if session_id is not None:
                    await llm.sandbox.delete_session(str(session_id))
            except Exception:
                pass

        process_status = str(execution_dict.get("process_status", "") or "")
        stdout = str(execution_dict.get("stdout", "") or "")
        stderr = str(execution_dict.get("stderr", "") or "")
        last_exec_stderr = stderr

        if not _tir_success(process_status, stdout, stderr):
            continue

        latex_value = _last_non_empty_line(stdout) or ""
        generated_solution_text = _format_tir_solution(
            problem_one_line,
            code_block,
            latex_value,
        )

        new_row["generated_solution"] = generated_solution_text
        new_row["expected_answer"] = extract_answer(generated_solution_text)
        new_row["execution_output"] = stdout if stdout else None
        new_row["output"] = generated_solution_text if generated_solution_text else None
        new_row["tir_status"] = "tir_success"
        new_row["fallback_used"] = False
        return new_row

    # TIR failed -> fallback
    new_row["tir_status"] = "tir_failed_fallback_used"
    new_row["fallback_used"] = True
    new_row["llm-code"] = last_code
    new_row["execution_output"] = last_exec_stderr if (last_exec_stderr or "") else None

    fallback_text = await _fallback_solve(args, llm, problem)
    new_row["generated_solution"] = fallback_text
    new_row["output"] = fallback_text if fallback_text else None
    new_row["expected_answer"] = extract_answer(fallback_text)
    if new_row["expected_answer"] is None:
        new_row["tir_status"] = "fallback_failed"
    return new_row


async def _run(args: argparse.Namespace) -> None:
    program_start_time = time.time()

    input_repo_id = args.repo_id
    output_repo_id = f"{input_repo_id}-TIR"
    # サーバログはホスト側で参照できるよう、可能なら output 配下へ出す
    if args.output_jsonl:
        log_dir = Path(args.output_jsonl).parent / ".log"
    else:
        log_dir = Path("output") / ".log"
    log_dir.mkdir(parents=True, exist_ok=True)

    started: list[_StartedProcess] = []

    sandbox_proc = await _ensure_local_sandbox(args, log_dir)
    if sandbox_proc:
        started.append(sandbox_proc)
    vllm_proc = await _ensure_vllm_server(args, log_dir)
    if vllm_proc:
        started.append(vllm_proc)

    def _cleanup() -> None:
        for sp in reversed(started):
            _terminate_process(sp.process, name=sp.name)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_a: _cleanup())

    try:
        print(f"Downloading dataset from {input_repo_id}...")
        dataset = load_dataset(input_repo_id, split="train")

        sandbox = get_sandbox(
            sandbox_type=args.tir_sandbox_type,
            host=args.tir_sandbox_host,
            port=args.tir_sandbox_port,
        )
        llm = get_code_execution_model(
            server_type=args.tir_llm_server_type,
            host=args.tir_llm_host,
            port=args.tir_llm_port,
            model=args.tir_model_name,
            sandbox=sandbox,
            code_execution={
                "code_execution_timeout": args.tir_code_timeout,
                "max_code_output_characters": args.tir_max_output_chars,
            },
        )

        inference_start_time = time.time()
        data: list[dict[str, Any]] = []
        for row in dataset:
            data.append(await _generate_tir_row(args, llm, row))
        inference_finish_time = time.time()
        print(f"Inference time: {inference_finish_time - inference_start_time}(s)")

        dataset_dict = DatasetDict()
        new_dataset = Dataset.from_list(data)
        dataset_dict["train"] = new_dataset

        if args.output_jsonl:
            new_dataset.to_json(
                args.output_jsonl,
                orient="records",
                lines=True,
                force_ascii=False,
            )
            print(f"Saved dataset to {args.output_jsonl}")

        if args.hf_token:
            dataset_dict.push_to_hub(output_repo_id, token=args.hf_token)
            print(f"Uploaded dataset to {output_repo_id}")
        else:
            print("HF token not provided. Skipping upload.")

        program_finish_time = time.time()
        print(f"Total time: {program_finish_time - program_start_time}(s)")
    finally:
        _cleanup()


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

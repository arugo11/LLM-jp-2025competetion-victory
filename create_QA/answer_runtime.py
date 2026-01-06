import argparse
import asyncio
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import answer_prompts as ap
import answer_utils as au


@dataclass
class _StartedProcess:
    name: str
    process: subprocess.Popen[str]


def _terminate_process(proc: subprocess.Popen[str], *, name: str) -> None:
    """sandboxが経っていなければ起動する"""
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
        {
            "role": "user",
            "content": ap.FALLBACK_PROMPT_TEMPLATE.format(problem=problem),
        },
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
    problem_one_line = au._normalize_one_line(problem)

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
        {"role": "system", "content": ap.TIR_SYSTEM_PROMPT},
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
                code_begin=ap.PYTHON_BEGIN,
                code_end=ap.PYTHON_END,
                code_output_begin=ap.PYTHON_OUTPUT_BEGIN,
                code_output_end=ap.PYTHON_OUTPUT_END,
                code_output_format="qwen",
                tokens_to_generate=args.max_tokens,
                temperature=args.tir_temperature,
                stop_phrases=[ap.PYTHON_END],
                max_code_executions=0,
                remove_stop_phrases=False,
            )
        except Exception:  # pragma: no cover
            continue

        raw_generation_text = str(generation_result.get("generation") or "")
        new_row["raw_generation"] = raw_generation_text
        generation_text = raw_generation_text.strip()
        if generation_text and generation_text.count(
            ap.PYTHON_BEGIN,
        ) > generation_text.count(ap.PYTHON_END):
            generation_text += f"\n{ap.PYTHON_END}"

        code_block = au._extract_python_code(generation_text)
        if not code_block:
            continue

        last_code = code_block
        new_row["llm-code"] = last_code

        generated_code_text = f"{ap.PYTHON_BEGIN}\n{code_block}\n{ap.PYTHON_END}"
        session_id = None
        try:
            _, execution_dict, session_id = await llm.execute_generated_code(
                tir_prompt,
                ap.PYTHON_BEGIN,
                ap.PYTHON_END,
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

        if not au._tir_success(process_status, stdout, stderr):
            continue

        latex_value = au._last_non_empty_line(stdout) or ""
        generated_solution_text = au._format_tir_solution(
            problem_one_line,
            code_block,
            latex_value,
        )

        new_row["generated_solution"] = generated_solution_text
        new_row["expected_answer"] = au.extract_answer(generated_solution_text)
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
    new_row["expected_answer"] = au.extract_answer(fallback_text)
    if new_row["expected_answer"] is None:
        new_row["tir_status"] = "fallback_failed"
    return new_row

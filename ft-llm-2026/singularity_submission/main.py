"""Singularity submission entry point for solving math problems via vLLM + NeMo-Skills sandbox."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from executor import (
    execute_code_safe,
    generate_once,
    is_execution_error,
    wait_for_llm_ready,
)
from nemo_skills.code_execution.sandbox import get_sandbox  # type: ignore
from nemo_skills.inference.model import get_model  # type: ignore
from prompts import (
    PYTHON_BEGIN,
    PYTHON_END,
    RESULT_BEGIN,
    RESULT_END,
    build_messages,
    build_repair_messages,
)
from utils import (
    SolveResult,
    TraceLogger,
    extract_blocks,
    extract_code_emergency,
    last_non_empty_line,
    read_problems,
    safe_exec_view,
)
from wandb_tracer import WeaveConfig, WeaveTracer, init_tracer

# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Singularity Submission Example")

    # I/O paths
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--log_path", type=Path, default=Path("inference_log.jsonl"))
    parser.add_argument(
        "--trace_log_path",
        type=Path,
        default=Path("inference_trace.jsonl"),
    )
    parser.add_argument("--trace_max_chars", type=int, default=8000)
    parser.add_argument("--log-raw-output", action="store_true")
    parser.add_argument("--repair-attempts", type=int, default=1)

    # Sandbox configuration
    parser.add_argument("--tir-sandbox-type", default="local")
    parser.add_argument("--tir-sandbox-host", default="127.0.0.1")
    parser.add_argument("--tir-sandbox-port", type=int, default=6000)

    # LLM server configuration
    parser.add_argument("--tir-llm-server-type", default="vllm")
    parser.add_argument("--tir-llm-host", default="127.0.0.1")
    parser.add_argument("--tir-llm-port", type=int, default=8000)
    parser.add_argument("--tir-model-name", type=str, default=None)
    parser.add_argument("--llm-ready-timeout", type=float, default=300.0)
    parser.add_argument("--llm-ready-interval", type=float, default=2.0)

    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retry-temperature", type=float, default=None)

    # Code execution parameters
    parser.add_argument("--code-language", type=str, default="ipython")
    parser.add_argument("--code-timeout", type=float, default=10.0)
    parser.add_argument("--max-output-chars", type=int, default=1000)

    # W&B Weave configuration
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Enable W&B Weave tracing (requires weave package)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="llm-jp-math-tir",
        help="W&B Weave project name",
    )
    parser.add_argument(
        "--wandb-disabled",
        action="store_true",
        help="Initialize Weave but disable sending (for debugging)",
    )

    return parser.parse_args()


# =============================================================================
# Problem Solver
# =============================================================================


class ProblemSolver:
    """Handles solving individual math problems."""

    def __init__(
        self,
        llm,
        sandbox,
        args: argparse.Namespace,
        trace: TraceLogger,
        weave_tracer: WeaveTracer | None = None,
    ):
        self.llm = llm
        self.sandbox = sandbox
        self.args = args
        self.trace = trace
        self.weave_tracer = weave_tracer
        self._event_i = 0
        self._weave_client: Any = None

        # Get Weave client for manual call tracking
        if weave_tracer and weave_tracer.enabled:
            import weave

            self._weave_client = weave

    async def solve(self, problem: dict[str, Any], idx: int) -> SolveResult:
        """Solve a single math problem with optional Weave tracing."""
        question = str(problem.get("problem", ""))
        session_id = str(problem.get("id", idx))

        # If Weave is enabled, wrap the solve operation
        if self._weave_client and self.weave_tracer:
            return await self._solve_with_weave_trace(
                problem, idx, question, session_id
            )

        return await self._solve_core(problem, idx, question, session_id)

    async def _solve_with_weave_trace(
        self,
        problem: dict[str, Any],
        idx: int,
        question: str,
        session_id: str,
    ) -> SolveResult:
        """Execute solve with Weave tracing."""
        weave = self._weave_client

        # Create the traced solve function dynamically
        @weave.op(name="solve_problem")
        async def traced_solve(
            question: str,
            session_id: str,
        ) -> dict[str, Any]:
            result = await self._solve_core(problem, idx, question, session_id)
            # Store result for return, output structured data for Weave
            traced_solve._result = result  # type: ignore
            return self.weave_tracer.build_solve_output(result)  # type: ignore

        await traced_solve(question, session_id)
        return traced_solve._result  # type: ignore

    async def _solve_core(
        self,
        problem: dict[str, Any],
        idx: int,
        question: str,
        session_id: str,
    ) -> SolveResult:
        """Core solve implementation."""
        t0 = time.time()
        self._event_i = 0

        temperatures = [self.args.temperature]
        if self.args.retry_temperature is not None:
            temperatures.append(self.args.retry_temperature)

        raw_output = ""
        python_blocks: list[str] = []
        result_blocks: list[str] = []
        exec_dict: dict[str, Any] | None = None
        used_temp = self.args.temperature
        repair_used = 0

        messages = build_messages(question)
        self._log_event(session_id, "input", question=self.trace.clip(question))

        # Initial generation + execution
        for attempt, temp in enumerate(temperatures, start=1):
            used_temp = temp
            raw_output, python_blocks, result_blocks = await self._generate(
                session_id,
                "initial",
                attempt,
                temp,
                messages,
            )

            if not python_blocks:
                self._log_event(
                    session_id,
                    "decision",
                    phase="initial",
                    decision="no_python_block",
                )
                if attempt < len(temperatures):
                    continue
                break

            exec_dict = await self._execute(
                session_id,
                "initial",
                attempt,
                python_blocks[0],
            )

            if is_execution_error(exec_dict):
                self._log_event(
                    session_id,
                    "decision",
                    phase="initial",
                    decision="execution_error",
                    will_retry=attempt < len(temperatures),
                )
                if attempt < len(temperatures):
                    continue
            break

        # Repair phase
        for r in range(max(0, self.args.repair_attempts)):
            if not python_blocks or not is_execution_error(exec_dict):
                break

            stdout = exec_dict.get("stdout", "") if exec_dict else ""
            stderr = exec_dict.get("stderr", "") if exec_dict else ""

            repair_messages = build_repair_messages(
                question,
                python_blocks[0],
                stdout,
                stderr,
            )

            repair_temp = (
                self.args.retry_temperature
                if self.args.retry_temperature is not None
                else max(self.args.temperature, 0.2)
            )
            used_temp = repair_temp

            raw_output, python_blocks, result_blocks = await self._generate(
                session_id,
                "repair",
                r + 1,
                repair_temp,
                repair_messages,
            )

            if not python_blocks:
                exec_dict = None
                repair_used += 1
                self._log_event(
                    session_id,
                    "decision",
                    phase="repair",
                    repair_round=r + 1,
                    decision="no_python_block",
                )
                continue

            exec_dict = await self._execute(
                session_id,
                "repair",
                r + 1,
                python_blocks[0],
            )
            repair_used += 1

        # Build result
        result = await self._build_result(
            session_id,
            raw_output,
            python_blocks,
            result_blocks,
            exec_dict,
            t0,
            repair_used,
            used_temp,
        )
        return result

    async def _generate(
        self,
        session_id: str,
        phase: str,
        attempt: int,
        temperature: float,
        messages: list[dict[str, str]],
    ) -> tuple[str, list[str], list[str]]:
        """Generate code from LLM with optional Weave tracing."""
        # If Weave is enabled, wrap the generate operation
        if self._weave_client and self.weave_tracer:
            return await self._generate_with_weave_trace(
                session_id,
                phase,
                attempt,
                temperature,
                messages,
            )
        return await self._generate_core(
            session_id,
            phase,
            attempt,
            temperature,
            messages,
        )

    async def _generate_with_weave_trace(
        self,
        session_id: str,
        phase: str,
        attempt: int,
        temperature: float,
        messages: list[dict[str, str]],
    ) -> tuple[str, list[str], list[str]]:
        """Generate with Weave tracing."""
        weave = self._weave_client

        @weave.op(name="llm_generate")
        async def traced_generate(
            phase: str,
            attempt: int,
            temperature: float,
        ) -> dict[str, Any]:
            t0 = time.time()
            raw, py_blocks, res_blocks = await self._generate_core(
                session_id,
                phase,
                attempt,
                temperature,
                messages,
            )
            traced_generate._result = (raw, py_blocks, res_blocks)  # type: ignore
            return self.weave_tracer.build_generate_output(  # type: ignore
                raw,
                py_blocks,
                res_blocks,
                time.time() - t0,
            )

        await traced_generate(phase, attempt, temperature)
        return traced_generate._result  # type: ignore

    async def _generate_core(
        self,
        session_id: str,
        phase: str,
        attempt: int,
        temperature: float,
        messages: list[dict[str, str]],
    ) -> tuple[str, list[str], list[str]]:
        """Core generate implementation."""
        self._log_event(
            session_id,
            "llm_request",
            phase=phase,
            attempt=attempt,
            temperature=temperature,
        )

        t0 = time.time()
        raw_output = await generate_once(
            self.llm,
            messages,
            temperature=temperature,
            max_new_tokens=self.args.max_new_tokens,
            min_tokens=self.args.min_tokens,
        )

        self._log_event(
            session_id,
            "llm_response",
            phase=phase,
            attempt=attempt,
            latency_s=time.time() - t0,
            raw_output=self.trace.clip(raw_output),
        )

        python_blocks = extract_blocks(raw_output, PYTHON_BEGIN, PYTHON_END)
        result_blocks = extract_blocks(raw_output, RESULT_BEGIN, RESULT_END)

        self._log_event(
            session_id,
            "parse",
            phase=phase,
            attempt=attempt,
            n_python=len(python_blocks),
            n_result=len(result_blocks),
        )

        return raw_output, python_blocks, result_blocks

    async def _execute(
        self,
        session_id: str,
        phase: str,
        attempt: int,
        code: str,
    ) -> dict[str, Any] | None:
        """Execute code in sandbox with optional Weave tracing."""
        # If Weave is enabled, wrap the execute operation
        if self._weave_client and self.weave_tracer:
            return await self._execute_with_weave_trace(
                session_id,
                phase,
                attempt,
                code,
            )
        return await self._execute_core(session_id, phase, attempt, code)

    async def _execute_with_weave_trace(
        self,
        session_id: str,
        phase: str,
        attempt: int,
        code: str,
    ) -> dict[str, Any] | None:
        """Execute with Weave tracing."""
        weave = self._weave_client

        @weave.op(name="sandbox_execute")
        async def traced_execute(
            phase: str,
            attempt: int,
            code_preview: str,
        ) -> dict[str, Any]:
            t0 = time.time()
            result = await self._execute_core(session_id, phase, attempt, code)
            traced_execute._result = result  # type: ignore
            return self.weave_tracer.build_execute_output(  # type: ignore
                result,
                time.time() - t0,
            )

        # Clip code for display in Weave UI
        code_preview = (
            self.weave_tracer.format_code_markdown(
                self.trace.clip(code),
            )
            if self.weave_tracer
            else code[:500]
        )

        await traced_execute(phase, attempt, code_preview)
        return traced_execute._result  # type: ignore

    async def _execute_core(
        self,
        session_id: str,
        phase: str,
        attempt: int,
        code: str,
    ) -> dict[str, Any] | None:
        """Core execute implementation."""
        self._log_event(
            session_id,
            "sandbox_request",
            phase=phase,
            attempt=attempt,
            code=self.trace.clip(code),
        )

        t0 = time.time()
        exec_dict, exception = await execute_code_safe(
            self.sandbox,
            code,
            self.args.code_language,
            self.args.code_timeout,
            self.args.max_output_chars,
        )

        if exception:
            self._log_event(
                session_id,
                "sandbox_exception",
                phase=phase,
                attempt=attempt,
                latency_s=time.time() - t0,
                exception=repr(exception),
            )
        else:
            self._log_event(
                session_id,
                "sandbox_response",
                phase=phase,
                attempt=attempt,
                latency_s=time.time() - t0,
                is_error=is_execution_error(exec_dict),
                execution=safe_exec_view(exec_dict, self.trace.max_chars),
            )

        return exec_dict

    async def _build_result(
        self,
        session_id: str,
        raw_output: str,
        python_blocks: list[str],
        result_blocks: list[str],
        exec_dict: dict[str, Any] | None,
        t0: float,
        repair_used: int,
        used_temp: float,
    ) -> SolveResult:
        """Build final result from execution."""
        has_python = bool(python_blocks)
        stdout = exec_dict.get("stdout", "") if exec_dict else ""
        stderr = exec_dict.get("stderr", "") if exec_dict else ""

        error = ""
        used_stdout = False
        used_emergency = False
        exec_success = not is_execution_error(exec_dict) if exec_dict else False
        result_mismatch = False
        final_output = ""

        # Emergency extraction if no python block
        if not has_python:
            emergency_code = extract_code_emergency(raw_output)
            if emergency_code:
                exec_dict, _ = await execute_code_safe(
                    self.sandbox,
                    emergency_code,
                    self.args.code_language,
                    self.args.code_timeout,
                    self.args.max_output_chars,
                )
                used_emergency = True
                exec_success = not is_execution_error(exec_dict)
                stdout = exec_dict.get("stdout", "") if exec_dict else ""
                stderr = exec_dict.get("stderr", "") if exec_dict else ""

            if not exec_success:
                error = "no_python_block"
                final_output = "ERROR: no executable code found"

        if not error and is_execution_error(exec_dict):
            error = "execution_error"
            final_output = "ERROR: execution failed"

        if not error:
            last_line = last_non_empty_line(stdout)
            if last_line:
                final_output = last_line
                used_stdout = True
                if result_blocks and result_blocks[0].strip() != last_line:
                    result_mismatch = True
            else:
                error = "empty_stdout"
                final_output = "ERROR: execution succeeded but no output"

        self._log_event(
            session_id,
            "final",
            elapsed_s=time.time() - t0,
            error=error,
            final_output=self.trace.clip(final_output),
            used_stdout=used_stdout,
            repair_used=repair_used,
            used_emergency=used_emergency,
            exec_success=exec_success,
        )

        return SolveResult(
            output=final_output,
            session_id=session_id,
            error=error,
            has_python_block=has_python,
            multiple_python_blocks=len(python_blocks) > 1,
            used_stdout=used_stdout,
            used_emergency_extract=used_emergency,
            parse_success=has_python,
            exec_success=exec_success,
            result_vs_stdout_mismatch=result_mismatch,
            temperature=used_temp,
            python_block=python_blocks[0] if python_blocks else "",
            result_block=result_blocks[0] if result_blocks else "",
            stdout=stdout,
            stderr=stderr,
            repair_used=repair_used,
            raw_output=raw_output,
        )

    def _log_event(self, session_id: str, event: str, **kwargs: Any) -> None:
        """Log a trace event."""
        self.trace.write(
            {
                "id": session_id,
                "event_i": self._event_i,
                "event": event,
                "ts": time.time(),
                **kwargs,
            },
        )
        self._event_i += 1


# =============================================================================
# Pipeline
# =============================================================================


async def run_pipeline(args: argparse.Namespace) -> None:
    """Run the complete math problem solving pipeline."""
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input path not found: {args.input_path}")

    # Initialize Weave tracer (if enabled)
    weave_tracer: WeaveTracer | None = None
    if args.enable_wandb:
        weave_config = WeaveConfig(
            enabled=True,
            project=args.wandb_project,
            disabled_send=args.wandb_disabled,
        )
        weave_tracer = init_tracer(weave_config)

    # vLLMのサーバが起動するのを待機する.
    await wait_for_llm_ready(
        host=args.tir_llm_host,
        port=args.tir_llm_port,
        timeout=args.llm_ready_timeout,
        interval=args.llm_ready_interval,
    )

    problems = read_problems(args.input_path)
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

    for path in [args.output_path, args.log_path, args.trace_log_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with (
            args.output_path.open("w", encoding="utf-8") as out_f,
            args.log_path.open("a", encoding="utf-8") as log_f,
            args.trace_log_path.open("a", encoding="utf-8") as trace_f,
        ):
            trace = TraceLogger(trace_f, args.trace_max_chars)
            solver = ProblemSolver(llm, sandbox, args, trace, weave_tracer)

            for idx, problem in enumerate(problems):
                result = await solver.solve(problem, idx)
                problem["output"] = result.output
                out_f.write(json.dumps(problem, ensure_ascii=False) + "\n")
                log_f.write(
                    json.dumps(
                        result.to_log_entry(args.log_raw_output),
                        ensure_ascii=False,
                    )
                    + "\n",
                )
    finally:
        await sandbox.close()


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main entry point."""
    args = parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()

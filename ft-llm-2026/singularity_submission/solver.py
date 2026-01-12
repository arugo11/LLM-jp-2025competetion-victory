"""LLM + サンドボックスを使用した数学問題ソルバー."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from config import SolverConfig
from executor import execute_code_safe, generate_once, is_execution_error
from prompts import (
    PYTHON_BEGIN,
    PYTHON_END,
    RESULT_BEGIN,
    RESULT_END,
    build_direct_answer_messages,
    build_format_repair_messages,
    build_messages,
    build_repair_messages,
)
from utils import (
    SolveResult,
    extract_blocks,
    extract_code_emergency,
    last_non_empty_line,
    safe_exec_view,
    to_latex_scalar,
)

if TYPE_CHECKING:
    from wandb_tracer import WeaveTracer


# LLM生成結果


@dataclass
class GenerationResult:
    """LLM生成の結果."""

    raw_output: str
    python_blocks: list[str]
    result_blocks: list[str]
    tokens: dict[str, int] | None = None

    @property
    def has_python(self) -> bool:
        """Pythonコードが含まれているか."""
        return bool(self.python_blocks)


# 問題ソルバー


class ProblemSolver:
    """LLM + コード実行を使用した数学問題ソルバー."""

    def __init__(
        self,
        llm: Any,
        sandbox: Any,
        config: SolverConfig,
        weave_tracer: WeaveTracer | None = None,
    ) -> None:
        """ソルバーを初期化."""
        self.llm = llm
        self.sandbox = sandbox
        self.config = config
        self.weave_tracer = weave_tracer

        self._event_i = 0
        self._session_id = ""

        self._weave_client: Any = None
        if weave_tracer and weave_tracer.enabled:
            import weave

            self._weave_client = weave

    def _clip(self, text: str) -> str:
        """ログ用にテキストを短縮."""
        max_chars = getattr(self.config, "trace_max_chars", 8000)
        return text if len(text) <= max_chars else text[:max_chars]

    # 公開API

    async def solve(self, problem: dict[str, Any], idx: int) -> SolveResult:
        """単一の問題を解く."""
        question = str(problem.get("problem", ""))
        self._session_id = str(problem.get("id", idx))
        if self._weave_client and self.weave_tracer:
            return await self._solve_with_weave(question)
        return await self._solve_impl(question)

    # 解決処理の実装

    async def _solve_with_weave(self, question: str) -> SolveResult:
        """Weaveトレース付きで解決を実行."""
        weave = self._weave_client

        @weave.op(name="solve_problem")
        async def traced_solve(question: str, session_id: str) -> dict[str, Any]:
            result = await self._solve_impl(question)
            traced_solve._result = result  # type: ignore
            return self.weave_tracer.build_solve_output(result)  # type: ignore

        await traced_solve(question, self._session_id)
        return traced_solve._result  # type: ignore

    async def _solve_impl(self, question: str) -> SolveResult:
        """完全な解決フローをオーケストレート.

        1. 初回生成（温度バリアント）
        2. フォーマット修正リトライ
        3. リペアフェーズ
        4. 直接回答フォールバック
        """
        t0 = time.time()
        self._event_i = 0

        gen_result = GenerationResult("", [], [])
        exec_dict: dict[str, Any] | None = None
        used_temp = self.config.temperature
        repair_used = 0

        messages = build_messages(question)

        # フェーズ1: 初回生成
        gen_result, exec_dict, used_temp = await self._phase_initial_generation(
            messages,
        )

        # フェーズ2: フォーマット修正
        if not gen_result.has_python and self.config.format_retry_attempts > 0:
            gen_result, exec_dict = await self._phase_format_fix(
                question,
                gen_result.raw_output,
                used_temp,
            )

        # フェーズ3: リペアフェーズ
        if gen_result.has_python and is_execution_error(exec_dict):
            gen_result, exec_dict, repair_used = await self._phase_repair(
                question,
                gen_result,
                exec_dict,
            )
            used_temp = self.config.get_repair_temperature()

        # フェーズ4: 直接回答フォールバック
        if self._should_use_direct_fallback(gen_result, exec_dict):
            gen_result, exec_dict = await self._phase_direct_answer(
                question,
                used_temp,
            )

        return await self._build_result(
            question=question,
            gen_result=gen_result,
            exec_dict=exec_dict,
            t0=t0,
            repair_used=repair_used,
            used_temp=used_temp,
        )

    # 解決フェーズ

    async def _phase_initial_generation(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[GenerationResult, dict[str, Any] | None, float]:
        """フェーズ1: 温度バリアントでの初回生成."""
        temperatures = self.config.get_temperature_sequence()
        gen_result = GenerationResult("", [], [])
        exec_dict: dict[str, Any] | None = None
        used_temp = temperatures[0]

        for attempt, temp in enumerate(temperatures, start=1):
            used_temp = temp
            gen_result = await self._generate(
                phase="initial",
                attempt=attempt,
                temperature=temp,
                messages=messages,
            )

            if not gen_result.has_python:
                # decision: no_python_block
                if attempt < len(temperatures):
                    continue
                break

            exec_dict = await self._execute(
                phase="initial",
                attempt=attempt,
                code=gen_result.python_blocks[0],
            )

            if is_execution_error(exec_dict):
                # decision: execution_error
                if attempt < len(temperatures):
                    continue
            break

        return gen_result, exec_dict, used_temp

    async def _phase_format_fix(
        self,
        question: str,
        raw_output: str,
        temperature: float,
    ) -> tuple[GenerationResult, dict[str, Any] | None]:
        """フェーズ2: タグ欠落時のフォーマット修正."""
        gen_result = GenerationResult(raw_output, [], [])
        exec_dict: dict[str, Any] | None = None

        for attempt in range(1, self.config.format_retry_attempts + 1):
            format_messages = build_format_repair_messages(question, raw_output)
            gen_result = await self._generate(
                phase="format_fix",
                attempt=attempt,
                temperature=temperature,
                messages=format_messages,
            )

            if gen_result.has_python:
                exec_dict = await self._execute(
                    phase="format_fix",
                    attempt=attempt,
                    code=gen_result.python_blocks[0],
                )
                if not is_execution_error(exec_dict):
                    break

        return gen_result, exec_dict

    async def _phase_repair(
        self,
        question: str,
        gen_result: GenerationResult,
        exec_dict: dict[str, Any] | None,
    ) -> tuple[GenerationResult, dict[str, Any] | None, int]:
        """フェーズ3: 実行エラー後のコード修復."""
        repair_used = 0
        repair_temp = self.config.get_repair_temperature()

        for attempt in range(1, self.config.repair_attempts + 1):
            if not gen_result.has_python or not is_execution_error(exec_dict):
                break

            stdout = exec_dict.get("stdout", "") if exec_dict else ""
            stderr = exec_dict.get("stderr", "") if exec_dict else ""

            repair_messages = build_repair_messages(
                question,
                gen_result.python_blocks[0],
                stdout,
                stderr,
            )

            gen_result = await self._generate(
                phase="repair",
                attempt=attempt,
                temperature=repair_temp,
                messages=repair_messages,
            )
            repair_used += 1

            if not gen_result.has_python:
                self._log_event(
                    "decision",
                    phase="repair",
                    repair_round=attempt,
                    decision="no_python_block",
                )
                exec_dict = None
                continue

            exec_dict = await self._execute(
                phase="repair",
                attempt=attempt,
                code=gen_result.python_blocks[0],
            )

        return gen_result, exec_dict, repair_used

    async def _phase_direct_answer(
        self,
        question: str,
        temperature: float,
    ) -> tuple[GenerationResult, dict[str, Any] | None]:
        """フェーズ4: コード実行なしの直接回答."""
        gen_result = GenerationResult("", [], [])

        for attempt in range(1, self.config.direct_answer_attempts + 1):
            direct_messages = build_direct_answer_messages(question)
            gen_result = await self._generate(
                phase="direct_answer",
                attempt=attempt,
                temperature=temperature,
                messages=direct_messages,
            )

            if gen_result.result_blocks:
                break

            last_line = last_non_empty_line(gen_result.raw_output)
            if last_line:
                gen_result = GenerationResult(
                    gen_result.raw_output,
                    [],
                    [last_line],
                )
                break

        return gen_result, None

    def _should_use_direct_fallback(
        self,
        gen_result: GenerationResult,
        exec_dict: dict[str, Any] | None,
    ) -> bool:
        """直接回答フォールバックを使用すべきか."""
        if self.config.direct_answer_attempts <= 0:
            return False
        return not gen_result.has_python or is_execution_error(exec_dict)

    # LLM生成

    async def _generate(
        self,
        phase: str,
        attempt: int,
        temperature: float,
        messages: list[dict[str, str]],
    ) -> GenerationResult:
        """LLMからコードを生成."""
        if self._weave_client and self.weave_tracer:
            return await self._generate_with_weave(
                phase,
                attempt,
                temperature,
                messages,
            )
        return await self._generate_impl(phase, attempt, temperature, messages)

    async def _generate_with_weave(
        self,
        phase: str,
        attempt: int,
        temperature: float,
        messages: list[dict[str, str]],
    ) -> GenerationResult:
        """Weaveトレース付きで生成."""
        weave = self._weave_client

        @weave.op(name="llm_generate")
        async def traced_generate(
            phase: str,
            attempt: int,
            temperature: float,
        ) -> dict[str, Any]:
            t0 = time.time()
            result = await self._generate_impl(phase, attempt, temperature, messages)
            traced_generate._result = result  # type: ignore
            return self.weave_tracer.build_generate_output(  # type: ignore
                result.raw_output,
                result.python_blocks,
                result.result_blocks,
                time.time() - t0,
                result.tokens or {},
            )

        await traced_generate(phase, attempt, temperature)
        return traced_generate._result  # type: ignore

    async def _generate_impl(
        self,
        phase: str,
        attempt: int,
        temperature: float,
        messages: list[dict[str, str]],
    ) -> GenerationResult:
        """生成の実装."""
        # llm_request

        t0 = time.time()
        gen = await generate_once(
            self.llm,
            messages,
            temperature=temperature,
            max_new_tokens=self.config.max_new_tokens,
            min_tokens=self.config.min_tokens,
        )
        raw_output = gen.get("generation", "")
        tokens: dict[str, int] | None = gen.get("usage") or None

        # llm_response

        python_blocks = extract_blocks(raw_output, PYTHON_BEGIN, PYTHON_END)
        result_blocks = extract_blocks(raw_output, RESULT_BEGIN, RESULT_END)

        # parse

        return GenerationResult(raw_output, python_blocks, result_blocks, tokens)

    # コード実行

    async def _execute(
        self,
        phase: str,
        attempt: int,
        code: str,
    ) -> dict[str, Any] | None:
        """サンドボックスでコードを実行."""
        if self._weave_client and self.weave_tracer:
            return await self._execute_with_weave(phase, attempt, code)
        return await self._execute_impl(phase, attempt, code)

    async def _execute_with_weave(
        self,
        phase: str,
        attempt: int,
        code: str,
    ) -> dict[str, Any] | None:
        """Weaveトレース付きで実行."""
        weave = self._weave_client

        @weave.op(name="sandbox_execute")
        async def traced_execute(
            phase: str,
            attempt: int,
            code_preview: str,
        ) -> dict[str, Any]:
            t0 = time.time()
            result = await self._execute_impl(phase, attempt, code)
            traced_execute._result = result  # type: ignore
            return self.weave_tracer.build_execute_output(  # type: ignore
                result,
                time.time() - t0,
            )

        code_preview = (
            self.weave_tracer.format_code_markdown(self._clip(code))
            if self.weave_tracer
            else code[:500]
        )

        await traced_execute(phase, attempt, code_preview)
        return traced_execute._result  # type: ignore

    async def _execute_impl(
        self,
        phase: str,
        attempt: int,
        code: str,
    ) -> dict[str, Any] | None:
        """実行の実装."""
        # sandbox_request

        t0 = time.time()
        exec_dict, exception = await execute_code_safe(
            self.sandbox,
            code,
            self.config.code_language,
            self.config.code_timeout,
            self.config.max_output_chars,
        )

        if exception:
            # sandbox_exception
            pass
        else:
            # sandbox_response
            pass

        return exec_dict

    # 結果構築

    async def _build_result(
        self,
        question: str,
        gen_result: GenerationResult,
        exec_dict: dict[str, Any] | None,
        t0: float,
        repair_used: int,
        used_temp: float,
    ) -> SolveResult:
        """実行状態から最終結果を構築."""
        python_blocks = gen_result.python_blocks
        result_blocks = gen_result.result_blocks
        raw_output = gen_result.raw_output

        stdout = exec_dict.get("stdout", "") if exec_dict else ""
        stderr = exec_dict.get("stderr", "") if exec_dict else ""

        error = ""
        used_stdout = False
        used_emergency = False
        used_result_fallback = False
        exec_success = not is_execution_error(exec_dict) if exec_dict else False
        result_mismatch = False
        final_output_raw = ""

        # 結果のみフォールバック
        if not gen_result.has_python and result_blocks:
            final_output_raw = result_blocks[0].strip()
            used_result_fallback = True
            if not final_output_raw:
                error = "empty_result"

        # 緊急抽出
        elif not gen_result.has_python and not result_blocks:
            (
                final_output_raw,
                exec_dict,
                error,
                used_emergency,
                exec_success,
            ) = await self._try_emergency_extraction(raw_output)
            stdout = exec_dict.get("stdout", "") if exec_dict else ""
            stderr = exec_dict.get("stderr", "") if exec_dict else ""

        # 実行エラーをチェック
        if not error and is_execution_error(exec_dict):
            error = "execution_error"

        # 通常ケース - 回答を抽出
        if not error:
            (
                final_output_raw,
                used_stdout,
                used_result_fallback,
                result_mismatch,
                error,
            ) = self._extract_final_answer(stdout, result_blocks)

        final_output = to_latex_scalar(final_output_raw)

        # final

        return SolveResult(
            output=final_output,
            session_id=self._session_id,
            error=error,
            has_python_block=gen_result.has_python,
            multiple_python_blocks=len(python_blocks) > 1,
            used_stdout=used_stdout,
            used_result_fallback=used_result_fallback,
            used_emergency_extract=used_emergency,
            parse_success=gen_result.has_python,
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

    async def _try_emergency_extraction(
        self,
        raw_output: str,
    ) -> tuple[str, dict[str, Any] | None, str, bool, bool]:
        """タグ欠落時の緊急コード抽出を試行."""
        emergency_code = extract_code_emergency(raw_output)
        if not emergency_code:
            return "", None, "no_python_block", False, False

        exec_dict, _ = await execute_code_safe(
            self.sandbox,
            emergency_code,
            self.config.code_language,
            self.config.code_timeout,
            self.config.max_output_chars,
        )

        exec_success = not is_execution_error(exec_dict)
        if not exec_success:
            return "", exec_dict, "no_python_block", True, False

        stdout = exec_dict.get("stdout", "") if exec_dict else ""
        last_line = last_non_empty_line(stdout)
        return last_line, exec_dict, "", True, True

    def _extract_final_answer(
        self,
        stdout: str,
        result_blocks: list[str],
    ) -> tuple[str, bool, bool, bool, str]:
        """stdoutまたは結果ブロックから最終回答を抽出.

        優先順位:
        1. stdoutの最後の非空行
        2. 最初の結果ブロック
        """
        last_line = last_non_empty_line(stdout)

        if last_line:
            used_stdout = True
            mismatch = bool(result_blocks and result_blocks[0].strip() != last_line)
            return last_line, used_stdout, False, mismatch, ""

        if result_blocks:
            return result_blocks[0].strip(), False, True, False, ""

        return "", False, False, False, "empty_stdout"

    # ロギング

    def _log_event(self, event: str, **kwargs: Any) -> None:
        """トレースイベントのJSONL書き出しは廃止（wandbに統合）。"""
        self._event_i += 1

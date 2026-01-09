"""コード実行とLLM対話のユーティリティ."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx


async def wait_for_llm_ready(
    host: str,
    port: int,
    timeout: float,
    interval: float,
) -> None:
    """LLMサーバーが準備できるまで待機."""
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(interval)
    raise RuntimeError(f"vLLM health check timed out after {timeout:.1f}s: {url}")


async def generate_once(
    llm,
    messages: list[dict[str, str]],
    temperature: float,
    max_new_tokens: int,
    min_tokens: int,
) -> dict[str, Any]:
    """LLMから1回テキストを生成し、トークン使用量を含む.

    戻り値は以下のキーを持つ辞書:
    - generation: str
    - usage: dict[str, int] で prompt/completion/reasoning トークン数（利用可能な場合）
    """
    extra_body = {"min_tokens": min_tokens} if min_tokens > 0 else None
    result = await llm.generate_async(
        prompt=messages,
        tokens_to_generate=max_new_tokens,
        temperature=temperature,
        extra_body=extra_body,
        include_response=True,
    )
    generation = result.get("generation", "")

    usage: dict[str, int] = {}
    # nemo_skillsで解析された生成トークン数
    completion = result.get("num_generated_tokens")
    if isinstance(completion, int):
        usage["completion_tokens"] = completion
    # 推論トークン数（利用可能な場合）
    reasoning = result.get("num_reasoning_tokens")
    if isinstance(reasoning, int):
        usage["reasoning_tokens"] = reasoning
    # 元のレスポンスからプロンプトトークン数を取得（公開されている場合）
    resp = result.get("response")
    try:
        prompt_tokens = getattr(getattr(resp, "usage", None), "prompt_tokens", None)
        if isinstance(prompt_tokens, int):
            usage["prompt_tokens"] = prompt_tokens
    except Exception:
        # 構造が異なる場合は無視
        pass

    return {"generation": generation, "usage": usage}


def is_execution_error(exec_dict: dict[str, Any] | None) -> bool:
    """実行がエラーに終わったかチェック."""
    if not exec_dict:
        return True
    if exec_dict.get("process_status") in {"error", "timeout"}:
        return True
    stdout = exec_dict.get("stdout", "")
    stderr = exec_dict.get("stderr", "")
    if "Traceback" in stdout or "Traceback" in stderr:
        return True
    if "SyntaxError" in stdout or "SyntaxError" in stderr:
        return True
    return False


async def execute_code_safe(
    sandbox,
    code: str,
    language: str,
    timeout: float,
    max_output_chars: int,
) -> tuple[dict[str, Any] | None, Exception | None]:
    """例外ハンドリング付きでサンドボックス内でコードを実行."""
    try:
        exec_dict, _ = await sandbox.execute_code(
            generated_code=code,
            language=language,
            timeout=timeout,
            max_output_characters=max_output_chars,
        )
        return exec_dict, None
    except Exception as e:
        return None, e

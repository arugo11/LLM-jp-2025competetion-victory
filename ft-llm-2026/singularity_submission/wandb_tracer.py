"""LLMトレース用のW&B Weave統合.

このモジュールは、コマンドライン引数で有効/無効を切り替え可能な
オプショナルWeaveトレースサポートを提供します。
無効化されている場合、すべてのトレース操作はno-opになります。
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from utils import SolveResult

# ジェネリックデコレーター用の型変数
F = TypeVar("F", bound=Callable[..., Any])

# weaveの利用可能性に関するグローバルフラグ
_WEAVE_AVAILABLE = False
_weave_module: Any = None

try:
    import weave as _weave_module_import

    _weave_module = _weave_module_import
    _WEAVE_AVAILABLE = True
except ImportError:
    pass


def is_weave_available() -> bool:
    """weaveがインストールされているかチェック."""
    return _WEAVE_AVAILABLE


@dataclass
class WeaveConfig:
    """Weaveトレースの設定."""

    enabled: bool = False
    project: str = "llm-jp-math-tir"
    disabled_send: bool = False  # 初期化するがデータは送信しない


class WeaveTracer:
    """W&B Weaveトレースラッパー.

    オフライン使用のために完全に無効化できる条件付きトレースを提供。
    無効化されている場合、すべてのメソッドは最小限のオーバーヘッドでno-opになります。
    """

    def __init__(self, config: WeaveConfig) -> None:
        """Weaveトレーサーを初期化.

        引数:
            config: Weave設定

        例外:
            ImportError: enabled=Trueだがweaveがインストールされていない場合

        """
        self.config = config
        self._client: Any = None

        if config.enabled:
            if not _WEAVE_AVAILABLE:
                msg = (
                    "weave is not installed but --enable-wandb was specified. "
                    "Install with: pip install weave"
                )
                raise ImportError(msg)

            # Weaveを初期化
            settings = (
                {"disabled": config.disabled_send} if config.disabled_send else None
            )
            self._client = _weave_module.init(config.project, settings=settings)

    @property
    def enabled(self) -> bool:
        """トレースが有効かチェック."""
        return self.config.enabled and _WEAVE_AVAILABLE

    def op(
        self,
        name: str | None = None,
        *,
        postprocess_inputs: Callable[..., Any] | None = None,
        postprocess_output: Callable[..., Any] | None = None,
    ) -> Callable[[F], F]:
        """@weave.opデコレーターの条件付きバージョン.

        トレースが無効の場合、関数をそのまま返します。
        有効な場合、トレース用に@weave.opでラップします。

        引数:
            name: 操作のオプショナル表示名
            postprocess_inputs: 表示用に入力をフォーマットする関数
            postprocess_output: 表示用に出力をフォーマットする関数

        戻り値:
            デコレーター関数

        """
        if not self.enabled:
            # 無効時は恐等元デコレーターを返す
            def identity(func: F) -> F:
                return func

            return identity

        # オプション付きweave.opデコレーターを構築
        op_kwargs: dict[str, Any] = {}
        if name:
            op_kwargs["name"] = name
        if postprocess_inputs:
            op_kwargs["postprocess_inputs"] = postprocess_inputs
        if postprocess_output:
            op_kwargs["postprocess_output"] = postprocess_output

        return _weave_module.op(**op_kwargs) if op_kwargs else _weave_module.op

    def format_code_markdown(self, code: str) -> Any:
        """Weave表示用にコードをMarkdownとしてフォーマット.

        引数:
            code: Pythonコード文字列

        戻り値:
            有効な場合はweave.Markdownオブジェクト、それ以外は普通の文字列

        """
        if not self.enabled or not code:
            return code

        formatted = f"```python\n{code}\n```"
        return _weave_module.Markdown(formatted)

    def format_output_markdown(self, output: str) -> Any:
        """実行出力をMarkdownとしてフォーマット.

        引数:
            output: stdout/stderr出力

        戻り値:
            有効な場合はweave.Markdownオブジェクト、それ以外は普通の文字列

        """
        if not self.enabled or not output:
            return output

        formatted = f"```\n{output}\n```"
        return _weave_module.Markdown(formatted)

    def build_solve_output(self, result: SolveResult) -> dict[str, Any]:
        """solve()トレース用の構造化された出力を構築.

        Weave UIで読みやすい形式で結果をフォーマット.

        引数:
            result: 解決結果オブジェクト

        戻り値:
            Weave表示用の構造化された辞書

        """
        output: dict[str, Any] = {
            "answer": result.output,
            "session_id": result.session_id,
            "success": not result.error,
        }

        if result.error:
            output["error_type"] = result.error

        # メトリクスセクション
        output["metrics"] = {
            "exec_success": result.exec_success,
            "parse_success": result.parse_success,
            "repair_used": result.repair_used,
            "temperature": result.temperature,
        }

        # デバッグ用フラグ
        if result.used_stdout:
            output["used_stdout"] = True
        if result.used_emergency_extract:
            output["used_emergency_extract"] = True
        if result.result_vs_stdout_mismatch:
            output["result_mismatch"] = True
        if result.multiple_python_blocks:
            output["multiple_python_blocks"] = True

        return output

    def build_generate_output(
        self,
        raw_output: str,
        python_blocks: list[str],
        result_blocks: list[str],
        latency_s: float,
        tokens: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """_generate()トレース用の構造化された出力を構築.

        引数:
            raw_output: 生のLLM出力
            python_blocks: 抽出されたPythonコードブロック
            result_blocks: 抽出された結果ブロック
            latency_s: 生成レイテンシ秒

        戻り値:
            Weave表示用の構造化された辞書

        """
        output: dict[str, Any] = {
            "latency_s": round(latency_s, 3),
            "n_python_blocks": len(python_blocks),
            "n_result_blocks": len(result_blocks),
        }

        # トークン使用量メトリクス (prompt/completion/reasoning)
        if tokens:
            # W&Bメトリック名制約に準拠: snake_case英数字とアンダースコア
            if "prompt_tokens" in tokens:
                output["prompt_tokens"] = int(tokens["prompt_tokens"])  # type: ignore[index]
            if "completion_tokens" in tokens:
                output["completion_tokens"] = int(tokens["completion_tokens"])  # type: ignore[index]
            if "reasoning_tokens" in tokens:
                output["reasoning_tokens"] = int(tokens["reasoning_tokens"])  # type: ignore[index]

        # 表示用にコードをフォーマット
        if python_blocks:
            output["code"] = self.format_code_markdown(python_blocks[0])
            if len(python_blocks) > 1:
                output["additional_blocks"] = len(python_blocks) - 1

        if result_blocks:
            output["result_block"] = result_blocks[0]

        # 生出力（大きな出力はクリップ）
        if len(raw_output) > 2000:
            output["raw_output_preview"] = (
                raw_output[:1000] + "\n...[clipped]...\n" + raw_output[-500:]
            )
        else:
            output["raw_output"] = raw_output

        return output

    def build_execute_output(
        self,
        exec_dict: dict[str, Any] | None,
        latency_s: float,
        exception: Exception | None = None,
    ) -> dict[str, Any]:
        """_execute()トレース用の構造化された出力を構築.

        引数:
            exec_dict: 実行結果辞書
            latency_s: 実行レイテンシ秒
            exception: 実行失敗時の例外

        戻り値:
            Weave表示用の構造化された辞書

        """
        output: dict[str, Any] = {
            "latency_s": round(latency_s, 3),
        }

        if exception:
            output["exception"] = str(exception)
            output["status"] = "exception"
            return output

        if not exec_dict:
            output["status"] = "no_result"
            return output

        output["status"] = exec_dict.get("process_status", "unknown")

        stdout = exec_dict.get("stdout", "")
        stderr = exec_dict.get("stderr", "")

        if stdout:
            output["stdout"] = self.format_output_markdown(stdout)
        if stderr:
            output["stderr"] = self.format_output_markdown(stderr)

        if exec_dict.get("return_code") is not None:
            output["return_code"] = exec_dict["return_code"]

        return output


# シングルトンインスタンス（mainで初期化）
_tracer: WeaveTracer | None = None


def get_tracer() -> WeaveTracer | None:
    """グローバルトレーサーインスタンスを取得."""
    return _tracer


def init_tracer(config: WeaveConfig) -> WeaveTracer:
    """グローバルトレーサーインスタンスを初期化.

    引数:
        config: Weave設定

    戻り値:
        初期化されたWeaveTracerインスタンス

    """
    global _tracer
    _tracer = WeaveTracer(config)
    return _tracer

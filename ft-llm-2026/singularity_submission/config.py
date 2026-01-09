"""数学問題ソルバーの設定と引数解析.

このモジュールでは以下を定義:
- `SolverConfig`: 全設定オプションを含むデータクラス
- `parse_args()`: CLI引数パーサー
- `config_from_args()`: 解析済み引数をSolverConfigに変換

使用例:
    args = parse_args()
    config = config_from_args(args)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SolverConfig:
    """数学問題ソルバーの設定.

    属性:
        model_path: モデルディレクトリのパス
        input_path: 入力JSONLファイルのパス
        output_path: 出力JSONLファイルのパス
        log_path: 推論ログファイルのパス
        trace_log_path: 詳細トレースログファイルのパス
        trace_max_chars: 各フィールドのログ最大文字数
        log_raw_output: 生のLLM出力をログに含めるかどうか
        repair_attempts: 実行エラー時のリペア試行回数
        format_retry_attempts: タグ欠落時のフォーマット修正リトライ回数
        direct_answer_attempts: コード無しフォールバックの試行回数
        sandbox_type: コード実行サンドボックスの種類
        sandbox_host: サンドボックスサーバのホスト
        sandbox_port: サンドボックスサーバのポート
        llm_server_type: LLMサーバの種類 (vllm, sglang, trtllm)
        llm_host: LLMサーバのホスト
        llm_port: LLMサーバのポート
        llm_ready_timeout: LLMサーバ起動待機のタイムアウト
        llm_ready_interval: LLMヘルスチェックのポーリング間隔
        max_new_tokens: 生成する最大トークン数
        min_tokens: 生成する最小トークン数
        temperature: サンプリング温度
        retry_temperature: リトライ時の温度
        code_language: サンドボックス実行の言語
        code_timeout: コード実行のタイムアウト
        max_output_chars: 実行結果の最大出力文字数
        enable_wandb: W&B Weaveトレースを有効にするかどうか
        wandb_project: W&Bプロジェクト名
        wandb_disabled: W&B送信を無効にするかどうか（デバッグ用）
    """

    # 入出力パス
    model_path: Path
    input_path: Path
    output_path: Path
    log_path: Path = field(default_factory=lambda: Path("inference_log.jsonl"))
    trace_log_path: Path = field(default_factory=lambda: Path("inference_trace.jsonl"))
    trace_max_chars: int = 8000
    log_raw_output: bool = False

    # リトライ設定
    repair_attempts: int = 1
    format_retry_attempts: int = 2
    direct_answer_attempts: int = 2

    # サンドボックス設定
    sandbox_type: str = "local"
    sandbox_host: str = "127.0.0.1"
    sandbox_port: int = 6000

    # LLMサーバ設定
    llm_server_type: str = "vllm"
    llm_host: str = "127.0.0.1"
    llm_port: int = 8000
    llm_ready_timeout: float = 300.0
    llm_ready_interval: float = 2.0

    # 生成パラメータ
    max_new_tokens: int = 512
    min_tokens: int = 0
    temperature: float = 0.0
    retry_temperature: float | None = None

    # コード実行パラメータ
    code_language: str = "ipython"
    code_timeout: float = 10.0
    max_output_chars: int = 1000

    # W&B Weave設定
    enable_wandb: bool = False
    wandb_project: str = "llm-jp-math-tir"
    wandb_disabled: bool = False

    def get_temperature_sequence(self) -> list[float]:
        """初回生成試行時の温度シーケンスを取得.

        戻り値:
            温度のリスト。retry_temperatureが設定されている場合はそれも含む
        """
        temps = [self.temperature]
        if self.retry_temperature is not None:
            temps.append(self.retry_temperature)
        return temps

    def get_repair_temperature(self) -> float:
        """リペア試行時に使用する温度を取得.

        戻り値:
            リペア温度（デフォルトは max(temperature, 0.2)）
        """
        if self.retry_temperature is not None:
            return self.retry_temperature
        return max(self.temperature, 0.2)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析.

    戻り値:
        解析された引数の名前空間
    """
    parser = argparse.ArgumentParser(
        description="Tool-Integrated Reasoning (TIR) を用いた数学問題ソルバー",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使い方
  python main.py --model_path ./model --input_path problems.jsonl --output_path results.jsonl

  # W&Bトレース有効
  python main.py --model_path ./model --input_path problems.jsonl --output_path results.jsonl --enable-wandb

  # カスタムリトライ設定
  python main.py --model_path ./model --input_path problems.jsonl --output_path results.jsonl \\
      --repair-attempts 3 --format-retry-attempts 2
        """,
    )

    # 入出力パス
    io_group = parser.add_argument_group("入出力パス")
    io_group.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="モデルディレクトリのパス",
    )
    io_group.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="問題を含む入力JSONLファイルのパス",
    )
    io_group.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="結果を出力するJSONLファイルのパス",
    )
    io_group.add_argument(
        "--log_path",
        type=Path,
        default=Path("inference_log.jsonl"),
        help="推論ログファイルのパス (デフォルト: inference_log.jsonl)",
    )
    io_group.add_argument(
        "--trace_log_path",
        type=Path,
        default=Path("inference_trace.jsonl"),
        help="詳細トレースログファイルのパス (デフォルト: inference_trace.jsonl)",
    )
    io_group.add_argument(
        "--trace_max_chars",
        type=int,
        default=8000,
        help="各フィールドの最大ログ文字数 (デフォルト: 8000)",
    )
    io_group.add_argument(
        "--log-raw-output",
        action="store_true",
        help="生のLLM出力をログに含める",
    )

    # リトライ設定
    retry_group = parser.add_argument_group("リトライ設定")
    retry_group.add_argument(
        "--repair-attempts",
        type=int,
        default=1,
        help="実行エラー時のリペア試行回数 (デフォルト: 1)",
    )
    retry_group.add_argument(
        "--format-retry-attempts",
        type=int,
        default=2,
        help="<python>/<result>タグ欠落時のフォーマット修正リトライ回数 (デフォルト: 2)",
    )
    retry_group.add_argument(
        "--direct-answer-attempts",
        type=int,
        default=2,
        help="コード実行なしのフォールバック試行回数 (デフォルト: 2)",
    )

    # サンドボックス設定
    sandbox_group = parser.add_argument_group("サンドボックス設定")
    sandbox_group.add_argument(
        "--tir-sandbox-type",
        default="local",
        help="コード実行サンドボックスの種類 (デフォルト: local)",
    )
    sandbox_group.add_argument(
        "--tir-sandbox-host",
        default="127.0.0.1",
        help="サンドボックスサーバのホスト (デフォルト: 127.0.0.1)",
    )
    sandbox_group.add_argument(
        "--tir-sandbox-port",
        type=int,
        default=6000,
        help="サンドボックスサーバのポート (デフォルト: 6000)",
    )

    # LLMサーバ設定
    llm_group = parser.add_argument_group("LLMサーバ設定")
    llm_group.add_argument(
        "--tir-llm-server-type",
        default="vllm",
        help="LLMサーバの種類: vllm, sglang, trtllm (デフォルト: vllm)",
    )
    llm_group.add_argument(
        "--tir-llm-host",
        default="127.0.0.1",
        help="LLMサーバのホスト (デフォルト: 127.0.0.1)",
    )
    llm_group.add_argument(
        "--tir-llm-port",
        type=int,
        default=8000,
        help="LLMサーバのポート (デフォルト: 8000)",
    )
    llm_group.add_argument(
        "--llm-ready-timeout",
        type=float,
        default=300.0,
        help="LLMサーバ起動待機のタイムアウト秒数 (デフォルト: 300)",
    )
    llm_group.add_argument(
        "--llm-ready-interval",
        type=float,
        default=2.0,
        help="LLMヘルスチェックのポーリング間隔 (デフォルト: 2.0)",
    )

    # 生成パラメータ
    gen_group = parser.add_argument_group("生成パラメータ")
    gen_group.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="生成する最大トークン数 (デフォルト: 512)",
    )
    gen_group.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="生成する最小トークン数 (デフォルト: 0)",
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="サンプリング温度 (デフォルト: 0.0)",
    )
    gen_group.add_argument(
        "--retry-temperature",
        type=float,
        default=None,
        help="リトライ時の温度 (デフォルト: None、temperatureを使用)",
    )

    # コード実行パラメータ
    code_group = parser.add_argument_group("コード実行パラメータ")
    code_group.add_argument(
        "--code-language",
        type=str,
        default="ipython",
        help="サンドボックス実行の言語 (デフォルト: ipython)",
    )
    code_group.add_argument(
        "--code-timeout",
        type=float,
        default=10.0,
        help="コード実行のタイムアウト秒数 (デフォルト: 10.0)",
    )
    code_group.add_argument(
        "--max-output-chars",
        type=int,
        default=1000,
        help="実行結果の最大出力文字数 (デフォルト: 1000)",
    )

    # W&B Weave設定
    wandb_group = parser.add_argument_group("W&B Weaveトレース")
    wandb_group.add_argument(
        "--enable-wandb",
        action="store_true",
        help="W&B Weaveトレースを有効化 (weaveパッケージが必要)",
    )
    wandb_group.add_argument(
        "--wandb-project",
        type=str,
        default="llm-jp-math-tir",
        help="W&B Weaveプロジェクト名 (デフォルト: llm-jp-math-tir)",
    )
    wandb_group.add_argument(
        "--wandb-disabled",
        action="store_true",
        help="Weaveを初期化するが送信は無効 (デバッグ用)",
    )

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> SolverConfig:
    """解析済み引数をSolverConfigに変換.

    引数:
        args: 解析済み引数の名前空間

    戻り値:
        SolverConfigインスタンス
    """
    return SolverConfig(
        # 入出力パス
        model_path=args.model_path,
        input_path=args.input_path,
        output_path=args.output_path,
        log_path=args.log_path,
        trace_log_path=args.trace_log_path,
        trace_max_chars=args.trace_max_chars,
        log_raw_output=args.log_raw_output,
        # リトライ設定
        repair_attempts=args.repair_attempts,
        format_retry_attempts=args.format_retry_attempts,
        direct_answer_attempts=args.direct_answer_attempts,
        # サンドボックス設定
        sandbox_type=args.tir_sandbox_type,
        sandbox_host=args.tir_sandbox_host,
        sandbox_port=args.tir_sandbox_port,
        # LLMサーバ設定
        llm_server_type=args.tir_llm_server_type,
        llm_host=args.tir_llm_host,
        llm_port=args.tir_llm_port,
        llm_ready_timeout=args.llm_ready_timeout,
        llm_ready_interval=args.llm_ready_interval,
        # 生成パラメータ
        max_new_tokens=args.max_new_tokens,
        min_tokens=args.min_tokens,
        temperature=args.temperature,
        retry_temperature=args.retry_temperature,
        # コード実行パラメータ
        code_language=args.code_language,
        code_timeout=args.code_timeout,
        max_output_chars=args.max_output_chars,
        # W&B Weave設定
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_disabled=args.wandb_disabled,
    )

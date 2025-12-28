"""Singularity submission entry point for solving math problems using vLLM."""

import argparse
import asyncio
import json
import textwrap
from pathlib import Path

from nemo_skills.code_execution.sandbox import get_sandbox  # type: ignore
from nemo_skills.inference.model import get_code_execution_model  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore

PYTHON_BEGIN = "<python>"
PYTHON_END = "</python>"
RESULT_BEGIN = "<result>"
RESULT_END = "</result>"

PROMPT_TEMPLATE: str = textwrap.dedent(
    """
以下は数学の問題です。
解答を段階的に考え、最終的な解答の数値のみを\boxタグ内に記述してください。

問題
    """,
)


TIR_TEMPLATE = str = textwrap.dedent(
    """
    あなたは厳密な数学のsolverです.
    あなたは以下のフォーマットに従ってPythonを使用して計算を行ってください.
    他の文字・説明・markdown・フェンスは使用禁止です.

    <python>
    # ここにPythonコードを記述し, 最終的な計算結果を求める
    </python>

    ルール:
    - <python>の前や</rusult>の後にテキストを出力するのは禁止です
    - Markdownの ```python などを使用するのは禁止です
    - 他のプログラミング言語を使用するのは禁止です
    - 自然言語による説明や要約などを含めるのは禁止です
    - 確実に 1つの <python> ブロックと <result> ブロックだけにしてください
    """,
)


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

    parser.add_argument("--tir-llm-server-type", default="vllm")
    parser.add_argument("--tir-llm-host", default="127.0.0.1")
    parser.add_argument("--tir-llm-port", type=int, default=8000)
    parser.add_argument("--tir-model-name", type=str, default=None)

    return parser.parse_args()


def _build_prompt(question: str) -> str:
    return f"{PROMPT_TEMPLATE}\n\n"


def main() -> None:
    """推論システムを非同期で実行する関数"""
    args: argparse.Namespace = _parse_args()
    asyncio.run(_run_math_pipeline(args))


def _read_problems(
    path: Path,
):
    with Path.open(path) as f:
        problems: list[dict] = list(map(json.loads, f))

        messages: list[list[dict]] = [
            [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(question=problem["problem"]),
                },
            ]
            for problem in problems
        ]
    return problems, messages


async def _run_math_pipeline(args: argparse.Namespace) -> None:
    if not args.model_path.exists():
        _model_path = args.model_path
        raise FileNotFoundError("Model path not found: " + _model_path)

    problems, messages = _read_problems(args.input_path)

    llm = LLM(model=str(args.model_path.resolve()))

    sandbox = get_sandbox(
        sandbox_type=args.tir_sanbox_type,
        host=args.tir_sandbox_host,
        port=args.tir_sandbox_port,
    )

    llm = get_code_execution_model(
        server_type=args.tir_llm_server_type,
        host=args.tir_llm_host,
        port=args.tir_llm_port,
        model=args.tir_model_name,
        sandbox=sandbox,
    )
    outputs = llm.chat(
        messages,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=64),
    )

    for idx, problem in enumerate(problems):
        question = problem.get("problem", "NO_PROBLEM")
        session_id = str(problem.get("id", idx))
        prompt = _build_prompt(question)
    for problem, output in zip(problems, outputs):
        problem["output"] = output.outputs[0].text

    with open(args.output_path, "w") as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

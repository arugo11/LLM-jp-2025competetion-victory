# Self-Consistencyを加えた推論コード
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from config import SolverConfig
from executor import wait_for_llm_ready
from nemo_skills.code_execution.sandbox import get_sandbox  # type: ignore
from nemo_skills.inference.model import get_model  # type: ignore
from solver import ProblemSolver
from utils import read_problems
from wandb_tracer import WeaveConfig, WeaveTracer, init_tracer

from math_verify import parse
from collections import Counter

# パイプライン初期化
def _init_weave_tracer(config: SolverConfig) -> WeaveTracer | None:
    """有効な場合、Weaveトレーサーを初期化."""
    if not config.enable_wandb:
        return None

    weave_config = WeaveConfig(
        enabled=True,
        project=config.wandb_project,
        disabled_send=config.wandb_disabled,
    )
    return init_tracer(weave_config)


def _ensure_output_dirs(config: SolverConfig) -> None:
    """出力ディレクトリが存在することを確認."""
    for path in [config.output_path, config.log_path]:
        path.parent.mkdir(parents=True, exist_ok=True)


# メインパイプライン
async def run_pipeline(config: SolverConfig) -> None:
    """完全な数学問題解決パイプラインを実行.

    1. パス検証とLLMサーバ待機
    2. LLMモデルとサンドボックス初期化
    3. ソルバーで全問題を処理
    4. 結果を出力ファイルに書き込み
    """
    # パスを検証
    if not config.model_path.exists():
        raise FileNotFoundError(f"モデルパスが見つかりません: {config.model_path}")
    if not config.input_path.exists():
        raise FileNotFoundError(f"入力パスが見つかりません: {config.input_path}")

    weave_tracer = _init_weave_tracer(config)

    await wait_for_llm_ready(
        host=config.llm_host,
        port=config.llm_port,
        timeout=config.llm_ready_timeout,
        interval=config.llm_ready_interval,
    )

    problems = read_problems(config.input_path)

    if str(config.llm_server_type).lower() == "vllm":
        model_name: str = "models/" + "/".join(config.model_path.parts[-2:])
    else:
        model_name = str(config.model_path)

    llm = get_model(
        server_type=config.llm_server_type,
        host=config.llm_host,
        port=config.llm_port,
        model=model_name,
    )

    sandbox = get_sandbox(
        sandbox_type=config.sandbox_type,
        host=config.sandbox_host,
        port=config.sandbox_port,
    )

    _ensure_output_dirs(config)

    try:
        await _process_problems(config, problems, llm, sandbox, weave_tracer)
    finally:
        await sandbox.close()


async def _process_problems(
    config: SolverConfig,
    problems: list[dict],
    llm,
    sandbox,
    weave_tracer: WeaveTracer | None,
) -> None:
    """全ての問題を処理し結果を書き込み."""
    with (
        config.output_path.open("w", encoding="utf-8") as out_f,
        config.log_path.open("a", encoding="utf-8") as log_f,
    ):
        solver = ProblemSolver(llm, sandbox, config, weave_tracer)

        # 問題を1問ごとに処理
        for idx, problem in enumerate(problems):
            tmp_outputs = [] # Self-Consistency用の出力を一時的に保存するリスト
            solution_dict = {}
            for i in range(10):
                result = await solver.solve(problem, idx)
                parsed_solution = parse(result.output) # 出力を解析
                # 解答が存在しない場合はスキップ
                if parsed_solution is None or len(parsed_solution) < 2:
                    continue
                else:
                    tmp_outputs.append(parsed_solution[0])
                    # 解がユニークであれば辞書に追加
                    if str(parsed_solution[0]) not in solution_dict.keys():
                        solution_dict[str(parsed_solution[0])] = parsed_solution[1]
            
            # 多数決で解答を決定
            # もし出力がなければNoneを設定
            if not tmp_outputs:
                ranked_solutions = None
            else:
                ranked_solutions = Counter(tmp_outputs).most_common()

            found_valid = False
            final_solution = None
            if ranked_solutions:
                for answer, count in ranked_solutions:
                    if answer is not None:
                        final_solution = answer
                        found_valid = True
                        break

            # 対応するLaTeX表現を保存
            if found_valid:
                problem["output"] = solution_dict[str(final_solution)]
            else:
                problem["output"] = "None"
            
            out_f.write(json.dumps(problem, ensure_ascii=False) + "\n")

            log_entry = (
                result.to_log_entry_with_raw()
                if config.log_raw_output
                else result.to_log_entry()
            )
            log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

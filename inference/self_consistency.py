# Self-Consistencyによる推論コード
# main.pyの内容をベースに、Self-Consistencyの処理を追加

import argparse
import json
from pathlib import Path
import time

from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = """\
以下は数学の問題です。
解答を段階的に考え、最終的な答えとなる数値や解を\\boxedタグ内に記述してください。

# 制約事項
- 必ず最終的な解答を\\boxedタグ内に記述する。
- 最終的な解答は必ず一つの数値または数式で出力する。
- \\displaystyleを用いてはいけない。
- 最終的な解答では単位を出力してはならない。
- 数式は必ずlatex表記で出力する。

# 問題
{question}
"""


def main():
    # プログラム開始時間を記録
    program_start_time = time.time()

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description="Singularity Submission Example")
    parser.add_argument(
        "--model_path", type=Path, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--input_path", type=Path, required=True, help="Path to the input file"
    )
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Path to the output file"
    )
    # 最大トークン数の引数を追加
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="Maximum number of tokens"
    )
    # サンプリング数の引数を追加
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples for self-consistency"
    )

    args = parser.parse_args()

    # LLMの初期化
    llm = LLM(model=str(args.model_path.resolve()))

    # 問題ファイルの読み込み
    with open(args.input_path) as f:
        problems = list(map(json.loads, f))

    # 各問題に対してプロンプトを作成
    messages = []
    for problem in problems:
        # サンプリング数分だけ同じプロンプトを追加
        messages.append(
            [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(question=problem["problem"]),
                } for _ in range(args.num_samples)
            ]
        )

    # 推論時間の計測
    inference_start_time = time.time()
    # 推論処理
    sampling_params = SamplingParams(
        temperature=0.8,
        min_p=0.05,
        top_p=0.90,
        max_tokens=args.max_tokens,
    )
    outputs = llm.chat(
        messages, sampling_params=sampling_params
    )

    # 推論時間の表示
    inference_finish_time = time.time()
    print("Inference time: {}(s)".format(inference_finish_time - inference_start_time))

    # 結果の後処理と保存
    for problem, output in zip(problems, outputs):
        problem["output"] = output.outputs[0].text

    with open(args.output_path, "w") as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")
    
    # プログラムの総実行時間を表示
    program_finish_time = time.time()
    print("Total time: {}(s)".format(program_finish_time - program_start_time))

if __name__ == "__main__":
    main()
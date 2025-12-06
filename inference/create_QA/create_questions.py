import argparse
import json
from pathlib import Path
import time

from vllm import LLM, SamplingParams

PROMPT_TEMPLATE = """\
あなたは日本の数学における入試テスト問題を作成する専門家である。
以下に基づき日本の数学における入試テスト問題を作成しなさい。
- レベル: {category}
- ジャンル: {unit}
- 出力形式: 問題文のみの出力

## 制約事項
- 数式は必ずlatex表記で出力する。

"""

from category import category

Question_Count = 100


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

    args = parser.parse_args()

    # LLMの初期化
    llm = LLM(model=str(args.model_path.resolve()))

    # 問題ファイルの読み込み
    # with open(args.input_path) as f:
    #     problems = list(map(json.loads, f))

    # 各問題に対してプロンプトを作成
    messages = []
    category_count = len(category)
    for i in range(Question_Count):
        messages.append(
            [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(category=category[i % category_count]["category"], unit=category[i % category_count]["unit"]),
                }
            ]
        )

    # 推論時間の計測
    inference_start_time = time.time()
    # 推論処理
    outputs = llm.chat(
        messages, sampling_params=SamplingParams(temperature=1.0, max_tokens=args.max_tokens)
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

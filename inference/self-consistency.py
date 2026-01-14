# Self-Consistencyによる推論コード
# main.pyの内容をベースに、Self-Consistencyの処理を追加

import argparse
import json
from pathlib import Path
import time
import re
import sys
import copy
from collections import Counter
from math_verify import parse
from vllm import LLM, SamplingParams


# MARK: プロンプトテンプレート
PROMPT_TEMPLATE = """\
以下は数学の問題です。
解答を段階的に考え、最終的な答えとなる数値や解を\\boxedタグ内に記述してください。

# 制約事項
- 最終的な解答を必ず\\boxedタグ内に記述する。
- 最終的な解答は必ず数値または数式で出力する。
- \\displaystyleを用いてはいけない。
- 最終的な解答では単位を出力してはならない。
- 数式は必ずlatex表記で出力する。

# 問題
{question}
"""

# MARK: main
def main():
    sys.set_int_max_str_digits(0) # 無制限に設定
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
    # 最大トークン数
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="Maximum number of tokens"
    )
    # サンプリング数
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples for self-consistency"
    )
    # サンプリング時の温度パラメータ
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Temperature for sampling"
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
        messages.append(
            [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(question=problem["problem"]),
                }
            ]
        )

    # 推論時間の計測
    inference_start_time = time.time()

    # サンプリング回数分の推論処理を実行
    all_outputs = [[] for _ in range(len(messages))]
    all_non_parsed_outputs = [[] for _ in range(len(messages))]
    tmp_outputs = [] # 各イテレーションの出力を保存するリスト
    # parse時の同値表現と対応するTeX記法の解答を保持する辞書
    solution_dict = {} # key: parse時の同値表現, value: list(元の回答文字列)
    for i in range(args.num_samples):
        print("--------------------------------")
        print(f"Sampling iteration: {i+1}/{args.num_samples}")
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_k=40,
        )
        outputs = llm.chat(
            messages, sampling_params=sampling_params
        )
        tmp_outputs.append(outputs)
        # 答えを抽出
        extracted_contents = [parse(output.outputs[0].text) for output in outputs]
        # 抽出結果を保存
        for j, content in enumerate(extracted_contents):
            if (content is not None) and (len(content) >= 2):
                all_outputs[j].append(str(content[0]))
                all_non_parsed_outputs[j].append(str(content[1]))
                if str(content[0]) not in list(solution_dict.keys()):
                    solution_dict[str(content[0])] = [str(content[1])]
                else:
                    solution_dict[str(content[0])].append(str(content[1]))
            else:
                all_outputs[j].append(None)

    # Self-Consistencyによる最終解答の決定
    final_outputs = []
    for outputs in all_outputs:
        if outputs:
            # 頻度順にすべての要素を取得（例: [('5', 3), (None, 2), ('4', 1)]）
            ranked_answers = Counter(outputs).most_common()
            
            found_valid = False
            for answer, count in ranked_answers:
                # Noneではない最初の解答を探す
                if answer is not None:
                    final_outputs.append(solution_dict[f"{answer}"][-1])
                    found_valid = True
                    break
            
            # 全てのサンプルがNoneだった場合のフォールバック
            if not found_valid:
                final_outputs.append(None)
        else:
            final_outputs.append(None)
        

    # 推論時間の表示
    inference_finish_time = time.time()
    print("Inference time: {}(s)".format(inference_finish_time - inference_start_time))

    # 結果の後処理
    for problem, output in zip(problems, final_outputs):
        problem["output"] = f"$${output}$$"
    solution_methods = copy.deepcopy(problems)
    for i, tmp_output in enumerate(tmp_outputs):
        for problem, output in zip(solution_methods, tmp_output):
            problem[f"output_sample_{i}"] = output.outputs[0].text
            
    # 多数決前の parsed 最終回答（n回分）を保存
    for problem, parsed_answers in zip(solution_methods, all_non_parsed_outputs):
        problem["parsed_final_answers"] = parsed_answers

    # 結果の保存
    with open(args.output_path, "w") as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")
    # すべてのサンプル出力を保存
    all_samples_path = str(args.output_path).replace(".jsonl", "_all_samples.jsonl")
    with open(all_samples_path, "w") as f:
        for problem in solution_methods:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")

    # プログラムの総実行時間を表示
    program_finish_time = time.time()
    print("Total time: {}(s)".format(program_finish_time - program_start_time))

if __name__ == "__main__":
    main()

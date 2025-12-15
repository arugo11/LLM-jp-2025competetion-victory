# Self-Consistencyによる推論コード
# main.pyの内容をベースに、Self-Consistencyの処理を追加

import argparse
import json
from pathlib import Path
import time
import re
from collections import Counter

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

# MARK: ヘルパー関数
def extract_boxed_content(outputs: list[str]):
    """
    boxedタグ内の内容を抽出するヘルパー関数
    Args:
        outputs (list[str]): vLLMの出力テキストのリスト
    returns:
        list: 抽出された\\boxedタグ内の内容のリスト
    """
    # 正規表現パターンの定義
    pattern = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)

    # 抽出処理
    extracted_contents = []
    for output in outputs:
        match = pattern.search(output)
        if match:
            extracted_contents.append(match.group(1).strip())
        else:
            extracted_contents.append(None)  # \\boxedタグが見つからなかった場合
    
    return extracted_contents


# MARK: main
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
    tmp_outputs = [] # 各イテレーションの出力を保存するリスト
    for i in range(args.num_samples):
        sampling_params = SamplingParams(
            temperature=0.9,
            max_tokens=args.max_tokens,
        )
        outputs = llm.chat(
            messages, sampling_params=sampling_params
        )
        tmp_outputs.append(outputs)
        # \\boxedタグ内の内容を抽出
        extracted_contents = extract_boxed_content([output.outputs[0].text for output in outputs])
        # 抽出結果を保存
        for j, content in enumerate(extracted_contents):
            all_outputs[j].append(content)

    # Self-Consistencyによる最終解答の決定
    final_outputs = []
    for outputs in all_outputs:
        # Noneを除外してカウント
        filtered_outputs = [output for output in outputs if output is not None]
        if filtered_outputs:
            # 最も頻出する解答を選択
            most_common = Counter(filtered_outputs).most_common()[0][0]
            final_outputs.append(most_common)
        else:
            final_outputs.append(None)  # すべてNoneの場合
        

    # 推論時間の表示
    inference_finish_time = time.time()
    print("Inference time: {}(s)".format(inference_finish_time - inference_start_time))

    # 結果の後処理と保存
    for problem, output in zip(problems, final_outputs):
        problem["output"] = f"$${output}$$"
    # 以下は各サンプル出力を保存する場合のコード例
    # だが、正答率の計算時に不具合が生じたため検証目的以外ではコメントアウトする
    for i, tmp_output in enumerate(tmp_outputs):
        for problem, output in zip(problems, tmp_output):
            problem[f"output_sample_{i}"] = output.outputs[0].text

    with open(args.output_path, "w") as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")
    
    # プログラムの総実行時間を表示
    program_finish_time = time.time()
    print("Total time: {}(s)".format(program_finish_time - program_start_time))

if __name__ == "__main__":
    main()
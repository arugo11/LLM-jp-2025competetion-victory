import argparse
import json
from pathlib import Path
import time
import os

from vllm import LLM, SamplingParams
from huggingface_hub import HfApi

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
        "--input_path", type=Path, required=False, help="Path to the input file"
    )
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Path to the output file"
    )
    # 最大トークン数の引数を追加
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="Maximum number of tokens"
    )
    parser.add_argument(
        "--repo_id", type=str, default="llm-jp-2025/test_questions", help="Hugging Face repository ID"
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="Hugging Face token"
    )

    args = parser.parse_args()

    # LLMの初期化
    llm = LLM(model=str(args.model_path.resolve()))

    # 問題のメタデータ作成
    problems = []
    category_count = len(category)
    for i in range(Question_Count):
        problems.append({
            "id": i,
            "category": category[i % category_count]["category"],
            "unit": category[i % category_count]["unit"]
        })

    # 各問題に対してプロンプトを作成
    messages = []
    for problem in problems:
        messages.append(
            [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(category=problem["category"], unit=problem["unit"]),
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
        problem["problem"] = output.outputs[0].text

    with open(args.output_path, "w") as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")
    
    # Hugging Faceへのアップロード
    if args.hf_token:
        api = HfApi(token=args.hf_token)
        api.upload_file(
            path_or_fileobj=args.output_path,
            path_in_repo=args.output_path.name,
            repo_id=args.repo_id,
            repo_type="dataset"
        )
        print(f"Uploaded {args.output_path} to {args.repo_id}")

    # プログラムの総実行時間を表示
    program_finish_time = time.time()
    print("Total time: {}(s)".format(program_finish_time - program_start_time))

if __name__ == "__main__":
    main()

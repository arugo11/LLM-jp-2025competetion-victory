import argparse
import time
from datasets import Dataset, DatasetDict
from vllm import LLM, SamplingParams
from category import category

PROMPT_TEMPLATE = """\
以下に基づき日本の数学における入試テスト問題を一つ作成しなさい。
- レベル: {category}
- ジャンル: {unit}
- 出力形式: 問題文のみ出力

## 制約事項
- 数式は必ずlatex表記で出力する。
- 問題文以外は出力してはならない。
- 必ず日本語で出力する。
- \displaystyleを用いてはいけない。
- 一つの数値または数式で解答できる問題にする。解答は出力してはいけない。
- 問題は必ず一つのみ出力する。
-「問題」という文字列は出力してはならない。

"""

Question_Count = 100

def main():
    # プログラム開始時間を記録
    program_start_time = time.time()

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description="Create Math Questions")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="Maximum number of tokens"
    )
    parser.add_argument(
        "--repo_id", type=str, default="llm-jp-2025/test_questions", help="Hugging Face repository ID"
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="Hugging Face token"
    )
    parser.add_argument(
        "--output_jsonl", type=str, default=None, help="Path to save the output dataset as JSONL"
    )

    args = parser.parse_args()

    # LLMの初期化
    llm = LLM(model=args.model_path)

    # 問題のメタデータ作成
    problems = []
    for i in category:
        problems.append([])
    category_count = len(category)
    for i in range(Question_Count):
        problems[i % category_count].append({
            "category": category[i % category_count]["category"],
            "unit": category[i % category_count]["unit"]
        })
    problems = [item for sublist in problems for item in sublist]
    for i in range(len(problems)):
        problems[i]["id"] = i

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
    print(f"Inference time: {inference_finish_time - inference_start_time}(s)")

    # 結果の整形
    data = []
    for problem, output in zip(problems, outputs):
        raw_text = output.outputs[0].text
        data.append({
            "id": problem["id"],
            "category": problem["category"],
            "unit": problem["unit"],
            "problem": raw_text.split("assistantfinal")[-1].strip(),
            "problem_source": args.model_path
        })

    # DatasetDictの作成とデータの追加
    dataset_dict = DatasetDict()
    dataset = Dataset.from_list(data)
    dataset_dict["train"] = dataset

    # Hugging Faceへのアップロード
    if args.hf_token:
        dataset_dict.push_to_hub(args.repo_id, token=args.hf_token)
        print(f"Uploaded dataset to {args.repo_id}")
    else:
        print("HF token not provided. Skipping upload.")

    # プログラムの総実行時間を表示
    program_finish_time = time.time()
    print(f"Total time: {program_finish_time - program_start_time}(s)")

if __name__ == "__main__":
    main()

import argparse
import time
from datasets import load_dataset, Dataset, DatasetDict
from vllm import LLM, SamplingParams

PROMPT_TEMPLATE = """\
以下は数学の問題です。
解答を段階的に考え、最終的な答えとなる数値や解を\\boxedタグ内に記述してください。

# 制約事項
- 必ず最終的な解答を\\boxedタグ内に記述する。
- 最終的な解答は必ず一つの数値または数式で出力する。
- \displaystyleを用いてはいけない。
- 最終的な解答では単位を出力してはならない。

# 問題
{problem}

"""

def extract_answer(text):
    """
    Extract the content inside the last \boxed{...} tag.
    Handles nested braces.
    """
    if not text:
        return None
        
    # Find all occurrences of \boxed{
    start_marker = "\\boxed{"
    start_indices = []
    idx = text.find(start_marker)
    while idx != -1:
        start_indices.append(idx)
        idx = text.find(start_marker, idx + 1)
        
    if not start_indices:
        return None
        
    # Check each occurrence from last to first
    for start_idx in reversed(start_indices):
        content_start = start_idx + len(start_marker)
        brace_count = 1
        current_idx = content_start
        
        while current_idx < len(text) and brace_count > 0:
            if text[current_idx] == '{':
                brace_count += 1
            elif text[current_idx] == '}':
                brace_count -= 1
            current_idx += 1
            
        if brace_count == 0:
            # Found the matching closing brace
            return text[content_start:current_idx-1]
            
    return None

def main():
    # プログラム開始時間を記録
    program_start_time = time.time()

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description="Create Math Answers")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="Maximum number of tokens"
    )
    parser.add_argument(
        "--repo_id", type=str, required=True, help="Hugging Face input repository ID (dataset)"
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

    # データセットのダウンロード
    print(f"Downloading dataset from {args.repo_id}...")
    dataset = load_dataset(args.repo_id, split="train")
    
    # 各問題に対してプロンプトを作成
    messages = []
    for row in dataset:
        messages.append(
            [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(problem=row["problem"]),
                }
            ]
        )

    # 推論時間の計測
    inference_start_time = time.time()
    
    # 推論処理
    outputs = llm.chat(
        messages, sampling_params=SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    )
    
    # 推論時間の表示
    inference_finish_time = time.time()
    print(f"Inference time: {inference_finish_time - inference_start_time}(s)")

    # 結果の整形
    data = []
    for row, output in zip(dataset, outputs):
        new_row = row.copy()
        generated_text = output.outputs[0].text
        new_row["generated_solution"] = generated_text
        new_row["expected_answer"] = extract_answer(generated_text)
        data.append(new_row)

    # DatasetDictの作成とデータの追加
    dataset_dict = DatasetDict()
    new_dataset = Dataset.from_list(data)
    dataset_dict["train"] = new_dataset

    # JSONL形式で保存
    if args.output_jsonl:
        new_dataset.to_json(args.output_jsonl, orient="records", lines=True, force_ascii=False)
        print(f"Saved dataset to {args.output_jsonl}")

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

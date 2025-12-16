import argparse
import time
from datasets import load_dataset, Dataset, DatasetDict
from vllm import LLM, SamplingParams

PROMPT_TEMPLATE = r"""
以下は数学の問題と答えです。
問題と答えに不備がある場合、0を出力しなさい。問題が成立し、答えもあっている場合、1を出力しなさい。
問題と答えに不備があるとは、問題文が間違っていたり、誤解を招く表現を使っている場合と問題は成立しているが、答えが間違っている場合を指します。また、答えが存在しないなどの答えが出ている場合も不備があるとしてください。偽陽性を最小限にしたいので、少しでも怪しければ不備として0を出力しなさい。

# 制約事項
- 必ず最終的な解答を\boxedタグ内に記述する。
- 最終的な解答は必ず半角の0または1を出力する。
- \displaystyleを用いてはいけない。

# 問題
{problem}

#答え
{answer}

"""

def extract_binary(text):
    """
    Extract the content(0 or 1) inside the last \boxed{...} tag.
    Handles nested braces.
    """
    if not text:
        return None
        
    # Find all occurrences of \boxed{
    start_marker = r"\boxed{"
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
        "--repo_id", type=str, required=True, help="Hugging Face output repository ID (dataset)"
    )
    parser.add_argument(
        "--input_jsonl", type=str, default=None, help="Path to download the input dataset as JSONL"
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

    # データセットの読み込み分岐
    if args.input_jsonl:
        print(f"Loading dataset from local file: {args.input_jsonl}...")
        dataset = load_dataset("json", data_files=args.input_jsonl, split="train")
    else:
        print(f"Downloading dataset from {args.repo_id}...")
        dataset = load_dataset(args.repo_id, split="train")
    
    # 各問題に対してプロンプトを作成
    messages = []
    for row in dataset:
        messages.append(
            [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(problem=row["problem"], answer=row["expected_answer"]),
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
        new_row["validation_cot"] = generated_text
        new_row["is_valid"] = extract_binary(generated_text.split("assistantfinal")[-1].strip())
        if (new_row["is_valid"] !=0) and (new_row["is_valid"] !=1):
            continue
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

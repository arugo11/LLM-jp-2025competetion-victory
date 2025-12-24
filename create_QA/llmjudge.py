import argparse
import time
import json
from datasets import Dataset, DatasetDict
from vllm import LLM, SamplingParams
from category import category  # 既存のcategory.pyをインポート

# --- プロンプトテンプレート ---
PROMPT_QUESTION = r"""
以下に基づき日本の数学における入試テスト問題を一つ作成しなさい。
- レベル: {category}
- ジャンル: {unit}
- 難易度: {difficulty} （最大難易度10）
- 出力形式: 問題文のみ出力

## 制約事項
- 数式は必ずlatex表記で出力する。
- 問題文以外は出力してはならない。
- 必ず日本語で出力する。
- 全角のコンマ"，"は使わないこと。半角のコンマ","または通常の読点「、」を使いなさい。
- \displaystyleを用いてはいけない。
- 一つの数値または数式で解答できる問題にする。解答は出力してはいけない。
- 問題は必ず一つのみ出力する。
-「問題」という文字列は出力してはならない。
"""

PROMPT_ANSWER = r"""
以下は数学の問題です。
解答を段階的に考え、最終的な答えとなる数値や解を\boxedタグ内に記述してください。
段階的に考え、解法も含めて出力すること。

# 制約事項
- 必ず最終的な解答を\boxedタグ内に記述する。
- 最終的な解答は必ず一つの数値または数式で出力する。
- \displaystyleを用いてはいけない。
- 最終的な解答では単位を出力してはならない。
- 数式は必ずlatex表記で出力する。

# 問題
{problem}
"""

PROMPT_VERIFY = r"""
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

# --- ユーティリティ関数 ---
def extract_boxed(text):
    if not text: return None
    start_marker = r"\boxed{"
    idx = text.rfind(start_marker) # 最後に見つかったboxedを優先
    if idx == -1: return None
    
    content_start = idx + len(start_marker)
    brace_count = 1
    current_idx = content_start
    while current_idx < len(text) and brace_count > 0:
        if text[current_idx] == '{': brace_count += 1
        elif text[current_idx] == '}': brace_count -= 1
        current_idx += 1
    return text[content_start:current_idx-1] if brace_count == 0 else None

# --- ワーカープロセス ---
def worker_main(rank, gpu_ids, args, start_idx, num_questions_for_worker, temp_output_file):
    """
    各GPU（またはGPUグループ）で実行される処理
    """
    # 割り当てられたGPUのみを可視化
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    print(f"[Worker {rank}] Started. GPUs: {gpu_ids}, Questions: {num_questions_for_worker}")

    # vllmは内部でimport (fork時のCUDA初期化エラー回避のため)
    from vllm import LLM, SamplingParams

    # LLMの初期化
    llm = LLM(
        model=args.model_path,
        max_num_seqs=512, # バッチサイズ調整
        gpu_memory_utilization=0.95,
        max_model_len=args.max_tokens,
        tensor_parallel_size=len(gpu_ids), # TPサイズを指定
        trust_remote_code=True
    )

    # 問題設定の生成
    category_count = len(category)
    problem_specs = []
    for i in range(num_questions_for_worker):
        global_idx = start_idx + i
        ref = category[global_idx % category_count]
        problem_specs.append({
            "id": global_idx,
            "category": ref["category"],
            "unit": ref["unit"],
            "difficulty": ((global_idx // category_count) % 10) + 1
        })

    # --- STEP 1: 質問生成 ---
    print(f"[Worker {rank}] Generating Questions...")
    q_messages = [[{"role": "user", "content": PROMPT_QUESTION.format(**p)}] for p in problem_specs]
    q_outputs = llm.chat(q_messages, sampling_params=SamplingParams(temperature=1.0, max_tokens=args.max_tokens))
    
    results = []
    for spec, out in zip(problem_specs, q_outputs):
        raw_text = out.outputs[0].text
        # モデルによって終了トークン等の扱いが異なるため適宜調整
        spec["problem"] = raw_text.split("assistantfinal")[-1].strip()
        results.append(spec)

    # --- STEP 2: 解答生成 ---
    print(f"[Worker {rank}] Generating Answers...")
    a_messages = [[{"role": "user", "content": PROMPT_ANSWER.format(problem=r["problem"])}] for r in results]
    a_outputs = llm.chat(a_messages, sampling_params=SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    for res, out in zip(results, a_outputs):
        generated_text = out.outputs[0].text
        res["generated_solution"] = generated_text
        res["expected_answer"] = extract_boxed(generated_text)

    # --- STEP 3: 検証 ---
    print(f"[Worker {rank}] Verifying...")
    v_indices = [i for i, r in enumerate(results) if r["expected_answer"] is not None]
    v_messages = [[{"role": "user", "content": PROMPT_VERIFY.format(problem=results[i]["problem"], answer=results[i]["expected_answer"])}] for i in v_indices]
    
    if v_messages:
        v_outputs = llm.chat(v_messages, sampling_params=SamplingParams(temperature=0.0, max_tokens=1024))
        for idx, out in zip(v_indices, v_outputs):
            val_text = out.outputs[0].text.split("assistantfinal")[-1].strip()
            val_binary = extract_boxed(val_text)
            
            if val_binary not in ["0", "1"]:
                results[idx]["is_valid"] = -1
            else:
                results[idx]["is_valid"] = int(val_binary)
            results[idx]["validation_cot"] = val_text
    
    # 補填
    for i, r in enumerate(results):
        if "is_valid" not in r: r["is_valid"] = -2

    # 部分的な結果をJSONLとして保存
    print(f"[Worker {rank}] Saving temporary output to {temp_output_file}...")
    with open(temp_output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, default="output/final_qa.jsonl")
    parser.add_argument("--num_questions", type=int, required=True)
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallelism size per worker. Default 1 (Data Parallelism preference).")
    args = parser.parse_args()

    # spawn方式でないとCUDAコンテキストが多重起動でクラッシュする
    set_start_method('spawn', force=True)

    # 利用可能なGPU数の確認
    total_gpus = torch.cuda.device_count()
    print(f"Total GPUs detected: {total_gpus}")

    if total_gpus < args.tp_size:
        raise ValueError(f"Not enough GPUs ({total_gpus}) for requested tp_size ({args.tp_size})")

    # ワーカー数の計算
    num_workers = total_gpus // args.tp_size
    questions_per_worker = args.num_questions // num_workers
    remainder = args.num_questions % num_workers

    print(f"Plan: Launching {num_workers} workers. (TP_SIZE={args.tp_size})")

    processes = []
    temp_files = []

    for i in range(num_workers):
        # GPU割り当ての計算
        start_gpu = i * args.tp_size
        gpu_ids = list(range(start_gpu, start_gpu + args.tp_size))
        
        # 担当する問題数
        q_count = questions_per_worker + (1 if i < remainder else 0)
        # 開始インデックス（問題のID重複を防ぐため）
        start_idx = i * questions_per_worker + min(i, remainder)
        
        temp_file = f"output/temp_worker_{i}.jsonl"
        temp_files.append(temp_file)

        p = Process(target=worker_main, args=(i, gpu_ids, args, start_idx, q_count, temp_file))
        processes.append(p)
        p.start()

    # 全プロセスの終了待機
    for p in processes:
        p.join()

    # --- 統合とアップロード ---
    print("=== Merging Results ===")
    all_results = []
    for tf in temp_files:
        if os.path.exists(tf):
            with open(tf, 'r', encoding='utf-8') as f:
                for line in f:
                    all_results.append(json.loads(line))
            # 一時ファイル削除
            os.remove(tf)
    
    # 最終保存
    dataset = Dataset.from_list(all_results)
    dataset.to_json(args.output_jsonl, orient="records", lines=True, force_ascii=False)
    print(f"Saved merged dataset to {args.output_jsonl}")

    if args.hf_token:
        print(f"Uploading to Hugging Face: {args.repo_id}...")
        ds_dict = DatasetDict({"train": dataset})
        ds_dict.push_to_hub(args.repo_id, token=args.hf_token)
    
    print("All processes completed successfully.")

if __name__ == "__main__":
    main()
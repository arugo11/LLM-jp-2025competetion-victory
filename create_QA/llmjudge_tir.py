import argparse
import time
import json
import os
import glob
from datasets import Dataset, DatasetDict
from multiprocessing import Process, set_start_method
import torch

# category.py が同じディレクトリにある前提
from category import category
import textwrap
import re

import subprocess
import sys

PYTHON_BEGIN = "<PYTHON>"
PYTHON_END = "</PYTHON>"

# 補助関数（ここに配置）
def _extract_python_code(generation: str) -> str | None:
    pattern = re.escape(PYTHON_BEGIN) + r"(.*?)" + re.escape(PYTHON_END)
    match = re.search(pattern, generation, re.DOTALL)
    return match.group(1).strip() if match else None
# --- 1つ目のスクリプトから必要な補助関数をコピー ---
def _extract_python_code(generation: str) -> str | None:
    PYTHON_BEGIN = "<PYTHON>"
    PYTHON_END = "</PYTHON>"
    pattern = re.escape(PYTHON_BEGIN) + r"(.*?)" + re.escape(PYTHON_END)
    match = re.search(pattern, generation, re.DOTALL)
    return match.group(1).strip() if match else None

def _last_non_empty_line(text: str) -> str | None:
    for line in reversed((text or "").splitlines()):
        if line.strip(): return line.strip()
    return None

def _format_tir_solution(problem: str, code: str, answer: str) -> str:
    return textwrap.dedent(f"""
        問題: {problem}
        Pythonを使用して計算します。
        <PYTHON>
        {code}
        </PYTHON>
        実行結果より、答えは {answer} です。
        最終答\\boxed{{{answer}}}
    """).strip()
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

TIR_SYSTEM_PROMPT = textwrap.dedent(
    f"""
    Environment: ipython

    あなたは数学問題を解くための Python コードだけを生成します。
    出力は次の形式 **のみ** を厳守してください(他の文字・説明・Markdown・フェンスは禁止)

    {PYTHON_BEGIN}
    # sympy を使って厳密に計算し、最終解の LaTeX 文字列だけを 1 行で print する
    {PYTHON_END}

    ルール:
    - 出力の最初の行は必ず '{PYTHON_BEGIN}'、最後の行は必ず '{PYTHON_END}'。
    - print は 1 回だけ。最終解の LaTeX 文字列のみを出力する(余計なログ禁止)。
    - 可能な限り sympy の厳密計算(Rational など)を使う。
    - LaTeX は `latex = sympy.latex(expr).replace(\" \", \"\")` のように空白を除去してから出力する
    """,  # noqa: E501
).strip()

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
    # サーバープロセスを起動
    sandbox_port = 6000 + rank
    sb_proc = subprocess.Popen([
        sys.executable, "-m", 
        "nemo_skills.code_execution.local_sandbox.local_sandbox_server",
        "--port", str(sandbox_port)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    time.sleep(5) # 起動待機

    try:
        # vllmは内部でimport (fork時のCUDA初期化エラー回避のため)
        from vllm import LLM, SamplingParams

        from nemo_skills.code_execution.sandbox import get_sandbox
        PYTHON_BEGIN = "<PYTHON>"
        PYTHON_END = "</PYTHON>"

        # LLMの初期化
        llm = LLM(
            model=args.model_path,
            max_num_seqs=4096, # バッチサイズ調整
            gpu_memory_utilization=0.90,
            max_model_len=args.max_tokens,
            tensor_parallel_size=len(gpu_ids), # TPサイズを指定
            trust_remote_code=True,
            enforce_eager=True
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
                "difficulty": ((global_idx // category_count) % 10) + 1,
                "problem_source": "models/openai/gpt-oss-20b"
            })

        
        sandbox = get_sandbox(sandbox_type="local", host="127.0.0.1", port=sandbox_port)
        # --- STEP 1: 質問生成 ---
        print(f"[Worker {rank}] Generating Questions...")
        q_messages = [[{"role": "user", "content": PROMPT_QUESTION.format(**p)}] for p in problem_specs]
        q_outputs = llm.chat(q_messages, sampling_params=SamplingParams(temperature=1.0, max_tokens=args.max_tokens))
        
        results = []
        for spec, out in zip(problem_specs, q_outputs):
            raw_text = out.outputs[0].text
            # モデルによって終了トークン等の扱いが異なるため適宜調整
            if len(raw_text.split("assistantfinal"))==1:
                spec["is_valid"] = -1
            spec["problem"] = raw_text.split("assistantfinal")[-1].strip()
            results.append(spec)

        # --- STEP 2: 解答生成 ---
        print(f"[Worker {rank}] Generating Answers...")
        valid_results = [r for r in results if r.get("is_valid") != -1]
        a_messages = [[{"role": "user", "content": PROMPT_ANSWER.format(problem=r["problem"])}] for r in valid_results]
        a_outputs = llm.chat(a_messages, sampling_params=SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
        
        for res in valid_results:
            # TIR用プロンプトの組み立て
            tir_prompt = [
                {"role": "system", "content": TIR_SYSTEM_PROMPT},
                {"role": "user", "content": f"次の数学問題を解いてください。出力は {PYTHON_BEGIN}...{PYTHON_END} のみ。\n\n{res['problem']}\n"},
            ]
            for attempt in range(1, 4): 
                # 1. コード生成
                out = llm.chat(tir_prompt, sampling_params=SamplingParams(temperature=0.7, max_tokens=args.max_tokens))
                raw_generation = out[0].outputs[0].text
                
                # 2. コード抽出
                code = _extract_python_code(raw_generation) # 前回の抽出関数を利用
                if not code: continue
                
                # 3. コード実行 (sandboxを利用)
                # 実際の nemo_skills のインターフェースに合わせて実行
                execution_result = sandbox.run_code(code) 
                
                if execution_result['process_status'] == 'completed':
                    # 成功したら結果を格納してループを抜ける
                    res["expected_answer"] = _last_non_empty_line(execution_result['stdout'])
                    res["generated_solution"] = _format_tir_solution(res['problem'], code, res["expected_answer"])
                    res["tir_status"] = "success"
                    break
            else:
                res["is_valid"] = -1 # 全試行失敗

        # --- STEP 3: 検証 ---
        print(f"[Worker {rank}] Verifying...")
        candidates = [r for r in results if r.get("is_valid") != -1 and r.get("expected_answer") is not None]
        valid_results_step2 = []

        # 【安全装置】文字数制限 (確実性重視のため4000文字で切る)
        # 日本語や数式が多い場合、1文字≒1～2トークン換算でバッファを持たせる
        MAX_CHAR_LIMIT = 4000 

        for r in candidates:
            # プロンプトテンプレートの長さも考慮し、問題文と解答の合計長さをチェック
            input_text_len = len(r["problem"]) + len(r["expected_answer"])
            
            if input_text_len > MAX_CHAR_LIMIT:
                print(f"[Worker {rank}] Skipped too long input: {input_text_len} chars")
                r["is_valid"] = -1 # 長すぎるので無効扱い（または別のフラグ）
                continue
                
            valid_results_step2.append(r)
        
        # フィルタリング済みのリストでメッセージ作成
        v_messages = [[{"role": "user", "content": PROMPT_VERIFY.format(problem=r["problem"], answer=r["expected_answer"])}] for r in valid_results_step2]
        
        if v_messages:
            v_outputs = llm.chat(v_messages, sampling_params=SamplingParams(temperature=0.0, max_tokens=1024))
            for r, out in zip(valid_results_step2, v_outputs):
                val_text = out.outputs[0].text
                val_binary = extract_boxed(val_text.split("assistantfinal")[-1].strip())
                
                if val_binary not in ["0", "1"]:
                    r["is_valid"] = -1 # 辞書(r)を直接書き換える
                else:
                    r["is_valid"] = int(val_binary)
                r["validation_cot"] = val_text
        
        # 補填
        for i, r in enumerate(results):
            if "is_valid" not in r: r["is_valid"] = -1

        # 部分的な結果をJSONLとして保存
        print(f"[Worker {rank}] Saving temporary output to {temp_output_file}...")
        with open(temp_output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    finally:
        # プロセス終了時にサーバーを確実に殺す
        sb_proc.terminate()
        sb_proc.wait()

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

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    
    # spawn方式でないとCUDAコンテキストが多重起動でクラッシュする
    set_start_method('spawn', force=True)

    # 利用可能なGPU数の確認
    total_gpus = torch.cuda.device_count()
    print(f"Total GPUs detected: {total_gpus}")

    print(f"DEBUG: torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"DEBUG: torch.cuda.device_count(): {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"DEBUG: GPU {i}: {torch.cuda.get_device_name(i)}")

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

        # --- 追加: 次のワーカーを起動するまで待機 ---
        # 重い初期化処理（モデルロード・Graph Capture）が重ならないように時間を空ける
        print(f"Waiting 15 seconds before launching next worker to avoid initialization storm...")
        time.sleep(15)

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
        ds_dict.push_to_hub(args.repo_id, token=args.hf_token, private=True)
    
    print("All processes completed successfully.")

if __name__ == "__main__":
    main()
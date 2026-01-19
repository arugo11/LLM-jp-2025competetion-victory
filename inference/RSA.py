import argparse
import json
from pathlib import Path
import time
import sys
import random
import re
from math_verify import parse
from vllm import LLM, SamplingParams


# --- テンプレート設定 ---
INITIAL_TEMPLATE = """\
以下は数学の問題です。
解答を段階的に考え、最終的な答えとなる数値や解を\\boxedタグ内に記述してください。

# 制約事項
- 必ず最終的な解答を\\boxedタグ内に記述する。
- 最終的な解答は必ず一つの数値または数式で出力する。
- \\displaystyleを用いてはいけない。
- 最終的な解答では単位を出力してはならない。
- 数式は必ずLaTeX表記で出力する。

# 問題
{question}
"""

REFINEMENT_TEMPLATE = """\
以下は数学の問題と、それに対するいくつかの解答候補です。
解答候補には誤りが含まれている可能性があります。
複数の候補の良い点を取り入れ、誤りを修正し、最も確実で質の高い解答を新たに作成してください。

# 制約事項
- 必ず最終的な解答を\\boxedタグ内に記述する。
- 最終的な解答は必ず一つの数値または数式で出力する。
- \\displaystyleを用いてはいけない。
- 最終的な解答では単位を出力してはならない。
- 数式は必ずLaTeX表記で出力する。

# 問題
{question}

# 解答候補
{candidates}

# 洗練された解答
"""

def clean_repetition(text, max_chars=20000):
    """連続する重複行を削除し、最大文字数でカットする"""
    # 連続する全く同じ行を1つにまとめる（簡易的なループ対策）
    lines = text.split('\n')
    new_lines = []
    prev_line = None
    for line in lines:
        if line.strip() != prev_line:
            new_lines.append(line)
            prev_line = line.strip()
    
    cleaned_text = '\n'.join(new_lines)
    # 文字数制限をかける
    return cleaned_text[:max_chars]


def main():
    program_start_time = time.time()

    parser = argparse.ArgumentParser(description="Iterative Refinement Submission Example")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the model directory")
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the input file")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to the output file")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens")
    parser.add_argument("--loops", type=int, default=4, help="Number of refinement loops")
    parser.add_argument("--population", type=int, default=40, help="Number of candidates per problem")
    parser.add_argument("--k", type=int, default=20, help="Number of candidates to sample for refinement")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")

    args = parser.parse_args()

    llm = LLM(model=str(args.model_path.resolve()))

    with open(args.input_path) as f:
        problems = list(map(json.loads, f))

    current_candidates_list = [[] for _ in problems]
    inference_start_time = time.time()

    for loop_idx in range(args.loops):
        print(f"\n{'='*30} Starting Loop {loop_idx + 1}/{args.loops} {'='*30}")

        messages = []
        for i, problem in enumerate(problems):
            if loop_idx == 0:
                prompt_content = INITIAL_TEMPLATE.format(question=problem["problem"])
            else:
                candidates_str = ""
                cands = current_candidates_list[i]
                num_samples = min(len(cands), args.k)
                selected_cands = random.sample(cands, num_samples)

                for idx, cand in enumerate(selected_cands):
                    if "<assistantfinal>" in cand:
                        final_part = cand.split("<assistantfinal>")[-1].strip()
                        # 重複除去と制限を適用
                        clean_cand = f"<assistantfinal>\n{clean_repetition(final_part)}"
                    else:
                        clean_cand = clean_repetition(cand.strip())
                    
                    candidates_str += f"--- 候補 {idx+1} ---\n{clean_cand}\n\n"
                
                prompt_content = REFINEMENT_TEMPLATE.format(
                    question=problem["problem"], 
                    candidates=candidates_str
                )

            # --- プロンプト表示用のコードを追加 ---
            print(f"\n[Problem {i+1}] Prompt Preview:")
            # 長すぎる場合を考慮して、問題文の一部とプロンプトの構成を表示
            preview_text = prompt_content
            print("-" * 50)
            print(preview_text)
            print("-" * 50)
            # ------------------------------------

            messages.append([{"role": "user", "content": prompt_content}])

        params = SamplingParams(
            temperature=args.temperature, 
            max_tokens=args.max_tokens,
            n=args.population 
        )

        outputs = llm.chat(messages, sampling_params=params)

        new_candidates_list = []
        for output in outputs:
            cands = [o.text for o in output.outputs]
            new_candidates_list.append(cands)
        
        current_candidates_list = new_candidates_list

    inference_finish_time = time.time()
    print(f"\nInference time: {inference_finish_time - inference_start_time:.2f}(s)")

    # 結果の保存処理
    for problem, candidates in zip(problems, current_candidates_list):
        problem["output"] = parse(candidates[0])[1] 
        problem["parsed_final_answers"] = []
        for idx, candidate in enumerate(candidates):
            problem[f"output_sample_{idx}"] = candidate
            problem["parsed_final_answers"].append(parse(candidate)[1])
            
    with open(args.output_path, "w") as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")
    
    program_finish_time = time.time()
    print(f"Total time: {program_finish_time - program_start_time:.2f}(s)")

if __name__ == "__main__":
    main()
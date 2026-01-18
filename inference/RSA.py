import argparse
import json
from pathlib import Path
import time
import sys
import random  # <--- 追加
from math_verify import parse
from vllm import LLM, SamplingParams
from collections import Counter  # <--- 追加

# 初回生成用のテンプレート（元のまま）
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

# 2回目以降の洗練（Refinement）用のテンプレート
# 過去の候補(candidates)を受け取り、より良い回答を作成させる指示を含みます
REFINEMENT_TEMPLATE = """\
以下は数学の問題と、別のアプローチで導出された答えの候補とその頻度です。
これらの答えを参考にしつつ、自身の思考プロセスで問題を解き、最も正確な解答を導き出してください。

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


def main():
    # プログラム開始時間を記録
    program_start_time = time.time()

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description="Iterative Refinement Submission Example")
    parser.add_argument(
        "--model_path", type=Path, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--input_path", type=Path, required=True, help="Path to the input file"
    )
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Path to the output file"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="Maximum number of tokens"
    )
    # 反復生成用のパラメータを追加
    parser.add_argument(
        "--loops", type=int, default=4, help="Number of refinement loops (default: 2)"
    )
    parser.add_argument(
        "--population", type=int, default=40, help="Number of candidates to generate per problem (default: 4)"
    )
    parser.add_argument(
        "--k", type=int, default=20, help="Number of candidates to sample for refinement (default: 4)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for generation (needs >0 for diversity)"
    )

    args = parser.parse_args()

    # LLMクラスの初期化
    llm = LLM(model=str(args.model_path.resolve()))

    # 問題ファイルの読み込み
    with open(args.input_path) as f:
        problems = list(map(json.loads, f))

    # 各問題の現在の回答候補を保持するリスト（初期状態は空）
    # shape: [problem_idx][candidate_idx] -> str
    current_candidates_list = [[] for _ in problems]

    # 推論時間の計測開始
    inference_start_time = time.time()

    # 指定された回数だけループ（生成 -> 洗練 -> 生成 -> ...）
    for loop_idx in range(args.loops):
        print(f"Starting Loop {loop_idx + 1}/{args.loops}...")

        messages = []
        for i, problem in enumerate(problems):
            if loop_idx == 0:
                # 初回ループ: 通常のプロンプトを使用
                prompt_content = INITIAL_TEMPLATE.format(question=problem["problem"])
            else:
                # 2回目以降: 前回の候補を含めたプロンプトを作成
                candidates_str = ""
                
                # --- 変更箇所ここから ---
                # 現在の候補リストを取得
                cands = current_candidates_list[i]
                
                # populationの中からランダムにk個サンプリングする
                # (もし候補数がk未満の場合はあるだけ全て使う)
                num_samples = min(len(cands), args.k)
                selected_cands = random.sample(cands, num_samples)

                # 選ばれた候補を文字列として結合
                #for idx, cand in enumerate(selected_cands):
                    #candidates_str += f"--- 候補 {idx+1} ---\n{cand}\n\n"
                parsed_answers = []
                for cand in selected_cands:
                    res = parse(cand)
                    # エラー対策: parse結果が期待通り（[0, 1]の形）かチェック
                    if isinstance(res, (list, tuple)) and len(res) > 1 and res[1] is not None:
                        parsed_answers.append(str(res[1]))
                
                # 重複を排除して結合（トークン激減）
                unique_answers = list(set(parsed_answers))
                if unique_answers:
                    candidates_str = "過去の試行で導出された答えの候補: " + ", ".join(unique_answers)
                else:
                    candidates_str = "（過去の試行で有効な形式の答えが得られませんでした）"
                
                prompt_content = REFINEMENT_TEMPLATE.format(
                    question=problem["problem"], 
                    candidates=candidates_str
                )
                # --- 変更箇所ここまで ---
                
                prompt_content = REFINEMENT_TEMPLATE.format(
                    question=problem["problem"], 
                    candidates=candidates_str
                )

            messages.append(
                [
                    {
                        "role": "user",
                        "content": prompt_content,
                    }
                ]
            )

        # SamplingParamsの設定
        # 初回や途中経過は多様性を持たせるためにtemperatureを設定し、複数の候補(n)を出す
        # 最終ループかどうかでパラメータを変える戦略もあるが、ここではシンプルに統一設定で実行
        params = SamplingParams(
            temperature=args.temperature, 
            max_tokens=args.max_tokens,
            n=args.population  # 1つのプロンプトに対して複数の候補を生成
        )

        # 推論処理
        outputs = llm.chat(messages, sampling_params=params)

        # 結果の保存（次回のループまたは最終出力用）
        new_candidates_list = []
        for output in outputs:
            # output.outputs は population 分の生成結果を含むリスト
            cands = [o.text for o in output.outputs]
            new_candidates_list.append(cands)
        
        current_candidates_list = new_candidates_list

    # 推論時間の終了
    inference_finish_time = time.time()
    print("Inference time: {}(s)".format(inference_finish_time - inference_start_time))

    # 結果の後処理と保存
    # 最終的な出力は、最後のループで生成された候補の「最初の1つ」を採用する
    # (Majority Voteなどを実装する場合はここを変更する)
    for problem, candidates in zip(problems, current_candidates_list):
        problem["output"] = parse(candidates[0])[1]  # 最初の候補の解答部分を採用
        
    # すべての候補も保存（多数決用）
    for problem, candidates in zip(problems, current_candidates_list):
        problem["parsed_final_answers"] = []
        for idx, candidate in enumerate(candidates):
            problem[f"output_sample_{idx}"] = candidate
            res = parse(candidate)
            # インデックスエラーを防ぐ安全な取得方法
            if isinstance(res, (list, tuple)) and len(res) > 1:
                ans = res[1]
            else:
                ans = None # パース失敗時はNoneを入れる
            
            problem["parsed_final_answers"].append(ans)
            #problem["parsed_final_answers"].append(parse(candidate)[1])
            

    #with open(args.output_path, "w") as f:
        #for problem in problems:
            #f.write(json.dumps(problem, ensure_ascii=False) + "\n")
    output_all_path = args.output_path.parent / (args.output_path.stem + "_all_samples.jsonl")

    with open(output_all_path, "w") as f:
        for problem in problems:
            # 各サンプルをリスト形式で保持していることを確認
            # 評価スクリプトの仕様に合わせ、必要ならここでフォーマットを整えます
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")
    
    print(f"Saved results to {output_all_path}")
    
    # プログラムの総実行時間を表示
    program_finish_time = time.time()
    print("Total time: {}(s)".format(program_finish_time - program_start_time))

if __name__ == "__main__":
    main()
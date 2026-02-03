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
import torch
from math_verify import parse
from vllm import LLM, SamplingParams


# MARK: プロンプトテンプレート
PROMPT_TEMPLATE = """\
以下は数学の問題です。
解答を段階的に考え、最終的な答えとなる数値や解を\\boxedタグ内に記述してください。

### 制約事項
- 最終的な解答を必ず\\boxedタグ内に記述する。
- 最終的な解答は必ず数値または数式で出力する。
- \\displaystyleを用いてはいけない。
- 最終的な解答では単位を出力してはならない。
- 数式は必ずlatex表記で出力する。

### 問題
{question}
"""

# MARK: Waitを用いた追加推論関数
def chat_with_wait(llm: LLM, messages: list[list[dict]], sampling_params: SamplingParams, wait_count: int, num_samples: int):
    """LLMの解答の最後にWaitを追加してさらに推論させる。"""
    # 1. 最初のプロンプトをトークンID化
    prompts_data_2 = llm.preprocess_chat(messages=messages)
    prompts_data = []
    for _ in range(num_samples):
        prompts_data.extend(copy.deepcopy(prompts_data_2))
    tokenizer = llm.get_tokenizer()
    
    # " Wait" のトークンIDを取得
    WAIT_STR = "Wait"
    STOP_STR = "assistantfinal"
    wait_token_ids = tokenizer.encode(WAIT_STR, add_special_tokens=False)
    stop_token_ids = tokenizer.encode(STOP_STR, add_special_tokens=False)
    
    # 現在の入力（トークンIDのリスト）を管理
    current_input_configs = prompts_data
    
    sampling_params_wait = copy.deepcopy(sampling_params)
    sampling_params_wait.stop_token_ids = stop_token_ids

    # --- Waitループ開始 ---
    for attempt in range(wait_count):
        # 現時点のコンテキストで生成を実行
        # sampling_params は適宜、中間生成用に調整してもOK
        outputs = llm.generate(current_input_configs, sampling_params=sampling_params_wait)
        
        new_configs = []
        token_lens = []
        token_len_sum = 0
        token_len_max = 0
        token_len_min = float('inf')
        for i, output in enumerate(outputs):
            generated_ids = list(output.outputs[0].token_ids)
            
            # --- ここで assistantfinal を除去 ---
            # stop_token_ids が生成結果の末尾に含まれているかチェックして削除
            #if generated_ids[-len(stop_token_ids):] == stop_token_ids:
            generated_ids = generated_ids[:-len(stop_token_ids)]
            
            # これまでの入力 + 今回の生成結果 + " Wait,"
            combined_ids = (
                current_input_configs[i]["prompt_token_ids"]
                + generated_ids 
                + wait_token_ids
            )
            error = 0
            if len(combined_ids) < sampling_params.max_tokens:
                new_configs.append({"prompt_token_ids": combined_ids})
            else:
                if len(new_configs) > 0:
                    new_configs.append({"prompt_token_ids": new_configs[-1]["prompt_token_ids"]})
                else:
                    error += 1
            token_lens.append(len(combined_ids))
            token_len_sum += len(combined_ids)
            token_len_max = max(token_len_max, len(combined_ids))
            token_len_min = min(token_len_min, len(combined_ids))
            
            if error > 0:
                if len(new_configs) > 0:
                    for e in range(error):
                        new_configs.append({"prompt_token_ids": new_configs[-1]["prompt_token_ids"]})
                else:
                    for e in range(error):
                        new_configs.append({"prompt_token_ids": current_input_configs[i]["prompt_token_ids"]})
        
        # 次のループ（または最終出力）のための入力を更新
        current_input_configs = new_configs
        # print(f"token len:  avg {token_len_sum / len(outputs):.1f}, max {token_len_max}, min {token_len_min}")
        # print("token lens per sample:", token_lens)
        
        # print(f"Wait Attempt {attempt + 1}/{wait_count} processed.")
        # print("decoded text after Wait addition:")
        # print(tokenizer.decode(combined_ids))
        # print("================================")
    # --- Waitループ終了 ---

    # 最終的な回答生成
    # ここでは "Wait," と言われた後の「本当の答え」を出力させる
    final_outputs = llm.generate(current_input_configs, sampling_params=sampling_params)
        
    return final_outputs

# MARK: Normalize Operator
def normalize_operator(expr: str) -> str:
    """
    数式内の演算子(+, -)を比較用に統一する関数
    Arg:
        expr (str): 数式を表す文字列
    return:
        s: 演算子が統一された数式文字列
    """
    s = expr.replace(" ", "") # 空白を削除
    s = s.replace("+", "@")
    s = s.replace("-", "@")

    return s

# MARK: Compare Terms
def compare_terms(expr1: str, expr2: str) -> bool:
    """
    2つの式中の数値どうか(演算子以外)を比較する関数
    Arg:
        expr1 (str): 数式を表す文字列1
        expr2 (str): 数式を表す文字列2
    return:
        is_equal (bool): 2つの式が同じかどうか
    """
    normalized_expr1 = normalize_operator(expr1)
    normalized_expr2 = normalize_operator(expr2)

    # 正規化した式が不一致ならばFalseを返す
    if normalized_expr1 != normalized_expr2:
        return False
    
    return True

# MARK: Squeeze Expression
def squeeze_expr(expr: str) -> str:
    """
    式中の空白スペースを除去する関数
    Arg:
        expr (str): 数式を表す文字列
    return:
        s: 空白スペースが除去された数式文字列
    """
    if expr is None:
        return ""
    s = expr.replace(" ", "") # 半角スペース
    s = s.replace("　", "") # 全角スペース
    s = s.replace("\t", "") # タブ
    s = s.replace("\n", "") # 改行
    
    return s

# MARK: Replace ±
def replace_pm(expr: str):
    """
    'a + b', 'a - b' を 'a \\pm b'に変換する関数
    Arg:
        expr (str): 数式を表す文字列, カンマで区切られている数式, ex. '$a + b, a - b$'
    return:
        new_expr (str): 変換後の数式文字列
        is_replaced (bool): 変換が行われたかどうかのフラグ
    """
    # 空白スペースを除去
    squeezed_expr = squeeze_expr(expr)

    # "\\pm"を既に含んでいる場合は明示的に前後に半角スペースを挿入して返す
    if "\\pm" in squeezed_expr:
        return squeezed_expr.replace("\\pm", " \\pm "), True

    # カンマ区切りで2つの式を分割
    parts = [p.strip() for p in squeezed_expr.split(",")]

    # 項数の確認
    # 2項でなければそのまま返す
    if len(parts) != 2:
        return expr, False
    
    # 2式をそれぞれ変数に代入
    expr1, expr2 = parts
    # $記号を削除
    expr1 = expr1.replace("$", "")
    expr2 = expr2.replace("$", "")

    # 2式が全く同じ場合はそのまま返す
    if expr1 == expr2:
        return expr, False
    
    # 数値が異なる場合はそのまま返す
    if compare_terms(expr1, expr2) is False:
        return expr, False
    
    # "+", "-"の位置を探す
    if "+" in expr1 and "-" in expr2:
        base = expr1.replace("+", " \\pm ", 1)
        return f"${base}$", True
    if "-" in expr1 and "+" in expr2:
        base = expr1.replace("-", " \\pm ", 1)
        return f"${base}$", True
    
    return expr, False

# MARK: extract_unnecessary_tokens
def extract_unnecessary_tokens(text: str) -> str:
    """
    不要な箇所を削除する関数
    Arg:
        text (str): LLMの出力テキスト
    return:
        cleaned_text (str): 不要な箇所が削除されたテキスト
    """
    # 不要なトークンのパターン
    patterns = [
        r"\\\\;",  # 長いものを先に書くか、自動ソートする仕組みにする
        r"\\;",
        r"\\\\!",
        r"\\!",
        r"\\\\quad",
        r"\\\\:",
        r"\\:",
        r"\\\\,",
        r"\\,",
    ]

    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text)

    return cleaned_text


def clean_text(text: str) -> str:
    """
    LLMの出力テキストをクリーンアップする関数
    Arg:
        text (str): LLMの出力テキスト
    return:
        cleaned_text (str): クリーンアップされたテキスト
    """
    if text is None:
        return None
    # 不要なトークンを削除
    text = extract_unnecessary_tokens(text)
    # ±変換を適用
    cleaned_text, _ = replace_pm(text)


    return cleaned_text

# MARK: main
def main():
    sys.set_int_max_str_digits(0) # 無制限に設定
    # プログラム開始時間を記録
    program_start_time = time.time()

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description="Singularity Submission Example")
    parser.add_argument(
        "--model_path", type=Path, default="models/HayatoHongoEveryonesAI/open-instruct-grpo-fast-12b", help="Path to the model directory"
    )
    parser.add_argument(
        "--input_path", type=Path, required=True, help="Path to the input file"
    )
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Path to the output file"
    )
    # 最大トークン数
    parser.add_argument(
        "--max_tokens", type=int, default=8192, help="Maximum number of tokens"
    )
    # サンプリング数
    parser.add_argument(
        "--num_samples", type=int, default=80, help="Number of samples for self-consistency"
    )
    # サンプリング時の温度パラメータ
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for sampling"
    )
    parser.add_argument(
        "--wait_count", type=int, default=2, help="Number of waits for LLM readiness"
    )

    args = parser.parse_args()
    
    print("Arguments:", args)

    # LLMの初期化
    llm = LLM(model=str(args.model_path.resolve()), tensor_parallel_size=torch.cuda.device_count())

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
    nowtime = 0
    
    sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_k=40,
        )
        # outputs = llm.chat(
        #     messages, sampling_params=sampling_params
        # )
    outputs_2 = chat_with_wait(llm, messages, sampling_params, args.wait_count, args.num_samples)
    print(len(outputs_2))
    for i in range(args.num_samples):
        nowtime = time.time()
        print("--------------------------------")
        print(f"Sampling iteration: {i+1}/{args.num_samples}")
        
        outputs = outputs_2[i*len(messages):(i+1)*len(messages)]
        
        tmp_outputs.append(outputs)
        # 答えを抽出
        extracted_contents = [parse(output.outputs[0].text) for output in outputs]

        # 抽出結果を保存
        for j, content in enumerate(extracted_contents):
            if (content is not None) and (len(content) >= 2):
                # content[0] = clean_text(str(content[0]))
                if content[1] is not None:
                    content_1_clean = clean_text(str(content[1]))
                else:
                    content_1_clean = None
                all_outputs[j].append(str(content[0]))
                all_non_parsed_outputs[j].append(str(content[1]))
                if str(content[0]) not in list(solution_dict.keys()):
                    solution_dict[str(content[0])] = [str(content_1_clean)]
                else:
                    solution_dict[str(content[0])].append(str(content_1_clean))
            else:
                all_outputs[j].append(None)
                all_non_parsed_outputs[j].append(None)
                
        print(f"Iteration time: {time.time() - nowtime}(s)")

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

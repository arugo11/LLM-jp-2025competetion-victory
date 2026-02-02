import json
import os
from collections import Counter, defaultdict
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse
import re

# 設定
INPUT_FILE = "output/output-open-instruct-grpo-fastWAIT0_all_samples.jsonl"
OUTPUT_FILE = INPUT_FILE.replace(".jsonl", "-judged.jsonl")

# ジャッジ用のシステムプロンプト
JUDGE_SYSTEM_PROMPT = r"""
以下は数学の問題、解答過程、答えです。
解答過程、答えに不備がある場合、0を出力しなさい。解答過程が正しく、答えもあっている場合、1を出力しなさい。
偽陽性を最小限にしたいので、少しでも怪しければ不備として0を出力しなさい。

# 制約事項
- 必ず最終的な解答を\boxedタグ内に記述する。
- 最終的な解答は必ず半角の0または1を出力する。
- \displaystyleを用いてはいけない。

# 問題
{problem}

# 解答過程
{solution}

# 答え
{answer}
"""

def extract_judge_score(text):
    """
    LLMの出力から \boxed{0} または \boxed{1} を抽出する。
    1であればTrue, それ以外はFalseを返す。
    """
    if not text:
        return False
    # \boxed{1} または \boxed{0} を探す（空白許容）
    match = re.search(r"\\boxed\s*\{\s*([01])\s*\}", text)
    if match:
        return match.group(1) == "1"
    
    # フォールバック: テキストの最後が単独の 1 か 0 で終わっている場合などを考慮する場合
    # 今回は厳密にboxedを要求するため、Noneの場合はFalse扱いとする
    return False

def calculate_metrics(tp, tn, fp, fn):
    """各種評価指標を計算して表示する"""
    total = tp + tn + fp + fn
    
    def safe_div(n, d):
        return n / d if d > 0 else 0.0

    tpr = safe_div(tp, tp + fn) # Recall
    tnr = safe_div(tn, tn + fp) # Specificity
    fpr = safe_div(fp, tn + fp)
    fnr = safe_div(fn, tp + fn)
    precision = safe_div(tp, tp + fp)
    recall = tpr
    f1 = safe_div(2 * precision * recall, precision + recall)

    print("\n" + "="*30)
    print("        EVALUATION METRICS        ")
    print("="*30)
    print(f"Total Samples Judged : {total}")
    print(f"True Positives  (TP) : {tp}")
    print(f"True Negatives  (TN) : {tn}")
    print(f"False Positives (FP) : {fp}")
    print(f"False Negatives (FN) : {fn}")
    print("-" * 30)
    print(f"Precision            : {precision:.4f}")
    print(f"Recall (TPR)         : {recall:.4f}")
    print(f"F1 Score             : {f1:.4f}")
    print("-" * 30)
    print(f"True Negative Rate   : {tnr:.4f}")
    print("="*30 + "\n")

def main():
    # 引数パース
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the judge model")
    parser.add_argument("--t", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--name", type=str, default="", help="Run name identifier")
    parser.add_argument("--input_file", type=str, default=INPUT_FILE, help="Input JSONL file path")
    args = parser.parse_args()

    # GPU設定（必要に応じて変更）
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # データ読み込み
    records = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.input_file}")
    # LLM初期化
    llm = LLM(model=args.model_path)
    sampling_params = SamplingParams(temperature=args.t, max_tokens=8192) # outputが短いので100で十分

    # プロンプト作成とリクエストのマッピング準備
    judge_requests = []
    # request_metadata[i] = (record_index, sample_index, parsed_answer_value)
    request_metadata = [] 

    print("Preparing judge requests...")
    for r_idx, record in enumerate(records):
        problem = record.get("problem", "")
        parsed_answers = record.get("parsed_final_answers", [])
        
        # parsed_answersがNoneまたは空の場合はスキップ
        if not parsed_answers:
            continue

        n_samples = len(parsed_answers)
        
        # レコードごとの有効なサンプルを保持する一時辞書を初期化
        record["_valid_candidates"] = [] 

        for i in range(n_samples):
            parsed_val = parsed_answers[i]
            
            # parsed_final_answersがNoneのものは除外
            if parsed_val is None:
                continue

            sample_key = f"output_sample_{i}"
            sample_text = record.get(sample_key, "")

            # assistantfinal以降の文章を取り出す
            if "assistantfinal" not in sample_text:
                continue
            
            extracted_text = sample_text.split("assistantfinal")[-1].strip()

            # ジャッジ用プロンプト作成
            user_content = JUDGE_SYSTEM_PROMPT.format(
                problem=problem,
                solution=extracted_text,
                answer=parsed_val
            )

            messages = [
                {"role": "system", "content": "You are a strict math grader."},
                {"role": "user", "content": user_content}
            ]
            
            judge_requests.append(messages)
            request_metadata.append((r_idx, i, parsed_val))

    if not judge_requests:
        print("No valid samples found to judge.")
        return

    print(f"Running Judge on {len(judge_requests)} samples...")
    # vllmで一括推論
    outputs = llm.chat(judge_requests, sampling_params=sampling_params)

    # 評価指標用カウンタ
    tp, tn, fp, fn = 0, 0, 0, 0

    # 推論結果を元のレコードにマッピング
    for k, output in enumerate(outputs):
        r_idx, s_idx, student_answer = request_metadata[k]
        
        # ジャッジモデルの出力テキスト
        judge_output_text = output.outputs[0].text.strip()
        
        # ジャッジによる判定 (0 or 1)
        judge_is_correct = extract_judge_score(judge_output_text)

        # --- 評価指標の計算 (Ground Truthとの比較) ---
        # 注意: 数学的な等価性ではなく、単純な文字列比較を行っています。
        # 厳密な評価には sympy 等を用いた数式等価性チェックが必要です。
        ground_truth = str(records[r_idx].get("solution", "")).strip()
        student_ans_str = str(student_answer).strip()
        
        # 簡易的な正誤判定 (Ground Truthと一致するか)
        # ※ ground_truthが長い解説文の場合、ここは常にFalseになる可能性があります
        is_actually_correct = (student_ans_str == ground_truth)

        if is_actually_correct and judge_is_correct:
            tp += 1
        elif not is_actually_correct and not judge_is_correct:
            tn += 1
        elif not is_actually_correct and judge_is_correct:
            fp += 1
        elif is_actually_correct and not judge_is_correct:
            fn += 1
        # ---------------------------------------------

        # ジャッジが「正解」とした場合のみ、候補リストに追加
        if judge_is_correct:
            records[r_idx]["_valid_candidates"].append(student_answer)
        
        # 後で確認できるよう、判定の詳細をレコードに一時保存（必要なら）
        # sample_key = f"output_sample_{s_idx}"
        # records[r_idx][f"{sample_key}_judge_log"] = judge_output_text

    # 指標の計算と表示
    calculate_metrics(tp, tn, fp, fn)

    # 結果の集計と書き出し
    print("Writing output...") 
    out = OUTPUT_FILE.replace(".jsonl", f"_{args.name}.jsonl")
    with open(out, "w", encoding="utf-8") as f_out:
        for record in records:
            # マッピング処理で追加したリストを取得
            valid_list = record.get("_valid_candidates", [])
            
            if valid_list:
                # 多数決 (Counter)
                count = Counter(valid_list)
                most_common_val, freq = count.most_common(1)[0]
                
                record["output"] = most_common_val
                record["valid_answers"] = valid_list
                record["valid_answer_counts"] = len(valid_list)
                
            else:
                # 有効な回答が一つもなかった場合
                record["valid_answers"] = []
                record["valid_answer_counts"] = 0
                
            for i, output in enumerate(outputs):
                sample_key = f"judge_output_{i}"
                record[sample_key] = output.outputs[0].text

            # 不要になった一時キーやsampleデータを削除（要件に従う）
            if "_valid_candidates" in record:
                del record["_valid_candidates"]
            
            # output_sample_X を削除する場合
            parsed = record.get("parsed_final_answers", [])
            for key in list(record.keys()):
                if key.startswith("output_sample_"):
                    del record[key]

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
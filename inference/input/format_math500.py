import json
import argparse

"""
math500データセットのフォーマットをdev.jsonl形式に変換するスクリプト
[dev.jsonのdictフォーマット]
- id: 問題ID
- category: 問題のカテゴリ(学年)
- unit: 問題の単元
- problem: 問題文
- soluiton: 解答(数値データのみ)

[math-500のdictフォーマット(必要な項目のみを抜粋)]
- id: 問題ID
- subject: 問題の単元(線形代数など)
- problem: 問題文
- answer: 解答(数値データのみ)
"""

def convert_math500_to_devjson(input_file, output_file):
    # 入力ファイルの読み込み
    with open(input_file, "r", encoding='utf-8') as f:
        data = list(map(json.loads, f))
    
    # 必要なデータのみを抽出して変換
    converted_data = []
    for record in data:
        new_record = {
            "id": record["id"],
            "category": record["subject"],
            "unit": "Hoge",
            "problem": record["problem"],
            "solution": record["answer"]
        }
        converted_data.append(new_record)
    
    # 出力ファイルに保存
    with open(output_file, "w", encoding='utf-8') as f:
        for record in converted_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert math500 dataset to dev.jsonl format")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input math500 JSON file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output dev.jsonl file"
    )

    args = parser.parse_args()

    convert_math500_to_devjson(args.input_file, args.output_file)
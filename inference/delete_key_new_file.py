import json

def remove_keys_from_jsonl(input_file, output_file, keys_to_remove):
    """
    JSONLファイルから指定されたキーを削除して新しいファイルを作成する
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            # 行を辞書オブジェクトに変換
            data = json.loads(line)
            
            # 指定されたキーを削除
            for key in keys_to_remove:
                if key in data:
                    del data[key]
            
            # 新しい行として書き出し
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

# --- 設定項目 ---
input_filename = 'output/output-llm-jp-4-8b-instruct-sft-expand-checkpoint-1900-12B-1000W1_all_samples.jsonl'   # 元のファイル名
output_filename = input_filename.replace('.jsonl', '_cleaned.jsonl')  # 出力ファイル名

# 削除したいキーのリスト（例として output_sample_0 から 80 までを想定する場合など）
# keys_to_delete = [
#     "output_sample_0", "output_sample_1", "output_sample_2", 
#     "output_sample_3", "output_sample_4", "output_sample_5",
#     "output_sample_6", "output_sample_7", "output_sample_8",
#     "output_sample_9"
# ]
keys_to_delete = [f"output_sample_{i}" for i in range(80)]  # output_sample_0 から output_sample_80 まで
keys_to_delete.append("id")
keys_to_delete.append("category")  # 追加で _valid_candidates も削除
keys_to_delete.append("unit")
keys_to_delete.append("problem")
keys_to_delete.append("evaluation_method")

# プログラムの実行
remove_keys_from_jsonl(input_filename, output_filename, keys_to_delete)
print(f"処理が完了しました。出力ファイル: {output_filename}")
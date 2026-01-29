import json

input_file = 'dev.jsonl'
output_file = 'dev_500.jsonl'

# 1. ファイルから全行を読み込み、辞書オブジェクトのリストを作成
f_in = open(input_file, 'r', encoding='utf-8')
lines = [json.loads(line) for line in f_in if line.strip()]
f_in.close()

# 2. データを5倍に複製（元の100件を繰り返して500件にする）
duplicated_data = lines * 5

# 3. IDを1から500まで振り直しつつ書き込み
f_out = open(output_file, 'w', encoding='utf-8')
for i, item in enumerate(duplicated_data, start=1):
    item['id'] = i
    # 1行ずつJSONとして書き出し
    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # IDが500に達したら終了（元のデータ数が100件でない場合への備忘策）
    if i >= 500:
        break
f_out.close()

print(f"完了: {input_file} を5倍に複製し、IDを500まで振ったものを {output_file} に保存しました。")
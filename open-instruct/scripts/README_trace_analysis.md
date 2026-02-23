# トレースファイル分析スクリプト

## 概要

`analyze_traces.py`は、GRPO学習で生成されたトレースファイル（JSONL形式）を読み込んで、分析しやすい形式（CSV/JSON）に変換するスクリプトです。

## 使用方法

### 基本的な使用方法

```bash
# CSVとJSONの両方で出力（デフォルト）
python scripts/analyze_traces.py output/grpo_fast__3__1768872547/traces_grpo_fast__3__1768872547.jsonl

# CSVのみ出力
python scripts/analyze_traces.py output/grpo_fast__3__1768872547/traces_grpo_fast__3__1768872547.jsonl --format csv

# JSONのみ出力
python scripts/analyze_traces.py output/grpo_fast__3__1768872547/traces_grpo_fast__3__1768872547.jsonl --format json

# 出力ディレクトリを指定
python scripts/analyze_traces.py output/grpo_fast__3__1768872547/traces_grpo_fast__3__1768872547.jsonl --output-dir ./analysis_results

# ファイル名のプレフィックスを指定
python scripts/analyze_traces.py output/grpo_fast__3__1768872547/traces_grpo_fast__3__1768872547.jsonl --prefix my_analysis
```

## 出力ファイル

スクリプトは以下のファイルを生成します：

1. **`traces_flattened.csv`** (または `--prefix` で指定した名前)
   - 各サンプルを1行に展開したCSVファイル
   - カラム: `training_step`, `sample_index`, `score`, `response`, `query`, `ground_truth`, `dataset`, `finish_reason`, その他のメトリクス

2. **`traces_flattened.json`** (または `--prefix` で指定した名前)
   - フラット化されたデータのJSON形式

3. **`traces_statistics.csv`**
   - 各トレーニングステップの統計情報
   - カラム: `training_step`, `average_score`, `solve_rate`, `total_samples`

## 出力例

```
Loading traces from: output/grpo_fast__3__1768872547/traces_grpo_fast__3__1768872547.jsonl
Loaded 2 training steps
Saving statistics to: ./traces_statistics.csv
Statistics saved: 2 steps

=== Summary Statistics ===
Total training steps: 2
Overall average score: 5.2345
Overall solve rate: 0.5234
Total samples processed: 360

Saving CSV to: ./traces_flattened.csv
CSV saved: 360 rows, 15 columns
Saving JSON to: ./traces_flattened.json
JSON saved: 360 records

✅ Analysis complete! Files saved to: ./
```

## デコード済みテキストの表示

`view_decoded_traces.py`スクリプトを使用して、デコード済みのレスポンスを読みやすく表示できます：

```bash
# 最初の10サンプルを表示
python scripts/view_decoded_traces.py traces_flattened.csv

# 正解のみ表示（スコア > 0）
python scripts/view_decoded_traces.py traces_flattened.csv --correct-only

# 不正解のみ表示（スコア == 0）
python scripts/view_decoded_traces.py traces_flattened.csv --incorrect-only

# 特定のステップのみ表示
python scripts/view_decoded_traces.py traces_flattened.csv -s 1

# 20サンプル表示
python scripts/view_decoded_traces.py traces_flattened.csv -n 20

# ステップ2の正解サンプルを5つ表示
python scripts/view_decoded_traces.py traces_flattened.csv -s 2 --correct-only -n 5
```

## データ分析例

生成されたCSVファイルをPandasで読み込んで分析：

```python
import pandas as pd

# CSVを読み込む
df = pd.read_csv('traces_flattened.csv')

# 各ステップの平均スコア
print(df.groupby('training_step')['score'].mean())

# 正解率（スコア > 0）
print(df.groupby('training_step')['score'].apply(lambda x: (x > 0).mean()))

# 特定のステップのデータを確認
step_1_data = df[df['training_step'] == 1]
print(step_1_data[['query', 'response', 'score']].head(10))
```

## オプション

- `--output-dir`: 出力ディレクトリ（デフォルト: 現在のディレクトリ）
- `--format`: 出力形式（`csv`, `json`, `both`、デフォルト: `both`）
- `--prefix`: 出力ファイル名のプレフィックス（デフォルト: `traces`）


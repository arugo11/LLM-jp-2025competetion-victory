# math-evalによる正解率の計算
`math-eval` は数学タスクの評価ツールです。
本ツールは [Math-Verify](https://github.com/huggingface/Math-Verify) を使用して回答の抽出，正規化，一致判定を行います。

今回は例として、予測ファイルを "predictions/sample.jsonl"、正解ファイルを "targets/dev.jsonl" とします。

## 事前準備
- uv のインストール
`uv` を[公式のガイドライン](https://docs.astral.sh/uv/getting-started/installation/)に従ってインストールしてください。

- 予測ファイルは"predictions"ディレクトリに保存しておいてください。
- 正解データのファイルは"targets"ディレクトリに保存しておいてください。

## 正解率の計算
以下のコマンドを実行します。
``` bash
uvx --from "git+https://github.com/llm-jp/ft-llm-2026#subdirectory=math-eval" math-eval ./predictions/sample.jsonl ./targets/dev.jsonl -o ./accuracy/acc-sample.jsonl
```
1/7時点では動作検証ができています。uvxを介して運営提供の正答率計算スクリプトを動作させているので、運営による修正が入った場合には要検証です。

- より一般的な形
``` bash
uvx --from "git+https://github.com/llm-jp/ft-llm-2026#subdirectory=math-eval" math-eval prediction_file target_file -o output_file
```
- prediction_file: 予測結果のファイルパス
- target_file: 正解データのファイルパス
- -o output_file: 正答率の出力先 (オプション、なくても良い)
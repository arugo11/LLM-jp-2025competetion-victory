# 概要
推論をたくさんして、それをヒントに推論することをなんども繰り返します。
実行未確認...

# 実行コマンド
``` bash
qsub -v MODEL_USER=HayatoHongoEveryonesAI,MODEL_REPO=llm-jp-4-8b-instruct-sft-v5-2 ./inference-eval-1gpu-rsa.sh
```

# 重要なファイル
inference/inference-eval-1gpu-rsa.sh
``` bash
パラメータ  
# model_path: 推論に使用するモデルのパス
# input_path: 推論に使用する入力データのパス
# output_path: 推論結果の出力パス
# max_tokens: モデルが生成する最大トークン数
# loops: 反復生成のループ回数
# population: 各問題に対して生成する候補の数
# k: 反復生成のためにサンプリングする候補の数
# temperature: 生成の多様性を制御する温度パラメータ
```

inference/RSA.py

# やるべきこと
1. 実行確認
2. RSAで評価して結果を評価ハブに代入
3. LLM自身に評価させて、間違いであるものをcurrent_candidates_listから取り除く。 (出来たら)
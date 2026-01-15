# SimpleRAG

### 主な機能

- **ベクトルデータベース構築**: Parquetファイルから数学問題と解答のペアを読み込み、FAISSベクトルストアを構築
- **類似問題検索**: 多言語対応の埋め込みモデル（multilingual-e5-small）を使用した高速な類似検索
- **RAG拡張**: 新しい問題に対して類似問題と解答を付加し、コンテキストを豊富にした入力を生成
- **インタラクティブな検索**: CLIインターフェースでのリアルタイム類似問題検索

## セットアップ

### 必要要件

- Python 3.12以上
- CUDA対応GPU（推奨）

### インストール

uvを使うことを推奨します：

```bash
uv sync
```

## 使い方

### 1. ベクトルストアの構築と検索（インタラクティブモード）

```bash
python src/main.py
```

初回実行時は、`data/`ディレクトリ内のParquetファイルから自動的にベクトルストアを構築します。
2回目以降は、保存されたベクトルストア（`data/qa_vectorstore/`）を読み込みます。

### 2. 開発用データセットのRAG拡張

```bash
python src/process_dev_with_rag.py
```

`data/dev.jsonl`内の問題に対して、類似問題を検索・付加し、`data/dev_with_rag.jsonl`として出力します。

## プロジェクト構成

```
SimpleRAG/
├── src/
│   ├── main.py                    # メインエントリーポイント（インタラクティブ検索）
│   ├── process_dev_with_rag.py    # 開発データセットのRAG処理
│   ├── embeddings.py              # 埋め込みモデルとベクトル検索の実装
│   └── utils.py                   # ユーティリティ関数（ファイル読み込み等）
├── data/
│   ├── train-*.parquet           # 訓練用数学問題データセット
│   ├── dev.jsonl                 # 開発用データセット
│   ├── dev_with_rag.jsonl        # RAG拡張された開発用データセット
│   └── qa_vectorstore/           # 構築されたベクトルストア
├── pyproject.toml                # プロジェクト設定
├── requirements.txt              # 依存パッケージ一覧
└── README.md                     # このファイル
```


## 埋め込みモデルのオプション

embeddings.py get_embedding_provider()のなかのモデル名を書き換えることで他のembeddingモデルに変更できます。

デフォルトでは`intfloat/multilingual-e5-small`を使用していますが、以下のモデルも利用可能です：

変更候補
- `intfloat/multilingual-e5-small` ( 3090にて20分ほどでvector化確認)
- `intfloat/multilingual-e5-base` 
- `intfloat/multilingual-e5-large-instruct` 



## create_questions

### 説明

問題と解答のデータセットを作成する。
huggingfaceにアップロードされる。
設定は、create_qa-1gpu.sh, createandverify_qa-1gpu.shで行う。

### 使い方

#### 0. 環境設定

```bash
uv venv
source .venv/bin/activate
uv sync
```

#### 1. モデルのダウンロード
```python
python download_model.py --model_name openai/gpt-oss-20b
```

#### 2. 問題解答作成
```bash
export HF_TOKEN="your_token_here"
qsub -v HF_TOKEN ./create_qa-1gpu.sh
```

#### 3. 問題解答作成＋それらが正しいかチェック
```bash
export HF_TOKEN="your_token_here"
qsub -v HF_TOKEN ./createandverify_qa-1gpu.sh
```

### 環境変数

- HF_TOKEN: Hugging Faceのトークン
## create_questions

### 説明

問題と解答のデータセットを作成する。
huggingfaceにアップロードされる。
設定は、create_qa-1gpu.shで行う。

### 問題作成

```bash
export HF_TOKEN="your_token_here"
qsub -v HF_TOKEN ./create_qa-1gpu.sh
```

### 環境変数

- HF_TOKEN: Hugging Faceのトークン
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
qsub -v HF_TOKEN,NUM_QUESTIONS=10000,REPO_ID=qa_verify_10k_test ./llmjudge-1gpu.sh
```
NUM_QUESTIONS,REPO_IDは自分でいじれる

ジョブ記録用
[ach18384ee@login5 create_QA]$ qsub -v HF_TOKEN,NUM_QUESTIONS=10000,REPO_ID=team-victory/qa_verify_10k_test_numseq1024 ./l
lmjudge-1gpu.sh
1474285.pbs1
[ach18384ee@login5 create_QA]$ qsub -v HF_TOKEN,NUM_QUESTIONS=10000,REPO_ID=team-victory/qa_verify_10k_test_numseq2048 ./l
lmjudge-1gpu.sh
1474286.pbs1
[ach18384ee@login5 create_QA]$ qsub -v HF_TOKEN,NUM_QUESTIONS=10000,REPO_ID=team-victory/qa_verify_10k_test_numseq4096 ./l
lmjudge-1gpu.sh
1474287.pbs1
[ach18384ee@login5 create_QA]$ qsub -v HF_TOKEN,NUM_QUESTIONS=10000,REPO_ID=team-victory/qa_verify_10k_test_numseq6192 ./l
lmjudge-1gpu.sh
1474288.pbs1

### 環境変数

- HF_TOKEN: Hugging Faceのトークン
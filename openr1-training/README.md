# 学習　Training

## 環境構築
openr1-trainingディレクトリで以下を実行する。venvが作成されライブラリがインストールされる。
```bash
$ module load python/3.12/3.12.9
$ module load cuda/12.8
$ . ./create_open-r1_env.sh
```

## 学習の実行

### PBS
openr1-trainingディレクトリで以下を実行する。
```bash
$ module load python/3.12/3.12.9
$ module load cuda/12.8
$ source venv/bin/activate
$ qsub ./commands/PBS/sft-llmjp4-8b-1gpu.sh
```

# 学習　Training

## 環境構築
HOMEで以下を実行する。venvが作成されライブラリがインストールされる。
```bash
$ . ./create_open-r1_env.sh
```

## 学習の実行

### PBS
openr1-trainingディレクトリで以下を実行する。
```bash
$ source env/bin/activate
$ module load python/3.12/3.12.9
$ module load cuda/12.8
$ qsub ./commands/PBS/sft-llmjp4-8b.sh
```

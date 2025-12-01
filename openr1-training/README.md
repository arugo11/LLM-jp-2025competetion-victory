# 学習　Training

## 環境構築
HOMEで以下を実行する。venvが作成されライブラリがインストールされる。
```bash
$ . ./create_open-r1_env.sh
```

## 学習の実行

### Slrum
HOMEで以下を実行する。
```bash
$ sbatch ./commands/PBS/sft-llmjp4-8b.sh
```

### PBS
HOMEで以下を実行する。
```bash
$ qsub ./commands/PBS/sft-llmjp4-8b.sh
```

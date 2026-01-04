# open-r1の学習コードを実行するための環境を構築する。
# 基本的にリポジトリのREADME.mdに従う。
# venvを使う

# cd openr1-training
module load python/3.12/3.12.9
module load cuda/12.8

uv venv venv # venvの作成
source venv/bin/activate # venvの有効化
uv pip install --upgrade pip # pipのアップグレード


uv pip install trl[vllm]
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e "open-r1[dev]" 
uv pip install setuptools

# flash-attnのインストール
uv pip install flash-attn --no-build-isolation 

# peftのインストール
uv pip install peft 


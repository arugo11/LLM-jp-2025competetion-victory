# open-r1の学習コードを実行するための環境を構築する。
# 基本的にリポジトリのREADME.mdに従う。
# venvを使う

# cd openr1-training
# module load python/3.12/3.12.9
# module load cuda/12.8

uv venv env # venvの作成
source env/bin/activate # venvの有効化
uv pip install --upgrade pip # pipのアップグレード


uv pip install trl[vllm]
uv pip install setuptools && uv pip install flash-attn --no-build-isolation 
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e "open-r1[dev]" 
uv pip install peft

# vllmのインストール
# uv pip install vllm==0.8.5.post1 

# flash-attnのインストール


# uv pip install peft

# open-r1及び、他のライブラリのインストール

# Installs flash attention.

echo "Installing Flash Attention ${TUNING_FLASH_ATTENTION_VERSION}"
source ${TARGET_DIR}/venv/bin/activate
pushd ${TARGET_DIR}/src
module load cuda/12.1
git clone https://github.com/Dao-AILab/flash-attention -b v${TUNING_FLASH_ATTENTION_VERSION}
pushd flash-attention
python -m pip install -e . --no-build-isolation
popd

deactivate

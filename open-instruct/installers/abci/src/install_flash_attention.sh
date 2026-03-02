# Installs flash attention.

echo "Installing Flash Attention ${OPEN_INSTRUCT_FLASH_ATTENTION_VERSION}"
source ${TARGET_DIR}/venv/bin/activate
pushd ${TARGET_DIR}/src

# Remove existing flash-attention directory if it exists
if [ -d "flash-attention" ]; then
    echo "Removing existing flash-attention directory"
    rm -rf flash-attention
fi

git clone https://github.com/Dao-AILab/flash-attention -b v${OPEN_INSTRUCT_FLASH_ATTENTION_VERSION}
pushd flash-attention

# Limit parallel compilation to reduce memory usage
# MAX_JOBS=1: sequential compilation (safest, slowest)
# MAX_JOBS=2: 2 parallel jobs (balanced)
export MAX_JOBS=2

python -m pip install -e . --no-build-isolation
popd

deactivate
popd

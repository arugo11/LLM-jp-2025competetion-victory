# Install pytorch

echo "Installing torch ${OPEN_INSTRUCT_TORCH_VERSION}+cu${OPEN_INSTRUCT_CUDA_VERSION_SHORT}"

source ${TARGET_DIR}/venv/bin/activate

python -m pip install \
    --no-cache-dir \
    torch==${OPEN_INSTRUCT_TORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/cu${OPEN_INSTRUCT_CUDA_VERSION_SHORT}

deactivate

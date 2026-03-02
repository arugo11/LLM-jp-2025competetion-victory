# Installs prerequisite packages from requirements.txt
# Note: flash-attn is excluded here and installed separately (needs PyTorch first)
# omegaconf is excluded here - it will be installed by ai2-olmo-core dependency

echo "Installing requirements from requirements.txt (excluding flash-attn and omegaconf)"

source ${TARGET_DIR}/venv/bin/activate

# Get the open-instruct directory (assuming this script is run from open-instruct root)
OPEN_INSTRUCT_DIR=$(dirname $(dirname $(dirname $(dirname $(realpath ${BASH_SOURCE[0]})))))

# Create a temporary requirements file without flash-attn and omegaconf
TEMP_REQUIREMENTS=$(mktemp)
grep -v "^flash-attn" ${OPEN_INSTRUCT_DIR}/requirements.txt | grep -v "^omegaconf" > ${TEMP_REQUIREMENTS} || true

python -m pip install --no-cache-dir -U -r ${TEMP_REQUIREMENTS}

# Clean up temporary file
rm -f ${TEMP_REQUIREMENTS}

# Install NLTK data
python -m nltk.downloader punkt punkt_tab

deactivate

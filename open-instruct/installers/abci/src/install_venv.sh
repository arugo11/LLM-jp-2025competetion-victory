# Script to create virtual environment
#
# This script will make the following directories:
#   * ${TARGET_DIR}/venv ... venv directory

echo "Setup venv"
pushd ${TARGET_DIR}

python/bin/python3 -m venv venv

source venv/bin/activate
pip install --upgrade pip wheel cython
pip install setuptools==75.6.0
pip install packaging
deactivate

popd  # ${TARGET_DIR}

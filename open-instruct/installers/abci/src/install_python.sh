# Script to install Python to TARGET_DIR
#
# This script will make the following directories:
#   * ${TARGET_DIR}/src/cpython ... Source of Python
#   * ${TARGET_DIR}/python ... installed Python binary

echo "Installing Python ${OPEN_INSTRUCT_PYTHON_VERSION}"

# Check if Python is already installed
if [ -f "${TARGET_DIR}/python/bin/python3" ]; then
    echo "Python ${OPEN_INSTRUCT_PYTHON_VERSION} is already installed at ${TARGET_DIR}/python"
    ${TARGET_DIR}/python/bin/python3 --version
    # Don't exit here - continue with other installation steps
else
    pushd ${TARGET_DIR}/src

    # Remove existing cpython directory if it exists
    if [ -d "cpython" ]; then
        echo "Removing existing cpython directory"
        rm -rf cpython
    fi

    git clone https://github.com/python/cpython -b v${OPEN_INSTRUCT_PYTHON_VERSION}
    pushd cpython
    ./configure --prefix="${TARGET_DIR}/python" --enable-optimizations
    make -j 64
    make install
    popd  # cpython

    popd  # ${TARGET_DIR}/src
fi

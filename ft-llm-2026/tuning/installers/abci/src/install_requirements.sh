# Installs prerequisite packages

echo "Installing requirements"
module load cuda/12.1
source ${TARGET_DIR}/venv/bin/activate
pip list
python -m pip install --no-cache-dir -U pip setuptools wheel
pip install --no-build-isolation mamba-ssm==2.2.2
pip install --no-build-isolation youtokentome==1.0.6
python -m pip install --no-cache-dir -U -r ${SCRIPT_DIR}/src/requirements.txt

deactivate

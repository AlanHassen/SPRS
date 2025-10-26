#!/usr/bin/env bash

echo "--- Starting SPRS environment and dependency installation ---"

# 1. Initialize Conda for the script's shell environment.
# This is the most important step and fixes the 'conda: command not found' error.
echo "--> Step 1/9: Initializing Conda for the script..."
eval "$(conda shell.bash hook)"

# 2. Deactivate current environment and remove the old 'sprs' environment.
# This ensures we start from a clean 'base' environment.
echo "--> Step 2/9: Cleaning up previous Conda environment (if it exists)..."
conda deactivate
# The '|| true' part prevents the script from failing if the environment doesn't exist yet.
conda env remove -n sprs --yes || true

# 3. Create the new Conda environment from the YAML file.
echo "--> Step 3/9: Creating new 'sprs' Conda environment from sprs.yml..."
conda env create -f sprs.yml

# 4. Activate the new environment for the subsequent commands.
echo "--> Step 4/9: Activating the 'sprs' environment..."
conda activate sprs

# 5. Clean up the old repository directory and clone a fresh copy.
echo "--> Step 5/9: Cloning the 'modelsmatter' repository..."
(mkdir -p external && cd external && git clone --recursive https://github.com/AlanHassen/modelsmatter)

# 6. Install the dependencies for the modelsmatter_modelzoo sub-project.
echo "--> Step 6/9: Installing 'modelsmatter_modelzoo' dependencies with Poetry..."
echo "If the this step fails you have to remove the deepspeed/fairscale dependency from the modelsmatter/external/modelsmatter_modelzoo/pyproject.toml"
(cd external/modelsmatter/external/modelsmatter_modelzoo && poetry update && poetry install)

# 7. Install the dependencies for the aizynthfinder sub-project.
echo "--> Step 7/9: Installing 'aizynthfinder' dependencies with Poetry..."
(cd aizynthfinder && poetry update && poetry install --all-extras)

# 8. Install the final package from GitHub using pip.
echo "--> Step 8/9: Installing 'syntheseus-retro-star-benchmark'..."
pip install git+https://github.com/AustinT/syntheseus-retro-star-benchmark

# 9. Install other missing things
echo "--> Step 9/9: Installing other missing packages and copying files. On a mac, please install onxruntime-silicon"
pip install protobuf==3.20.1
pip install onnxruntime
pip install onnxruntime-silicon
# downgrade because 0.1.3 doesnt work with models matter
pip install registry-factory==0.1.2
pip install upsetplot
pip install seaborn

echo "Copy model_retrostar_mlp.py to model_zoo"

#10. Copy the models_matter files to the right folder and replace files
# --- Configuration ---
SOURCE_DIR="models_matter_retrostar_mlp"
DEST_DIR="external/modelsmatter/external/modelsmatter_modelzoo/ssbenchmark/ssmodels"

# --- Copy the specific files ---
echo "--> Copying model and init files necessary to use the retrostar_mlp"

cp "${SOURCE_DIR}/__init__.py" "${DEST_DIR}/"
cp "${SOURCE_DIR}/model_retrostar_mlp.py" "${DEST_DIR}/"

mkdir -p data

echo ""
echo "✅ --- Installation completed successfully! ---"
echo "To use the environment in your terminal, run: conda activate sprs"

echo "--> Testing aizynthfinder. Check out the documentation: https://molecularai.github.io/aizynthfinder/cli.html"
aizynthcli -h

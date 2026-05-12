#!/bin/bash
# ==============================================================================
# OpenMMLab MMDetection 3.3.0 Deployment Script
# Target Hardware: NVIDIA RTX A5000 (Ampere SM_86) | Linux | Driver 580.126.09
# Target Stack: Python 3.10 | PyTorch 2.4.1 | CUDA 12.1 | MMCV 2.2.0
# ==============================================================================

# Enforce strict error handling: exit immediately if any pipeline command fails
set -e

echo "[INFO] Phase 1: Initiating Pristine Conda Environment Orchestration..."
# Generate a pristine Python 3.10 environment. This avoids the pkgutil 
# deprecation errors found in Python 3.12+ that break OpenMMLab tooling.
conda create -n mmdet_prod python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmdet_prod

echo "[INFO] Phase 2: Patching Python Build Isolation Vulnerabilities..."
# Force the installation of specific build mechanisms.
# Restricting setuptools to <82.0.0 preserves the legacy pkg_resources module, 
# preventing fatal crashes during dynamic wheel evaluations or egg-info parsing.
pip install "setuptools<82.0.0" wheel ninja packaging

echo "[INFO] Phase 3: Enforcing C-ABI NumPy Compatibility..."
# Pin NumPy to the 1.x branch to maintain C-ABI compatibility with pre-built MMCV 
# binaries, avoiding the 'numpy.dtype size changed' segmentation faults.
pip install "numpy<2.0.0"

echo "[INFO] Phase 4: Deploying PyTorch 2.4.1 Engine with CUDA 12.1 Acceleration..."
# Install the core PyTorch ecosystem targeting CUDA 12.1 to perfectly match 
# the availability matrix of the official OpenMMLab MMCV wheels.
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[INFO] Phase 5: Deploying OpenMMLab Core Utility Engines..."
# Install OpenMIM and MMEngine. 
pip install -U openmim
mim install mmengine==0.10.5

echo "[INFO] Phase 6: Deploying MMCV 2.2.0 (Strict Binary Mode)..."
# Retrieve the official pre-built mmcv wheel for PyTorch 2.4.1 / CUDA 12.1.
# The --no-build-isolation flag is absolutely critical here. It guarantees pip 
# will not spin up an ephemeral container that attempts to download setuptools>=82.
pip install mmcv==2.2.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html \
    --no-build-isolation

echo "[INFO] Phase 7: Resolving OpenMIM Dependency Corruption (Sympy/Fsspec)..."
# OpenMIM and its sub-packages frequently calculate conflicts and downgrade sympy.
# This destroys the PyTorch Dynamo backend's ability to perform symbolic tracing.
# Force re-installation of the exact versions required by PyTorch 2.4.x.
pip install sympy==1.13.1 fsspec==2024.6.1

echo "[INFO] Phase 8: Deploying MMDetection 3.3.0 Framework..."
# The --no-deps flag is critical. If dependencies are resolved dynamically, pip 
# will parse that MMDetection 3.3.0 requires mmcv<2.2.0. Pip would automatically 
# uninstall mmcv 2.2.0 and trigger a catastrophic source build of mmcv 2.1.0.
pip install mmdet==3.3.0 --no-deps

# Manually satisfy MMDetection's peripheral graphing and processing dependencies 
# safely, bypassing the internal conflict solver and pinning OpenCV versions to 
# respect the NumPy 1.x requirement.
pip install pycocotools matplotlib terminaltables scipy shapely \
    "opencv-python-headless<4.10.0" "opencv-python<4.10.0" pyclipper rich

echo "[INFO] Phase 9: Executing Synchronous Syntax Patch for MMDetection Catch-22..."
# Dynamically discover the exact absolute path to the active Conda environment.
CONDA_ENV_PATH=$(python -c "import sys; print(sys.prefix)")
MMDET_INIT_FILE="$CONDA_ENV_PATH/lib/python3.10/site-packages/mmdet/__init__.py"

# Utilize sed (Stream Editor) to surgically rewrite the hardcoded assertion.
# Altering mmcv_maximum_version from 2.2.0 to 2.3.0 mathematically bypasses 
# the digit_version assertion lock, enabling execution alongside mmcv 2.2.0.
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET_INIT_FILE"

echo "fAdvanced Deep Learning Infrastructure Deployment Complete."
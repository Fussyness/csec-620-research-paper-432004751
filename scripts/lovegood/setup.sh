#!/usr/bin/env bash
# One-time setup on lovegood.cs.rit.edu (or any Linux GPU server without a scheduler).
# Run from the repo root:  bash scripts/lovegood/setup.sh
set -euo pipefail

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# CUDA 12.8 wheels bundle their own CUDA runtime; the server's driver (CUDA 13.x) supports them.
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt

# Download the ImageNet ResNet weights now, so no job has to fetch them mid-run.
.venv/bin/python -c "import torchvision; torchvision.models.resnet50(weights='DEFAULT'); print('weights cached')"

.venv/bin/python -c "import torch; print('torch', torch.__version__, '| GPUs visible:', torch.cuda.device_count())"
.venv/bin/python -m pytest -q

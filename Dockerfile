# Portable CPU image for reproducing the test suite and figure generation.
#
# This image installs docker-requirements.txt (the package runtime deps plus
# pytest/ruff, no Jupyter stack) on CPU torch wheels, so it builds and runs
# `pytest -q` on any x86_64 or arm64 host without a GPU.
#
# The CSD3 GPU training environment is pinned separately in hpc-requirements.txt
# (CUDA 11.8 PyTorch wheels). Those wheels resolve only on an x86_64 CUDA host, so
# they are installed into the CSD3 venv directly, not built into this image:
#   pip install -r hpc-requirements.txt
FROM python:3.11-slim

# opencv-python and matplotlib need these shared libraries at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so the layer caches across source edits.
COPY docker-requirements.txt .
# Install CPU torch/vision wheels explicitly so the default CUDA build (and its
# ~1 GB of nvidia-* deps) is never pulled into this CPU-only image; the versions
# pinned in docker-requirements.txt are then already satisfied.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r docker-requirements.txt

# Copy the project (see .dockerignore for what is excluded).
COPY . .

RUN pip install --no-cache-dir -e .

# Default command verifies the build by running the test suite.
CMD ["pytest", "-q"]

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set up workspace
WORKDIR /workspace

# Copy openpi first (heavier dependency layer, changes less often)
COPY openpi/ /workspace/openpi/

# Install openpi dependencies via uv
WORKDIR /workspace/openpi
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync && \
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Copy ctrl-world requirements and install into the same venv
COPY requirements.txt /workspace/requirements.txt
RUN uv pip install -r /workspace/requirements.txt

# Replace opencv-python with headless variant to avoid cv2/config.py
# polluting sys.path and shadowing the project's config.py
RUN uv pip uninstall opencv-python 2>/dev/null; \
    uv pip install opencv-python-headless

# Copy the rest of the ctrl-world source
WORKDIR /workspace
COPY config.py /workspace/config.py
COPY models/ /workspace/models/
COPY scripts/ /workspace/scripts/
COPY dataset/ /workspace/dataset/
COPY dataset_meta_info/ /workspace/dataset_meta_info/

# Make the openpi venv the default Python
ENV PATH="/workspace/openpi/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/workspace/openpi/.venv"

# Ensure project root is found before venv site-packages (avoids cv2/config.py collision)
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# JAX memory management (prevent full GPU pre-allocation)
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

# Default to interactive shell
CMD ["/bin/bash"]

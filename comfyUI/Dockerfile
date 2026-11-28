# Description: Builds ComfyUI with CUDA-enabled PyTorch and GGUF support.
# Author: Dipesh Karmakar <dipesh.karmakar@gmail.com>
# Modifications:
# - 7de329d: Updated PyTorch to the CUDA 13.0 wheel.
# - 130fab8: Added the initial ComfyUI container build.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system libraries required by ComfyUI and its image-processing dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install ComfyUI, CUDA-enabled PyTorch, and the GGUF custom nodes.
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /app && \
    pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git /app/custom_nodes/ComfyUI-GGUF && \
    pip install --no-cache-dir -r /app/custom_nodes/ComfyUI-GGUF/requirements.txt

EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]

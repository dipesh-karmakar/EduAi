# Description: Builds the CUDA-enabled llama.cpp inference server.
# Author: Dipesh Karmakar <dipesh.karmakar@gmail.com>
# Modifications:
# - 130fab8: Added the initial inference container build.

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install the CUDA build toolchain required by llama-cpp-python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Build llama-cpp-python with CUDA support for the configured GPU architecture.
RUN pip install --no-cache-dir -U pip && \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86" \
    pip install --no-cache-dir "llama-cpp-python[server] @ git+https://github.com/JamePeng/llama-cpp-python.git" && \
    rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1

WORKDIR /app
EXPOSE 8000

# Start the inference server with the repository's model configuration.
CMD ["python", "/app/infer/run_server.py", "--config_file", "/app/infer/models.json"]

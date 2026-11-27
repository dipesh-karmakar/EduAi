# Welcome to the EduAi

This app is a self-hosted, open-source AI tutor designed to answer educational questions for students.
It runs completely locally on a single mid-range PC (specifically tuned for Ubuntu 24.04 with an NVIDIA RTX 3060 12 GB GPU) without relying on paid cloud AI APIs like ChatGPT or Claude.

## A. Git Setup (Git Wiki – Initial Setup & Daily Synchronization)

Prerequisites (Existing setup of the auther)

Hardware and OS:
Ubuntu 24.04 with a working NVIDIA driver, 32 GB RAM, RTX 3060 12 GB. At least 100 GB free disk (models about 40 GB, container images about 15 GB, ComfyUI and Python packages about 10 GB, plus room to grow). Internet access for downloads (initial downloads total about 50 GB).

To check the same :
```bash
lsb_release -a    # Ubuntu 24.04
nvidia-smi        # must show RTX 3060 and 12 GB
df -h ~           # free space
free -h           # RAM
```


### 1. Install Git and GitHub CLI

Update the package repository and install Git and GitHub CLI.

```bash
sudo apt update && sudo apt install git gh -y
```

### 2. Configure Git Identity

Configure your Git username and email address.

```bash
git config --global user.name "Your Name" && git config --global user.email "your.email@example.com"
```

### 3. Authenticate with GitHub

Authenticate your local Git environment with GitHub.

```bash
gh auth login
```

Follow the interactive prompts to complete authentication.

### 4. Clone an Existing GitHub Repository

Download the existing repository to your local machine.

```bash
git clone https://github.com/dipesh-karmakar/EduAi
```

This creates a local copy of the `EduAi` repository.

### 5. Daily Sync Workflow

Execute the following commands whenever you want to synchronize your local changes with the GitHub repository.

#### Daily Synchronization – Quick Reference

```bash
git pull origin main
git add .
git commit -m "Describe the changes you made"
git push -u origin main
```

**Note:** Replace the example username, email and commit message with your actual details.

## B. Tools installation, step by step

### 1. Update the system

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl git python3-venv python3-pip
```
### 2. Install Docker Engine
Do not use the snap version of Docker.

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```
Log out and back in (or reboot), then check:

```bash
docker --version
docker compose version
```

### 3. Install the NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update

sudo apt install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker

docker run --rm --gpus all ubuntu nvidia-smi
```

### 4. To create the searxng and litellm folders in the project directory
```bash
mkdir -p searxng litellm
```

SSL key generation to setting up SearXNG
```bash
openssl rand -hex 32
```

Now open searxng/settings.yml with nano or gedit and paste the following inside (replacing PASTE_THE_GENERATED_KEY_HERE with the key generated) and save the file

```bash
use_default_settings: true
server:
  secret_key: 'PASTE_THE_GENERATED_KEY_HERE'
  limiter: false
search:
  safe_search: 2
  formats:
    - html
    - json
```

### 5. NVIDIA CUDA compiler (nvcc)

To check the nvcc is already installed on the machine
```bash
nvcc --version
```

If not present,
```bash
sudo apt install -y build-essential cmake ninja-build git python3-dev python3-venv
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6
export PATH=/usr/local/cuda-12.6/bin:$PATH
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
nvcc --version
```

### 6. Preparing (build) llama-cpp-python with CUDA support
Upgrade pip, build llama-cpp-python[server] with CUDA support, and verify GPU offloading works.

```bash
cd <Local_Path>/EduAi
python3 -m venv infer-venv
source infer-venv/bin/activate

pip install -U pip
CMAKE_ARGS='-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86' pip install --no-cache-dir 'llama-cpp-python[server] @ git+https://github.com/JamePeng/llama-cpp-python.git'
python -c 'import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())'
```
After this step, CUDA GPU offloading should be ready.

### 7. Downloading a test model and test

Install the Hugging Face download tool and download Qwen3-14B-GGUF (~9 GB) into the models/ folder.

```bash
pip install -U 'huggingface_hub[cli]'
hf download Qwen/Qwen3-14B-GGUF --include '*Q4_K_M.gguf' --local-dir /<Local_Path>/EduAi/models/qwen3-14b
```
OR to download the model manually, Open https://huggingface.co/Qwen/Qwen3-14B-GGUF in browser, download the *Q4_K_M.gguf file (~9 GB), and save/move it into:
/<Local_Path>/EduAi/models/qwen3-14b/

Construct a testModel.py with the following test code to run the model and check the output
```bash
import glob
from llama_cpp import Llama

path = glob.glob('/<Local_Path>/EduAi/models/qwen3-14b/*Q4_K_M*.gguf')[0]

llm = Llama(model_path=path, n_ctx=8192, n_gpu_layers=-1, flash_attn=True, verbose=False)

# out = llm.create_chat_completion(messages=[{'role': 'user', 'content': 'Say hello in one sentence.'}], max_tokens=64,)
out = llm.create_chat_completion(messages=[{'role': 'user', 'content': 'Who is Christopher Columbus? Kindly reply in 2 lines, no advertisement please'}])

print(out['choices'][0]['message']['content'])
```
```bash
source infer-venv/bin/activate # If not already in the virtual environment, because all the python packages been installed in the infer-venv
python ./test/testModel.py
```

### 8. Check the GPU usage

```bash
watch -c gpustat --color
```
Example output
<img width="1487" height="126" alt="Screenshot from 2026-11-25 01-14-40" src="https://github.com/user-attachments/assets/f465b6f2-32eb-4215-bd65-d8e2f4c1fe4e" />

### 9. Create the Docker Compose file
Open /<Project path>/docker-compose.yml

```bash
name: eduai

services:
  searxng:
    image: searxng/searxng
    container_name: searxng
    restart: unless-stopped
    volumes:
      - ../searxng:/etc/searxng
    ports:
      - 127.0.0.1:8888:8080

  qdrant:
    image: qdrant/qdrant
    container_name: qdrant
    restart: unless-stopped
    volumes:
      - ../data/qdrant:/qdrant/storage
    ports:
      - 127.0.0.1:6333:6333

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    restart: unless-stopped
    ports:
      - 3000:8080
    volumes:
      - ../data/openwebui:/app/backend/data
    extra_hosts:
      - host.docker.internal:host-gateway
    environment:
      - ENABLE_OLLAMA_API=false
      - OPENAI_API_BASE_URL=http://host.docker.internal:9000/v1
      - OPENAI_API_KEY=sk-local
      - ENABLE_TITLE_GENERATION=false
      - ENABLE_TAGS_GENERATION=false
      - ENABLE_AUTOCOMPLETE_GENERATION=false
```

### 10. Start the Docker Stack

Run docker containers directory to pull the images (one time) and start SearXNG, Qdrant, and Open WebUI in the background
```bash
docker compose -f containers/docker-compose.yml up -d
docker compose -f containers/docker-compose.yml down
```

and check 
```bash
docker compose -f containers/docker-compose.yml ps
```

### 11. Download models

The chat Model (Qwen3-VL 8B + Vision Encoder), the free Model (Qwen3 0.6B), the embedding Model (bge-m3 Q8_0)

Make sure infer-venv is active, then run
```bash
hf download Qwen/Qwen3-VL-8B-Instruct-GGUF --include '*Q4_K_M.gguf' --include 'mmproj*F16.gguf' --local-dir ./models/qwen3-vl-8b
hf download Qwen/Qwen3-0.6B-GGUF --include '*Q8_0.gguf' --local-dir ./models/qwen3-0.6b
```

### 12. Create infer/models.json

```bash
{
  "host": "0.0.0.0",
  "port": 8000,
  "models": [
    {
      "model": "./models/qwen3-vl-8b/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
      "model_alias": "chat",
      "chat_format": "qwen3-vl",
      "clip_model_path": "./models/qwen3-vl-8b/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
      "n_gpu_layers": -1,
      "n_ctx": 8192,
      "flash_attn": true
    },
    {
      "model": "./models/qwen3-14b/Qwen3-14B-Q4_K_M.gguf",
      "model_alias": "deep",
      "n_gpu_layers": -1,
      "n_ctx": 8192,
      "flash_attn": true
    },
    {
      "model": "./models/bge-m3/bge-m3-q8_0.gguf",
      "model_alias": "embedding",
      "embedding": true,
      "n_gpu_layers": -1
    },
    {
      "model": "./models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf",
      "model_alias": "free",
      "n_gpu_layers": 0,
      "n_ctx": 512
    }
  ]
}
```

### 13. Start the Inference Server Manually to Test

With (infer-venv) active inside /<Local_Path>/EduAi, run:

```bash
python -m llama_cpp.server --config_file infer/models.json
```

And to test the llama server, run the following command from another terminal
```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "chat", "messages": [{"role": "user", "content": "Who is Christopher Columbus? Kindly reply in 2 lines, no additional response and no advertisement please, I just need the answer from a third person perspective."}], "max_tokens": 100}'
```

Note: In the llama_cpp.server version 0.4.8, support for "qwen3-vl" is not available. Kindly apply fix to make it work OR use other models like "Gemma 3 12B" / "Qwen2.5-VL 7B" models.
The fix can be found in the separate python file /<Local-Path>/EduAi/infer/run_server.py

```bash
import llama_cpp
import llama_cpp.server.app as server_app
import llama_cpp.server.model as server_model
from llama_cpp.llama_chat_format import Qwen3VLChatHandler
from llama_cpp.server.__main__ import main


class SafeQwen3VLChatHandler(Qwen3VLChatHandler):
    def __call__(self, *args, **kwargs):
        res = super().__call__(*args, **kwargs)
        if isinstance(res, dict) and "choices" in res:
            for choice in res["choices"]:
                if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                    choice["message"].setdefault("refusal", None)
        return res


_BaseLlamaProxy = server_model.LlamaProxy
_orig_load = _BaseLlamaProxy.load_llama_from_model_settings


class CustomLlamaProxy(_BaseLlamaProxy):
    @staticmethod
    def load_llama_from_model_settings(settings):
        if settings.chat_format == "qwen3-vl":
            assert settings.clip_model_path is not None, "clip model not found"
            handler = SafeQwen3VLChatHandler(
                clip_model_path=settings.clip_model_path,
                verbose=settings.verbose,
            )
            orig_llama = llama_cpp.Llama

            def _llama_with_handler(*args, **kwargs):
                kwargs["chat_handler"] = handler
                return orig_llama(*args, **kwargs)

            llama_cpp.Llama = _llama_with_handler
            try:
                return _orig_load(settings)
            finally:
                llama_cpp.Llama = orig_llama
        return _orig_load(settings)


server_model.LlamaProxy = CustomLlamaProxy
if hasattr(server_app, "LlamaProxy"):
    server_app.LlamaProxy = CustomLlamaProxy

if __name__ == "__main__":
    main()
```

and, test the same,
```bash
python infer/run_server.py --config_file infer/models.json

it should have same output as,

python -m llama_cpp.server --config_file infer/models.json

```

### 14. Create infer/Dockerfile

Create the dockerfile at /<Local-Path>/EduAi/infer/Dockerfile
```bash
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

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

RUN pip install --no-cache-dir -U pip && \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86" \
    pip install --no-cache-dir "llama-cpp-python[server] @ git+https://github.com/JamePeng/llama-cpp-python.git" && \
    rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1

WORKDIR /app
EXPOSE 8000

CMD ["python", "/app/infer/run_server.py", "--config_file", "/app/infer/models.json"]
```

And infer to containers/docker-compose.yml
```bash
  infer:
    build: ../infer
    container_name: infer
    restart: unless-stopped
    gpus: all
    volumes:
      - ../infer:/app/infer:ro
      - ../models:/app/models:ro
    ports:
      - 127.0.0.1:8000:8000
```
Build and Start the Docker Stack

```bash
docker compose -f containers/docker-compose.yml up -d --build #(one time to generate the docker image)
```

### 15. What is engine?
engine is the brain of your chatbot (a lightweight Python FastAPI service on port 9000). Open WebUI sends every student message to engine, which runs the following step flow:
```bash
Reads photos using Qwen3-VL (via your infer container)
Searches your textbooks in Qdrant
Searches the web via SearXNG if textbooks don't cover it
Requests an illustration from ComfyUI if needed
Recomputes maths (simpleeval), checks claims against sources, and checks reading level (textstat) before replying.
```

### 16. Create engine/requirements.txt
```bash
fastapi
uvicorn
httpx
pyyaml
qdrant-client
trafilatura
simpleeval
textstat
pypdf
requests
```

Create engine/Dockerfile
```bash
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
```

Create engine/ingest.py
```bash
# ingest.py: python ingest.py book.pdf class-9-12 physics
import os
import sys
import uuid
import httpx
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

path, level, subject = sys.argv[1:4]
EMBED_URL = os.getenv('EMBED_URL', 'http://infer:8000/v1/embeddings')
QDRANT_URL = os.getenv('QDRANT_URL', 'http://qdrant:6333')

qdrant = QdrantClient(url=QDRANT_URL)
if not qdrant.collection_exists('textbooks'):
    qdrant.create_collection('textbooks', vectors_config=VectorParams(size=1024, distance=Distance.COSINE))


def embed(texts):
    r = httpx.post(EMBED_URL, json={'model': 'embedding', 'input': texts}, timeout=600)
    r.raise_for_status()
    return [d['embedding'] for d in r.json()['data']]


def chunks(text, size=900, overlap=150):
    for start in range(0, len(text), size - overlap):
        yield text[start:start + size]


pieces = []
for page_no, page in enumerate(PdfReader(path).pages, start=1):
    for piece in chunks(page.extract_text() or ''):
        if len(piece.strip()) >= 80:
            pieces.append((piece, page_no))

for i in range(0, len(pieces), 32):
    batch = pieces[i:i + 32]
    vectors = embed([text for text, _ in batch])
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                'text': text,
                'page': page_no,
                'level': level,
                'subject': subject,
                'source': path.split('/')[-1],
            },
        )
        for vec, (text, page_no) in zip(vectors, batch)
    ]
    qdrant.upsert('textbooks', points)

print('stored', len(pieces), 'chunks')
```

### 17. ComfyUI

ComfyUI docker file - /<Local-Path>/EduAi/comfyUI/Dockerfile

```bash
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /app && \
    pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git /app/custom_nodes/ComfyUI-GGUF && \
    pip install --no-cache-dir -r /app/custom_nodes/ComfyUI-GGUF/requirements.txt

EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
```

Add comfyui to containers/docker-compose.yml

```bash
  comfyui:
    build: ../comfyUI
    container_name: comfyui
    restart: unless-stopped
    gpus: all
    volumes:
      - ../models/comfyui:/app/models
      - ../data/comfyui-output:/app/output
    ports:
      - 127.0.0.1:8188:8188
```

### 18. Download the FLUX unet, clip and ae.safetensors models
```bash
./infer-venv/bin/hf download city96/FLUX.1-schnell-gguf --include '*Q4_K_S.gguf' --local-dir ./models/comfyui/unet
./infer-venv/bin/hf download comfyanonymous/flux_text_encoders --include 'clip_l.safetensors' --include 't5xxl_fp8_e4m3fn.safetensors' --local-dir ./models/comfyui/clip
./infer-venv/bin/hf download black-forest-labs/FLUX.1-schnell --include 'ae.safetensors' --local-dir ./models/comfyui/vae
```

### 19. Test Generating One Image in ComfyUI
```bash
Open http://localhost:8188 in your browser (if it was already open, refresh the page so it sees the newly downloaded model files).

Click Workflow → Browse Templates → pick the FLUX.1 schnell template.

Swap the diffusion model loader for the GGUF loader:
    Delete the default Load Diffusion Model node.
    Double-click the canvas, search Unet Loader (GGUF), add it, select flux1-schnell-Q4_K_S.gguf, and connect its MODEL output to the same node the old loader connected to.

Verify the other two loaders point to your downloaded files:
    DualCLIPLoader: clip_l.safetensors and t5xxl_fp8_e4m3fn.safetensors (type: flux)
    Load VAE: ae.safetensors

Click Run (or Queue) and wait for one test image to appear.
```

### 19. Create engine/config.yaml (See the git source)
### 20. Create engine/components/__init__.py (See the git source)
### 21. Create engine/components/llm.py (See the git source)
### 22. Create engine/components/retriever.py (See the git source)
### 23. Create engine/components/websearch.py (See the git source)
### 24. Create engine/components/imagegen.py (See the git source)
### 25. Create engine/components/verifier.py (See the git source)
### 26. Create engine/pipeline.py (See the git source)
### 27. Create engine/main.py (See the git source)

### 28. Add engine to containers/docker-compose.yml
```bash
  engine:
    build: ../engine
    container_name: engine
    restart: unless-stopped
    volumes:
      - ../engine:/app
      - ../workflows:/app/workflows:ro
    ports:
      - 127.0.0.1:9000:9000
    depends_on:
      - infer
      - qdrant
      - searxng
      - comfyui
```

### 29. Build engine and Start the Full Stack
```bash
docker compose -f containers/docker-compose.yml up -d --build
```

### 30. Final Setup Step: Connect via Open WebUI (http://localhost:3000)

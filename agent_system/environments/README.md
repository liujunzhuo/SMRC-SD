# Environment Setup

## Table of Contents
- [1. ALFWorld](#1-alfworld)  
- [2. WebShop](#2-webshop)  
- [3. Sokoban](#3-sokoban)  
- [4. Gym Cards](#4-gym-cards)  
- [5. AppWorld (Experimental)](#5-appworld-experimental)  

## 1. ALFWorld
Install with pip:
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
pip install vllm==0.8.5
```

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in `~/.cache/alfworld/`):
```bash
alfworld-download -f
```

Use `--extra` to download pre-trained checkpoints and seq2seq data.

Play a Textworld game:
```bash
alfworld-play-tw
```
---

## 2. WebShop
WebShop requires Python <=3.10, so begin by creating a new `verl-agent-webshop` environment
```bash
conda create -n verl-agent-webshop -c conda-forge --override-channels \
  python==3.10 openjdk=11 faiss-cpu mkl=2025.3.1 pip -y
conda activate verl-agent-webshop
```

Install the 1,000-product subset used by `env.webshop.use_small=True`:
```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d small
```

Note: If you encounter issues with gdown, you may need visit `https://drive.google.com/`, get your Google Drive cookie, and paste it into `.cache/gdown/cookies.txt`.
Or you may need to manually download the files.


Verify that WebShop was installed correctly by running:
```bash
python run_web_agent_text_env.py
```

After WebShop is installed, return to the root directory of the repository and install the verl package in `verl-agent`:
```bash
cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.5
pip3 install numpy==1.26.4 opencv-python-headless==4.11.0.86 \
  typer==0.9.4 click==8.2.1 websockets==11.0.3 cupy-cuda12x==13.3.0 \
  google-api-core==2.24.2 protobuf==4.25.9 \
  'ray[default,cgraph]==2.43.0' \
  opentelemetry-api==1.26.0 opentelemetry-sdk==1.26.0 \
  opentelemetry-exporter-otlp==1.26.0 \
  opentelemetry-exporter-prometheus==0.47b0 \
  swanlab==0.7.19
pip3 uninstall -y fastapi-cli fastapi-cloud-cli decord
pip3 check
```

`setup.sh` pins MKL 2025 for the FAISS 1.10 ABI and persists the conda JVM
paths used by PyJNIus. Reactivate the environment once after setup.
Ray 2.43 is intentional: vLLM 0.8.5 requires OpenTelemetry below 1.27, whereas
the Ray 2.49/2.50 dashboard metrics implementation expects a newer API.  Keep
the WebShop Python 3.10 stack isolated from the ALFWorld Python 3.12 stack.

---
## 3. Sokoban
```bash
pip install matplotlib
pip install gym==0.26.2
pip install gym_sokoban==0.0.6
```
---
## 4. Gym Cards

```bash
cd repo_root/
pip3 install -e ./agent_system/environments/env_package/gym_cards/gym-cards/
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
```
---
### 5. AppWorld (Experimental)
Install AppWorld package
```bash
cd repo_root/
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
pip install -e .
pip install vllm==0.8.5
```
You can ignore the warning of incompatiblity for appworld, because we don't run appworld in `verl-agent` environment.

Create a dedicated conda environment `appworld` for the AppWorld server:
```bash
conda create -n appworld python=3.12 -y
conda activate appworld
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
appworld download data
```

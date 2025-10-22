# Awesome-GRPO: Modular RLHF & RLVR Training


This repo is a clean, swappable architecture. You can switch the optimization rule (GRPO or DAPO) via a flag. 

The generation worker (vLLM) runs in a separate process and streams training batches through the Bottle service (`ref_client.py`, which actually serves as the HTTP server).

Protocol compatibility: import helpers from `ref_client.py` (`tensor_to_bytes`, `bytes_to_tensor`, `make_bytes_list`, `bytes_list_to_list`) and use the same `/upload` and `/get` endpoints, so you can drop the code in without changing servers.



## 📦 Repository Structure

```bash
Awesome-GRPO/
│
├── CODE/                     # Source code for GRPO and its variants
│   ├── algorithms/           # Core implementation (GRPO, DAPO, Dr.GRPO etc.)
│   ├── configs/              # Configs
│   ├── utils/                # Common utilities
│   ├── README.md             # How to run each strategy
│   └── ...

│
├── papers/                   # Collected and categorized PDFs for related research
│   ├──README.md              # Summary of different strategies
│   ├── GRPO.pdf
│   ├── DAPO.pdf
│   ├── DrGRPO.pdf
│   ├── GTPO.pdf
│   ├── GCPO.pdf
│   ├── GRPO-S.pdf
│   └── ...
│
└── Image/                   # Figures, tables, diagrams for visualization
````



## 🧠 Overview of Implemented Methods

| Method        | Code Configuration | Description with Summary |
| :------------ | :----------------: | :----------------------- |
| **GRPO**      |         ✅          | ✅                        |
| **DAPO**      |         ✅          | ✅                        |
| **Dr.GRPO**   |         ✅          | ✅                        |
| **2-GRPO** |         ✅          | ✅                        |
| **GTPO**      |         ✅          | ✅                        |
| **GRPO-S**    |         ☐          | ✅                        |
| **Pref-GRPO** |         ☐          | ✅                        |
| **L2T-GRPO** |         ☐          | ✅                        |
| **EDGE-GRPO** |         ☐          | ✅                        |
| ...           |                    |                          |

✅ means available in `CODE/` and `Paper and Summary/`
☐ means planned for upcoming releases.



## ⚙️ Quick Start


### 1️⃣ Environment Setup

```bash
# Clone the repository
git clone https://github.com/WangJingyao07/Awesome-GRPO.git
cd Awesome-GRPO/CODE

# (Recommended) Create a new environment
conda create -n awesome-grpo python=3.10 -y
conda activate awesome-grpo

# Install dependencies
pip install -r requirements.txt
# Or install specific versions
pip install deepspeed==0.14.4 vllm==0.6.0.post1 transformers==4.44.0 accelerate==0.33.0
```

> 💡 **Tip:** Ensure your PyTorch + DeepSpeed versions match your CUDA installation.
> For multi-GPU training, verify that `NCCL` is correctly configured (most servers have it by default).



### 2️⃣ Model Download

Place model weights under your local directory (for example, `./models/` ).

```bash
# Main model (used in training config) & Reference model (used in ref_client.py)
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir /data/models/Qwen2.5-1.5B-Instruct
```

> If you don’t have the CLI tool yet:
> `pip install -U "huggingface_hub[cli]"`



## 3️⃣ Update Model Paths in Code

### (A) Main Model Path

In your training configuration file (for example, `configs/grpo_config.py`), make sure the `model_path` points to the correct local directory:

```python
# CODE/configs/grpo_config.py
model_path: str = "/data/models/Qwen2.5-1.5B-Instruct"
```

### (B) Reference Model Path

In `ref_client.py`, update the model path to the downloaded model:

```python
# CODE/ref_client.py
model_path = "/data/models/Qwen2.5-7B-Instruct"
```

> Since it has multiple configurations (e.g., for DAPO or Dr.GRPO), double-check all `model_path` field.



## 4️⃣ Training Commands

Once the environment and model paths are properly set up, you can launch training.

```bash
# Start Running

# Run the following command:
CUDA_VISIBLE_DEVICES=7 python ref_client.py

# Open another bash:
# Run GRPO
CUDA_VISIBLE_DEVICES=2,3,4,5,6 deepspeed train.py --algo grpo

# Run DAPO
CUDA_VISIBLE_DEVICES=2,3,4,5,6 deepspeed train.py --algo dapo

# Run Dr.GRPO
CUDA_VISIBLE_DEVICES=2,3,4,5,6 deepspeed train.py --algo drgrpo

# Run 2-GRPO
CUDA_VISIBLE_DEVICES=2,3,4,5,6 deepspeed train.py --algo 2grpo

# Run GTPO
CUDA_VISIBLE_DEVICES=2,3,4,5,6 deepspeed train.py --algo gtpo

...

```

> You can also append additional arguments, e.g.:
>
> ```bash
> --seed 42 --max_steps 10000 --save_dir outputs/grpo_run1
> ```
>
> DeepSpeed configuration (e.g., ZeRO stage, gradient accumulation, bf16/fp16) follows your `configs/` settings or command-line overrides.



* ✅ **DeepSpeed ZeRO-2/3** for distributed fine-tuning
* ✅ **vLLM inference engine** for efficient generation
* ✅ **KL regularization control** and token-level reward processing
* ✅ **WandB logging** for experiment tracking



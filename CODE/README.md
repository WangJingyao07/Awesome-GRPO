# Awesome-GRPO: Modular RLHF & RLVR Training


This repo is a clean, swappable architecture. You can switch the optimization rule (GRPO or DAPO) via a flag. 

The generation worker (vLLM) runs in a separate process and streams training batches through the Bottle service (`ref_client.py`, which actually serves as the HTTP server).

Protocol compatibility: import helpers from `ref_client.py` (`tensor_to_bytes`, `bytes_to_tensor`, `make_bytes_list`, `bytes_list_to_list`) and use the same `/upload` and `/get` endpoints, so you can drop the code in without changing servers.


## 🆕 Latest Updates

| Date           | Update                                                       |
| -------------- | ------------------------------------------------------------ |
| **2025-10-19** | Added configuration and runnable scripts for **DAPO** and **Dr.GRPO** variant. |
| **2025-10-17** | Integrated **Modularated** core **GRPO** training pipeline with DeepSpeed and vLLM inference. |
| **2025-10-15** | Uploaded key papers and organized the `papers/` directory (GRPO, DAPO, Dr.GRPO, etc.). |
| **2025-09-30** | Repository initialization and documentation setup.           |



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
| **GTPO**      |         ☐          | ✅                        |
| **GRPO-S**    |         ☐          | ✅                        |
| **2-GRPO** |         ☐          | ✅                        |
| **Pref-GRPO** |         ☐          | ✅                        |
| **L2T-GRPO** |         ☐          | ✅                        |
| **EDGE-GRPO** |         ☐          | ✅                        |
| ...           |                    |                          |

✅ means available in `CODE/` and `Paper and Summary/`
☐ means planned for upcoming releases.



## ⚙️ Quick Start for Implementation

**CODE** provides a concise, switchable implementation of GRPO and its variants, allowing to switch optimization rules with a single flag.

You can quickly try each strategy with just two lines of command. We've included a README.md file with CODE.

### Quick Start

```bash
# Clone the repo
git clone https://github.com/WangJingyao07/Awesome-GRPO.git
cd Awesome-GRPO/CODE

# Example: run GRPO training
CUDA_VISIBLE_DEVICES=0,1 deepspeed train.py --algo grpo

# Example: run DAPO
CUDA_VISIBLE_DEVICES=2,3 deepspeed train.py --algo dapo

# Example: run Dr.GRPO
CUDA_VISIBLE_DEVICES=2,3 deepspeed train.py --algo drgrpo
...

```

* ✅ **DeepSpeed ZeRO-2/3** for distributed fine-tuning
* ✅ **vLLM inference engine** for efficient generation
* ✅ **KL regularization control** and token-level reward processing
* ✅ **WandB logging** for experiment tracking



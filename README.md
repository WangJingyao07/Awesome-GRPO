# Awesome-GRPO

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Static Badge](https://img.shields.io/badge/GRPO-green)![Static Badge](https://img.shields.io/badge/to_be_continue-orange)![Repo Clones](https://img.shields.io/badge/Clones-722-blue)![Stars](https://img.shields.io/github/stars/WangJingyao07/Awesome-GRPO)


🔥 A curated and extensible repository for **GRPO** and its **variants**, combining both *code implementations* and *paper collections* for advanced **LLM reinforcement fine-tuning**.



## 🆕 Latest Updates

| Date           | Update                                                       |
| -------------- | ------------------------------------------------------------ |
| **2025-12-10** | Added configuration and runnable scripts for **GMPO** variant. |
| **2025-11-25** | Optimize the CODE for token-level GRPO-style variants. |
| **2025-11-20** | Added More Resources (before 11/2025). |
| **2025-11-07** | Added infer.py for evaluation. |
| **2025-11-06** | Added resources, e.g., GEPO, Flow-GRPO, etc. |
| **2025-11-05** | Added configuration and runnable scripts for **GSPO** variant. |
| **2025-11-04** | Added test wandb report and detailed settings. |
| **2025-10-22** | Added configuration and runnable scripts for **GTPO** variant. |
| **2025-10-19** | Added configuration and runnable scripts for **DAPO**, **Dr.GRPO**, and **2-GRPO** variants. |
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
| **2-GRPO** |         ✅          | ✅                        |
| **GTPO**      |         ✅          | ✅                        |
| **GRPO-S**    |         ✅          | ✅                        |
| **GMPO** |         ✅          | ✅                        |
| **GEPO** |         ✅          | ✅                        |
| **Pref-GRPO** |         ☐          | ✅                        |
| **L2T-GRPO** |         ☐          | ✅                        |
| **TreePO** |         ☐          | ✅                        |
| **GPO** |         ☐          | ✅                        |
| **GiGPO** |         ☐          | ✅                        |
| **Flow-GRPO** |         ☐          | ✅                        |
| **GRPO-SCS** |         ☐          | ✅                        |
| **SGPO** |         ☐          | ✅                        |
| **D(irect)A(dvantage)PO** |         ☐          | ✅                        |
| **D(iversity)A(ware)PO** |         ☐          | ✅                        |
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


# Start Running

# Run the following command:
CUDA_VISIBLE_DEVICES=7 python ref_client.py

# Open another bash:
CUDA_VISIBLE_DEVICES=2,3,4,5,6 deepspeed train.py --algo grpo
CUDA_VISIBLE_DEVICES=2,3,4,5,6 deepspeed train.py --algo dapo
...

```

* ✅ **DeepSpeed ZeRO-2/3** for distributed fine-tuning
* ✅ **vLLM inference engine** for efficient generation
* ✅ **KL regularization control** and token-level reward processing
* ✅ **WandB logging** for experiment tracking





## 📘 Papers and Summary

All relevant works and their detailed analyses are included in the [`papers/`](./papers) directory.

Each PDF corresponds to a variant or conceptual expansion of GRPO, with summaries and comparisons in the subdirectory README.





## 📄 Citation

If you find this repository useful, please consider cite and star our repository (🥰🎉Thanks!!!):

```bibtex
@misc{Awesome-GRPO,
  author = {Wang, Jingyao},
  title = {Awesome-GRPO: A Unified Framework for GRPO and Its Variants},
  year = {2025},
  url = {https://github.com/WangJingyao07/Awesome-GRPO}
}
```
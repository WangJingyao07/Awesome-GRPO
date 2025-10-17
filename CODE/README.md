# Modular RLHF & RLVR Training


This repo is a clean, swappable architecture. You can switch the optimization rule (GRPO or DAPO) via a flag. 

The generation worker (vLLM) runs in a separate process and streams training batches through the Bottle service (`ref_client.py`, which actually serves as the HTTP server).

Protocol compatibility: import helpers from `ref_client.py` (`tensor_to_bytes`, `bytes_to_tensor`, `make_bytes_list`, `bytes_list_to_list`) and use the same `/upload` and `/get` endpoints, so you can drop the code in without changing servers.


## Quick start

```bash
# Install deps
pip install requirements.txt

# Start reference server
CUDA_VISIBLE_DEVICES=7 python ref_client.py

# Launch training
CUDA_VISIBLE_DEVICES=2,3,4,5,6 deepspeed train.py --algo grpo
CUDA_VISIBLE_DEVICES=2,3,4,5,6 deepspeed train.py --algo dapo
...

```

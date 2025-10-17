from dataclasses import dataclass

@dataclass
class Config:
    model_path: str = "/data/users/wjy/models/Qwen2.5-1.5B-Instruct"
    gen_device: int = 1
    ref_server: str = "http://localhost:59875"

    all_steps: int = 1000
    Q_batch_size: int = 5
    num_pre_Q: int = 8
    train_batch_size: int = 1
    gen_update_steps: int = 16
    save_steps: int = 200

    beta: float = 0.04
    compute_gen_logps: bool = True
    clip_param: float = 0.2

def make_ds_config(cfg: "Config") -> dict:
    return {
        "train_micro_batch_size_per_gpu": cfg.train_batch_size,
        "gradient_accumulation_steps": 4,
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-6}},
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "stage3_gather_16bit_weights_on_model_save": True,
            "offload_optimizer": {"device": "cpu"}
        }
    }

# -*- coding: utf-8 -*-
import os, json, re, random, time, importlib, argparse, requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM



import deepspeed

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
gen_device = 1    # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES

from ref_client import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list  


# get batch: use ref_client.py
def get_batch(ref_server_url: str):
    try:
        r = requests.get(f"{ref_server_url}/get").content
        if r == b'empty':
            return None
    except Exception:
        return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0])
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    if len(dd) == 5:
        data['gen_logps'] = bytes_to_tensor(dd[4])
    return data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='grpo', help="choose algorithms")
    # parser.add_argument('--gen', type=str, default='grpo', help="generation worker")
    parser.add_argument('--local_rank', type=int, default=-1)  

    # args = parser.parse_args()
    args, _ = parser.parse_known_args()   
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    algo_mod = importlib.import_module(f"algorithms.{args.algo}")
    # gen_mod = importlib.import_module(f"generation.{args.algo}")
    cfg_mod  = importlib.import_module(f"configs.config_{args.algo}")

    Config = cfg_mod.Config
    make_ds_config = cfg_mod.make_ds_config
    AlgoClass = algo_mod.Algorithm  

    cfg = Config()

    # only rank 0 prints info
    deepspeed.init_distributed()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    if dist.get_rank() == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn', force=True)
        Q = mp.Queue()

        # get gen_worker
        gen_worker = getattr(AlgoClass, "gen_worker", None)
        if gen_worker is None:
            raise RuntimeError(f"Algorithm '{args.algo}' must provide a gen_worker(Q, cfg) function.")

        p = mp.Process(target=gen_worker,
                       args=(Q, cfg.model_path, cfg.gen_device, cfg.ref_server,
                             cfg.num_pre_Q, cfg.train_batch_size, cfg.compute_gen_logps, cfg.Q_batch_size))
        p.start()
    else:
        Q = None

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa"
    )

    ds_config = make_ds_config(cfg)
    engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_config, model=model, model_parameters=model.parameters()
    )

    # load algo
    # algo = AlgoClass(engine=engine,
    #                  tokenizer=tokenizer,
    #                  beta=cfg.beta,
    #                  clip_param=cfg.clip_param,
    #                  compute_gen_logps=cfg.compute_gen_logps)
    algo_kwargs = {
        "beta": cfg.beta,
        "clip_param": cfg.clip_param,
        "compute_gen_logps": cfg.compute_gen_logps,
    }

    if hasattr(cfg, "eps_up"):
        algo_kwargs["eps_up"] = cfg.eps_up
    if hasattr(cfg, "eps_down"):
        algo_kwargs["eps_down"] = cfg.eps_down

    algo = AlgoClass(engine=engine, tokenizer=tokenizer, **algo_kwargs)


    progress = range(1, cfg.all_steps + 1)
    if dist.get_rank() == 0:
        progress = tqdm(progress)

    for step in progress:
        batch = get_batch(cfg.ref_server)
        while batch is None:
            print('waiting for batch...')
            time.sleep(1)
            batch = get_batch(cfg.ref_server)

        loss = algo.step(batch)
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"{args.algo.upper()} | Loss: {loss.item():.6f}")

        if step % cfg.gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                Q.put(state_dict)
                print('[TRAINING PROC] send state_dict ok!')
            dist.barrier()

        if step % cfg.save_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('saving model')
                save_name = f"./Qwen2.5-1.5B-Instruct/step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()

if __name__ == '__main__':
    main()

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


# # Generate sub-processes sampling & scoring
# def gen_worker(Q, model_path: str, gen_device: int, ref_server_url: str,
#                num_pre_Q: int, train_batch_size: int, compute_gen_logps: bool, Q_batch_size: int):
#     os.environ["CUDA_VISIBLE_DEVICES"] = f'{gen_device}'
#     torch.cuda.set_device(0)
#     print(f"Generation worker process uses GPU {gen_device}")


#     # gen_worker
#     from vllm import LLM, SamplingParams
#     from modelscope.msdatasets import MsDataset
#     from math_verify import parse, verify, ExprExtractionConfig
#     from torch.nn.utils.rnn import pad_sequence
#     from transformers import AutoTokenizer


#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.5)
#     ref_server_ver = 'tensor'

#     sampling_params = SamplingParams(n=num_pre_Q, temperature=0.9, max_tokens=700)
#     gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

#     # load data
#     dataset = MsDataset.load('modelscope/gsm8k', subset_name='main', split='train')
#     QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(dataset['question'], dataset['answer'])]

#     # set prompt
#     system_prompt = (
#         "You are a helpful assistant. A conversation between User and Assistant. "
#         "The user asks a question, and the Assistant solves it. The Assistant first thinks "
#         "about the reasoning process in the mind and then provides the user with the answer."
#         "The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, "
#         "respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."
#     )

#     # get answer
#     def gen_answers(prompts):
#         tip_text = []
#         for x in prompts:
#             tip_text.append(tokenizer.apply_chat_template(
#                 [{"role": "system", "content": system_prompt},
#                  {"role": "user", "content": x}],
#                 tokenize=False, add_generation_prompt=True))
#         voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
#         answers, ans_token_ids = [], []
#         for v in voutputs:
#             for z in v.outputs:
#                 answers.append(z.text)
#                 ans_token_ids.append(z.token_ids)
#         return answers, ans_token_ids

#     # calculate reward: correct + format
#     # correct
#     def reward_correct(item, answer):
#         pattern = r'\d+\.\d+|\d+/\d+|\d+'
#         nums = re.findall(pattern, answer)
#         if len(nums) == 0:
#             return -1.0
#         lastnum = nums[-1]
#         ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
#         ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
#         return 1 if verify(ans, ground_truth) else -1

#     # format
#     def reward_format(_item, answer):
#         pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
#         think_count = answer.count("<think>") + answer.count("</think>")
#         answer_count = answer.count("<answer>") + answer.count("</answer>")
#         return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count == 2 and answer_count == 2 else -1


#     def gen_samples(inputs):
#         prompts = [x["Q"] for x in inputs]
#         answers, ans_token_ids = gen_answers(prompts)
#         rewards = []
#         for i, inp in enumerate(inputs):
#             for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
#                 rewards.append(reward_correct(inp, a) + reward_format(inp, a))
#         prompts_text = [tokenizer.apply_chat_template(
#             [{"role": "system", "content": system_prompt},
#              {"role": "user", "content": x}],
#             tokenize=False, add_generation_prompt=True) for x in prompts]
#         return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, ans_token_ids


#     def try_update_model():
#         try:
#             new_state_dict = Q.get_nowait()
#             print('[VLLM PROC] recving new model ...')
#             llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
#             llm_model.load_weights(new_state_dict.items())
#             print('[VLLM PROC] model updated')
#             del new_state_dict
#         except Exception:
#             return

#     for it in range(999999999):
#         if it % 3 == 0:
#             try_update_model()
#         inputs = random.sample(QAs, Q_batch_size)
#         tic = time.time()
#         prompt_inputs, rewards, answers, ans_token_ids = gen_samples(inputs)
#         print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards)
#         if it % 5 == 0:
#             print('answers:', answers[0])

#         # advantage standardization
#         for i, pp in enumerate(prompt_inputs):
#             prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
#             plen = prompt_ids.shape[1]
#             curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
#             curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
#             curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
#             if curr_rewards.max() - curr_rewards.min() < 1e-4:
#                 continue

#             if ref_server_ver == 'tensor':
#                 curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
#                 for ii in range(0, num_pre_Q, train_batch_size):
#                     sub_rewards = curr_rewards[ii:ii+train_batch_size]
#                     sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
#                     tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
#                     from torch.nn.utils.rnn import pad_sequence
#                     output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id)
#                     Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
#                     merged_ids = torch.cat([Qrep, output_ids], dim=1)
#                     data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]

#                     if compute_gen_logps:
#                         zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
#                         zz = [xx.prompt_logprobs[plen:] for xx in zz]
#                         gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
#                         data.append(tensor_to_bytes(gen_logps))

#                     xdata = make_bytes_list(data)
#                     r = requests.post(f"{ref_server_url}/upload", data=xdata)
#                     if r.content == b'string':
#                         ref_server_ver = 'string'
#             elif ref_server_ver == 'string':
#                 xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(),
#                                          tensor_to_bytes(curr_rewards)])
#                 r = requests.post(f"{ref_server_url}/upload", data=xdata)
#                 if r.content == b'tensor':
#                     ref_server_ver = 'tensor'


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

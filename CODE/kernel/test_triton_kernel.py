from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, requests, io, sys, time
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

model_path = "/data2/Qwen/Qwen2.5-7B"
beta = 0.04
num_pre_Q = 8
Q_batch_size = 1

ref_server = "http://localhost:59875"

def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b))
def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    return data


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, 
        torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
engine = model.to('cuda')
opt = torch.optim.AdamW(engine.parameters(), lr=1e-6)


def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    rewards = batch['rewards'].to(engine.device)

    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
    
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    print('logps:', per_token_logps[0,:10])
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    mean_grouped_rewards = rewards.view(-1, num_pre_Q).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, num_pre_Q).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss


batch = get_batch()

model(batch['inputs'].to(engine.device)[:,:1])

tic = time.time()
opt.zero_grad()
loss = GRPO_step(batch)
loss.backward()
print('grad:', model.model.layers[-1].mlp.gate_proj.weight.grad[3])
print(f'torch: {time.time()-tic:3f}s')


from ce_kernel import fast_log_softmax_gather
fast_log_softmax_gather(torch.randn(1, 1, 5).to('cuda'), torch.tensor([[1]]).to('cuda'))
# init
get_per_token_logps = fast_log_softmax_gather

print('\n-----\n')

tic = time.time()
opt.zero_grad()
loss = GRPO_step(batch)
loss.backward()
print('grad:', model.model.layers[-1].mlp.gate_proj.weight.grad[3])
print(f'triton: {time.time()-tic:3f}s')

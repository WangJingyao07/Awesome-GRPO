import torch, json, io, time
from ref_server import raw_queue, result_queue, bytes_list_to_list, tensor_to_bytes, make_bytes_list


from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/data/users/wjy/models/Qwen2.5-1.5B-Instruct"
ref_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
ref_model.eval()
ref_model.requires_grad_(False)

def get_per_token_logps(input_ids):
    logits = ref_model(input_ids).logits  # (B, L, V)
    logits = logits[:, :-1, :]  # (B, L-1, V)
    input_ids = input_ids[:, 1:]  # (B, L-1)
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

while True:
    try:
        d = raw_queue.get()
        dd = bytes_list_to_list(d)
        meta = json.loads(dd[0])
        inputs = bytes_to_tensor(dd[1])
        rewards = bytes_to_tensor(dd[2])
        prompt_length = meta['plen']
        with torch.inference_mode():
            per_token_logps = get_per_token_logps(inputs.to(ref_model.device))
        per_token_logps = per_token_logps[:, prompt_length-1:]
        data = [json.dumps(meta).encode(), tensor_to_bytes(inputs),
                tensor_to_bytes(rewards), tensor_to_bytes(per_token_logps)]
        if len(dd) > 3:
            data.append(dd[3])  
        xdata = make_bytes_list(data)
        result_queue.put(xdata)
        print('Processed batch, put to result_queue')
    except Exception as e:
        print('Batch process error:', e)
        time.sleep(1)

import bottle, queue, torch, io, json, threading, time
from bottle import request

import os

os.environ["WANDB_MODE"] = "disabled"

raw_queue = queue.Queue()
result_queue = queue.Queue()
app = bottle.Bottle()

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()
def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)
def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()
def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

@app.route('/upload', method='POST')
def do_upload():
    dd = request.body.read()
    raw_queue.put(dd)
    print('[SERVER] received batch! Queue size:', raw_queue.qsize())
    return b'tensor'

@app.route('/get', method='GET')
def do_get():
    if result_queue.empty():
        return b'empty'
    print('[SERVER] send batch from result_queue, size:', result_queue.qsize()-1)
    return result_queue.get()

def batch_worker():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = "/data/users/wjy/models/Qwen2.5-7B-Instruct"
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
            print('[SERVER] processed batch, result_queue size:', result_queue.qsize())
        except Exception as e:
            print('[SERVER] Batch process error:', e)
            time.sleep(1)

if __name__ == '__main__':
    worker = threading.Thread(target=batch_worker, daemon=True)
    worker.start()
    print("Launching HTTP server on 59875 ...")
    bottle.run(app, host='0.0.0.0', port=59875, server='tornado')

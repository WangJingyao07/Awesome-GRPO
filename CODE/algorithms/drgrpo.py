# -*- coding: utf-8 -*-
from typing import Dict
import torch
from .base import AlgorithmBase

import os, re, random, time, json, requests

from ref_client import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list 


class Algorithm(AlgorithmBase):

    def __init__(self, engine, tokenizer, beta: float, clip_param: float, compute_gen_logps: bool, **_extra):
        super().__init__(engine, tokenizer, beta=beta, clip_param=clip_param, compute_gen_logps=compute_gen_logps)


    @staticmethod
    # Generate sub-processes sampling & scoring
    def gen_worker(Q, model_path: str, gen_device: int, ref_server_url: str,
                num_pre_Q: int, train_batch_size: int, compute_gen_logps: bool, Q_batch_size: int):
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{gen_device}'
        torch.cuda.set_device(0)
        print(f"Generation worker process uses GPU {gen_device}")


        # gen_worker
        from vllm import LLM, SamplingParams
        from modelscope.msdatasets import MsDataset
        from math_verify import parse, verify, ExprExtractionConfig
        from torch.nn.utils.rnn import pad_sequence
        from transformers import AutoTokenizer


        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.5)
        ref_server_ver = 'tensor'

        sampling_params = SamplingParams(n=num_pre_Q, temperature=0.9, max_tokens=700)
        gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

        # load data
        dataset = MsDataset.load('modelscope/gsm8k', subset_name='main', split='train')
        QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(dataset['question'], dataset['answer'])]

        # set prompt
        system_prompt = (
            "You are a helpful assistant. A conversation between User and Assistant. "
            "The user asks a question, and the Assistant solves it. The Assistant first thinks "
            "about the reasoning process in the mind and then provides the user with the answer."
            "The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, "
            "respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."
        )

        # get answer
        def gen_answers(prompts):
            tip_text = []
            for x in prompts:
                tip_text.append(tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": x}],
                    tokenize=False, add_generation_prompt=True))
            voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
            answers, ans_token_ids = [], []
            for v in voutputs:
                for z in v.outputs:
                    answers.append(z.text)
                    ans_token_ids.append(z.token_ids)
            return answers, ans_token_ids

        # calculate reward: correct + format
        # correct
        def reward_correct(item, answer):
            pattern = r'\d+\.\d+|\d+/\d+|\d+'
            nums = re.findall(pattern, answer)
            if len(nums) == 0:
                return -1.0
            lastnum = nums[-1]
            ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
            ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
            return 1 if verify(ans, ground_truth) else -1

        # format
        def reward_format(_item, answer):
            pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
            think_count = answer.count("<think>") + answer.count("</think>")
            answer_count = answer.count("<answer>") + answer.count("</answer>")
            return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count == 2 and answer_count == 2 else -1


        def gen_samples(inputs):
            prompts = [x["Q"] for x in inputs]
            answers, ans_token_ids = gen_answers(prompts)
            rewards = []
            for i, inp in enumerate(inputs):
                for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
                    rewards.append(reward_correct(inp, a) + reward_format(inp, a))
            prompts_text = [tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                {"role": "user", "content": x}],
                tokenize=False, add_generation_prompt=True) for x in prompts]
            return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, ans_token_ids


        def try_update_model():
            try:
                new_state_dict = Q.get_nowait()
                print('[VLLM PROC] recving new model ...')
                llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(new_state_dict.items())
                print('[VLLM PROC] model updated')
                del new_state_dict
            except Exception:
                return

        for it in range(999999999):
            if it % 3 == 0:
                try_update_model()
            inputs = random.sample(QAs, Q_batch_size)
            tic = time.time()
            prompt_inputs, rewards, answers, ans_token_ids = gen_samples(inputs)
            print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards)
            if it % 5 == 0:
                print('answers:', answers[0])

            # advantage standardization
            for i, pp in enumerate(prompt_inputs):
                prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
                plen = prompt_ids.shape[1]
                curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
                if curr_rewards.max() - curr_rewards.min() < 1e-4:
                    continue

                if ref_server_ver == 'tensor':
                    # remove std
                    curr_rewards = (curr_rewards - curr_rewards.mean())
                    for ii in range(0, num_pre_Q, train_batch_size):
                        sub_rewards = curr_rewards[ii:ii+train_batch_size]
                        sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                        tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                        from torch.nn.utils.rnn import pad_sequence
                        output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id)
                        Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                        merged_ids = torch.cat([Qrep, output_ids], dim=1)
                        data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]

                        if compute_gen_logps:
                            zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                            zz = [xx.prompt_logprobs[plen:] for xx in zz]
                            gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                            data.append(tensor_to_bytes(gen_logps))

                        xdata = make_bytes_list(data)
                        r = requests.post(f"{ref_server_url}/upload", data=xdata)
                        if r.content == b'string':
                            ref_server_ver = 'string'
                elif ref_server_ver == 'string':
                    xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(),
                                            tensor_to_bytes(curr_rewards)])
                    r = requests.post(f"{ref_server_url}/upload", data=xdata)
                    if r.content == b'tensor':
                        ref_server_ver = 'tensor'


    @staticmethod
    def _get_per_token_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def step(self, batch: Dict) -> torch.Tensor:
        engine = self.engine
        tokenizer = self.tokenizer

        prompt_length = batch['plen']
        inputs = batch['inputs'].to(engine.device)
        advantages = batch['rewards'].to(engine.device).unsqueeze(1)

        logits = engine(inputs).logits
        logits = logits[:, :-1, :]  # (B, L-1, V)
        input_ids = inputs[:, 1:]   # (B, L-1)

        per_token_logps = self._get_per_token_logps(logits, input_ids)
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        ref_per_token_logps = batch['refs'].to(per_token_logps.device)

        d = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(d) - d - 1

        completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

        if 'gen_logps' in batch:
            ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
            per_token_obj = torch.min(ratio * advantages, clipped_ratio * advantages)
        else:
            per_token_obj = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
            assert self.compute_gen_logps is False

        per_token_loss = -(per_token_obj - self.beta * per_token_kl)

        # remove sequence normalization
        loss = (per_token_loss * completion_mask).sum(dim=1).mean()
        return loss

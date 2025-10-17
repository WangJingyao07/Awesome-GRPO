# -*- coding: utf-8 -*-
from typing import Dict
import torch
from .base import AlgorithmBase

class Algorithm(AlgorithmBase):

    def __init__(self, engine, tokenizer, beta: float, clip_param: float, compute_gen_logps: bool, **_extra):
        super().__init__(engine, tokenizer, beta=beta, clip_param=clip_param, compute_gen_logps=compute_gen_logps)

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
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        return loss

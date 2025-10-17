
import torch

def get_per_token_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Compute per-token log-probs via stable log_softmax + gather.
    logits: (B, L, V) over tokens 0..L-1 for next-token predictions
    input_ids: (B, L) corresponding labels
    returns: (B, L) token-wise logp
    """
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


#!/usr/bin/env python3

# Example usage:
# python infer.py --model_dir /path/to/model --prompt "Explain GRPO in one sentence." --device cuda \
#     --max_new_tokens 128 --temperature 0.2 --top_k 5 --output ./preds.jsonl
#
# python infer.py --model_dir /path/to/model --prompts_file prompts.txt --device cpu --batch_size 4

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for a HF-style causal LM.")
    parser.add_argument("--model_dir", required=True,
                        help="Path or HF id of the model directory (must be compatible with AutoModelForCausalLM).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="Single prompt string.")
    group.add_argument("--prompts_file", type=str,
                       help="Path to a text file with one prompt per line (empty lines ignored).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device string, e.g. 'cuda', 'cuda:0', or 'cpu'.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of prompts to process per loop iteration (we still generate per-example to keep outputs clean).")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature. 0 -> greedy (do_sample=False).")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences returned per prompt.")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling. If not set and temperature==0, greedy decoding is used.")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Use fast tokenizer if available.")
    parser.add_argument("--use_fp16", action="store_true", help="Try to load model in fp16 (if GPU available).")
    parser.add_argument("--output", type=str, default="preds.jsonl", help="Output JSONL file (one JSON per line).")
    parser.add_argument("--save_raw_sequences", action="store_true",
                        help="Save full model-generated sequences (including prompt) in addition to the extracted continuation.")
    # Advanced / optional:
    parser.add_argument("--deepspeed_ckpt", type=str, default=None,
                        help="Path to DeepSpeed inference checkpoint (optional). If provided, user must have deepspeed and follow DS init flow.")
    # Safety / debugging:
    parser.add_argument("--max_prompt_len", type=int, default=1024,
                        help="(Optional) truncate prompts longer than this many tokens to avoid OOM.")
    return parser.parse_args()


def read_prompts(args) -> List[str]:
    if args.prompt:
        return [args.prompt]
    else:
        p = Path(args.prompts_file)
        if not p.exists():
            raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")
        prompts = []
        with p.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    prompts.append(line)
        return prompts


def try_load_model_and_tokenizer(model_dir: str, device: str, use_fp16: bool, use_fast_tokenizer: bool):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=use_fast_tokenizer)
    # ensure pad token exists
    if tokenizer.pad_token is None:
        # many causal LM tokenizers lack pad_token; set it to eos_token
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    # Try device_map auto if transformers/newer version and multiple GPUs
    try:
        if use_fp16 and ("cuda" in device) and torch.cuda.is_available():
            # try to load in float16 for efficiency on GPU(s)
            model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
        # If device_map='auto' succeeded, the model may already be on GPU; still set eval()
        model.eval()
        return model, tokenizer
    except Exception:
        # fallback: load to CPU then move
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir)
            model.eval()
            # move to desired device if single device string (not 'auto')
            try:
                model.to(device)
            except Exception:
                # if move fails (e.g., multi-GPU expected), keep on CPU and warn
                print(f"[warning] failed to move model to {device}. Model remains on CPU.")
            return model, tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_dir}: {e}")


def generate_continuation_for_prompt(model, tokenizer, prompt: str, device: str, args) -> Dict:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_len)
    input_ids = inputs["input_ids"].to(model.device if hasattr(model, "device") else device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device if hasattr(model, "device") else device)

    # prepare generation kwargs
    do_sample = args.do_sample or (args.temperature > 0.0)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else 1.0,
        top_k=args.top_k if do_sample else None,
        top_p=args.top_p if do_sample else None,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    # call generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )

    # outputs shape: (num_return_sequences, seq_len)
    results = []
    input_len = input_ids.shape[1]
    for seq_idx in range(outputs.shape[0]):
        seq = outputs[seq_idx].tolist()

        # extract generated piece only (tokens after input_len)
        if len(seq) <= input_len:
            gen_tokens = []
        else:
            gen_tokens = seq[input_len:]
        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Also keep the full decoded sequence if requested (may duplicate prompt)
        full_text = tokenizer.decode(seq, skip_special_tokens=True).strip()
        res = {
            "prompt": prompt,
            "completion": generated_text,
            "full_text": full_text if args.save_raw_sequences else None,
            "meta": {
                "input_length": int(input_len),
                "generated_token_count": len(gen_tokens),
            },
        }
        results.append(res)

    # if single sequence requested, return the first result (for easier downstream use)
    return results if len(results) > 1 else results[0]



# Main loop
def main():
    args = parse_args()

    # validate device
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # read prompts
    prompts = read_prompts(args)
    if len(prompts) == 0:
        raise ValueError("No prompts provided.")

    # load model and tokenizer
    if args.deepspeed_ckpt:
        raise NotImplementedError(
            "DeepSpeed checkpoint specified but DeepSpeed inference flow is not implemented in this script. "
            "Please either provide a HF-compatible model directory, or ask me to generate a DeepSpeed inference script."
        )

    model, tokenizer = try_load_model_and_tokenizer(args.model_dir, device, args.use_fp16, args.use_fast_tokenizer)


    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results_all = []
    # process prompts in batches
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        for prompt in batch:
            try:
                res = generate_continuation_for_prompt(model, tokenizer, prompt, device, args)

                if isinstance(res, list):
                    for r in res:
                        results_all.append(r)
                        # print to stdout for quick inspection
                        print(f"> PROMPT: {r['prompt']}")
                        print(f"> COMPLETION: {r['completion']}\n")
                else:
                    results_all.append(res)
                    print(f"> PROMPT: {res['prompt']}")
                    print(f"> COMPLETION: {res['completion']}\n")
            except RuntimeError as e:
                # capture OOM or other runtime errors and continue
                print(f"[error] generation failed for prompt: {prompt[:80]!r}... error: {e}")
                # try a smaller generation size as a fallback
                if "out of memory" in str(e).lower() and device != "cpu":
                    print("[info] Out of memory. Retry on CPU with smaller max_new_tokens.")
                    # move model to CPU and retry (best-effort)
                    model.to("cpu")
                    args_backup = args.max_new_tokens
                    args.max_new_tokens = max(8, args.max_new_tokens // 2)
                    try:
                        res = generate_continuation_for_prompt(model, tokenizer, prompt, "cpu", args)
                        if isinstance(res, list):
                            for r in res:
                                results_all.append(r)
                        else:
                            results_all.append(res)
                    except Exception as e2:
                        print(f"[error] retry failed: {e2}")
                    finally:
                        args.max_new_tokens = args_backup
                else:
                    # record the failure item with error
                    results_all.append({"prompt": prompt, "completion": None, "error": str(e)})

    # write JSONL output
    with out_path.open("w", encoding="utf-8") as fout:
        for item in results_all:
            if item.get("full_text") is None:
                item.pop("full_text", None)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(results_all)} records to {out_path.resolve()}.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Utility to dump the *first* Transformer layer of
`codellama/CodeLlama-7b-hf` (or any compatible Llama-based checkpoint)
to a standalone PyTorch file.  The resulting .pt contains only four
keys – q_proj, k_proj, v_proj, o_proj – each mapped to its weight
matrix with shape (d, d).

Example:

    python save_first_layer.py --model codellama/CodeLlama-7b-hf \
                               --out layer0.pt

The produced `layer0.pt` can later be supplied to `run_recover_R.py`
via  `--layer_state layer0.pt`  to execute the lightweight, single-layer
recovery path (no 7-B model download required).
"""

import argparse, torch
from transformers import AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="codellama/CodeLlama-7b-hf",
                   help="Model name or local path (default: CodeLlama-7B)")
    p.add_argument("--out", default="layer0.pt",
                   help="Where to save the layer state dict (default: layer0.pt)")
    return p.parse_args()


def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )

    layer = model.model.layers[0]
    state = {
        "q_proj": layer.self_attn.q_proj.weight.data.cpu().clone(),
        "k_proj": layer.self_attn.k_proj.weight.data.cpu().clone(),
        "v_proj": layer.self_attn.v_proj.weight.data.cpu().clone(),
        "o_proj": layer.self_attn.o_proj.weight.data.cpu().clone(),
    }

    torch.save(state, args.out)
    print(f"Saved first layer to {args.out} (keys: {list(state)})")


if __name__ == "__main__":
    main()

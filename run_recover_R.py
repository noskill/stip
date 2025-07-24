#!/usr/bin/env python3
"""
Generate masked weight pairs (W, W') for R-recovery experiments as described in recover_R.txt.

If CUDA is available, gradient-based recovery computations will run on GPU.

This script loads a pretrained Llama checkpoint, selects a single Transformer layer,
adds dense Gaussian noise ΔW to its four projection matrices (q, k, v, o) at a specified
Frobenius-norm ratio, generates a random orthogonal matrix R, computes W' = Rᵀ (W + ΔW) R,
and saves the original and masked weights for downstream recovery of R.
"""
import argparse
import torch
import torch.optim
from transformers import AutoConfig, AutoModelForCausalLM

from utils import better_add_dense_noise, better_rope_preserving_R, _rotate_model_parameters_with_R
from utils import dense_orthogonal_R


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate (W, W') pairs for R-recovery experiments on one Transformer layer"
    )
    p.add_argument(
        "--model", required=True,
        help="Name or path of the pretrained Llama checkpoint",
    )
    p.add_argument(
        "--layer", type=int, default=0,
        help="Index of the Transformer layer to corrupt (0-based)",
    )
    p.add_argument(
        "--noise_ratio", type=float, default=0.05,
        help="Relative noise ratio: ΔW Frobenius norm ≈ noise_ratio * ||W||_F",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Optional random seed for reproducibility",
    )
    p.add_argument(
        "--output", default=None,
        help="Output path for the saved weight pairs (torch .pt file)",
    )
    p.add_argument(
        "--recover_steps", type=int, default=100,
        help="Number of gradient descent steps to recover R",
    )
    p.add_argument(
        "--lr", type=float, default=0.1,
        help="Learning rate for the optimizer",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and model on CPU in float32
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()

    # Select layer and clone original weights
    try:
        layer = model.model.layers[args.layer]
    except (AttributeError, IndexError):
        raise ValueError(f"Layer index out of range: {args.layer}")

    Wq = layer.self_attn.q_proj.weight.data.clone()
    Wk = layer.self_attn.k_proj.weight.data.clone()
    Wv = layer.self_attn.v_proj.weight.data.clone()
    Wo = layer.self_attn.o_proj.weight.data.clone()

    # Optionally fix randomness
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Add dense Gaussian noise ΔW to all 2D weights (will affect selected layer among others)
    # better_add_dense_noise(model, noise_ratio=args.noise_ratio)

    # Build a block-diagonal orthogonal R preserving RoPE pairs
    head_dim = config.hidden_size // config.num_attention_heads
    R = better_rope_preserving_R(
        config.num_attention_heads,
        head_dim,
        max_angle=1.0,
        dtype=torch.float32,
        device=device,
    )
#    R = dense_orthogonal_R(model.config.hidden_size, dtype=torch.float32, device="cuda", seed=42)

    # Apply rotation to all model weights in-place
    _rotate_model_parameters_with_R(model, R)

    # Extract masked weights W' from the same layer
    Wq_p = layer.self_attn.q_proj.weight.data.clone()
    Wk_p = layer.self_attn.k_proj.weight.data.clone()
    Wv_p = layer.self_attn.v_proj.weight.data.clone()
    Wo_p = layer.self_attn.o_proj.weight.data.clone()

    # Prepare output dict
    out = {
        'layer': args.layer,
        'noise_ratio': args.noise_ratio,
        'seed': args.seed,
        'R': R,
        'Wq': Wq,
        'Wk': Wk,
        'Wv': Wv,
        'Wo': Wo,
        'Wq_p': Wq_p,
        'Wk_p': Wk_p,
        'Wv_p': Wv_p,
        'Wo_p': Wo_p,
    }

    # Recover R via gradient descent (minimize sum ||W'_i − Rᵀ W_i R||²_F with R ∈ O(d))
    d = Wq.size(0)
    W_list = [Wq.to(device), Wk.to(device), Wv.to(device), Wo.to(device)]
    Wp_list = [Wq_p.to(device), Wk_p.to(device), Wv_p.to(device), Wo_p.to(device)]
    R = R.to(device)
    R_est = torch.eye(d, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([R_est], lr=args.lr)
    for step in range(args.recover_steps):
        optimizer.zero_grad()
        loss = sum((Wp - R_est.t() @ W @ R_est).pow(2).sum() for W, Wp in zip(W_list, Wp_list))
        loss.backward()
        optimizer.step()

        print(f"Step {step+1}/{args.recover_steps}, loss={loss.item():.4e}")
 
        # Re-project to nearest orthogonal matrix via SVD
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(R_est, full_matrices=False)
            R_est.copy_(U @ Vh)
            cosines = torch.abs(torch.diagonal(R_est.t() @ R))
            sim = cosines.mean().item()
            print(f"Recovered R similarity (average |cos|): {sim:.6f}")


    with torch.no_grad():
        cosines = torch.abs(torch.diagonal(R_est.t() @ R))
        sim = cosines.mean().item()
    print(f"Recovered R similarity (average |cos|): {sim:.6f}")
    out['R_est'] = R_est.detach()

    # Determine output path and save all data
    out_path = args.output or f"recover_layer{args.layer}_noise{int(args.noise_ratio*100)}pct.pt"
    torch.save(out, out_path)
    print(f"Saved masked weight pairs and recovered R to {out_path}")


if __name__ == '__main__':
    main()

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
from utils import dense_orthogonal_R, rope_incompatible_head_R


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
        "--layer_state",
        default=None,
        help="Optional .pt file with a saved Transformer layer (dict with q_proj, k_proj, v_proj, o_proj weights). "
             "If provided, the script skips loading the full model and operates on this layer only.",
    )
    p.add_argument(
        "--recover_steps", type=int, default=100,
        help="Number of gradient descent steps to recover R",
    )
    p.add_argument(
        "--lr", type=float, default=0.1,
        help="Learning rate for the optimizer",
    )
    p.add_argument(
        "--optimizer",
        choices=["sgd", "adam"],
        default="sgd",
        help="Optimizer to use for recovering R (sgd | adam).",
    )
    p.add_argument(
        "--orth_penalty", type=float, default=1.0,
        help="Weight λ for orthogonality penalty term λ·‖RᵀR−I‖²_F",
    )
    p.add_argument(
        "--proj_every", type=int, default=10,
        help="Perform SVD-based projection every N steps (0 disables)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Obtain the clean layer weights  (either from checkpoint or .pt)
    # ------------------------------------------------------------------
    # Always load the small JSON-only config to know hidden_size / n_heads.
    config = AutoConfig.from_pretrained(args.model)

    if args.layer_state is None:
        # --- full-model path ( original behaviour ) --------------------
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).eval()

        try:
            layer = model.model.layers[args.layer]
        except (AttributeError, IndexError):
            raise ValueError(f"Layer index out of range: {args.layer}")

        Wq = layer.self_attn.q_proj.weight.data.clone()
        Wk = layer.self_attn.k_proj.weight.data.clone()
        Wv = layer.self_attn.v_proj.weight.data.clone()
        Wo = layer.self_attn.o_proj.weight.data.clone()
    else:
        # --- lightweight path: load from .pt ---------------------------
        state = torch.load(args.layer_state, map_location="cpu")

        # Accept either raw tensors or state-dict style keys ending with ".weight"
        def _get(name):
            if name in state:
                return state[name]
            if f"{name}.weight" in state:
                return state[f"{name}.weight"]
            raise KeyError(f"Missing key '{name}' in {args.layer_state}")

        Wq = _get("q_proj").clone()
        Wk = _get("k_proj").clone()
        Wv = _get("v_proj").clone()
        Wo = _get("o_proj").clone()

        # No full model present, so set model=None for the rest of the script
        model = None

    # Optionally fix randomness
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # 2. Add Gaussian noise  ΔW  with given Frobenius ratio
    # ------------------------------------------------------------------
    if model is not None:
        # full-model: reuse existing helper that works in-place on model
        better_add_dense_noise(model, noise_ratio=args.noise_ratio)
    else:
        def _add_noise(W: torch.Tensor) -> torch.Tensor:
            fro = W.norm(p='fro')
            sigma = args.noise_ratio * fro / (W.numel() ** 0.5)
            return W + torch.randn_like(W) * sigma

        Wq_noisy = _add_noise(Wq)
        Wk_noisy = _add_noise(Wk)
        Wv_noisy = _add_noise(Wv)
        Wo_noisy = _add_noise(Wo)

    # ------------------------------------------------------------------
    # 3. Build orthogonal R and apply it
    # ------------------------------------------------------------------
    head_dim = config.hidden_size // config.num_attention_heads
    R = rope_incompatible_head_R(
        config.num_attention_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device=device,
    )

    # R = better_rope_preserving_R(
    #     config.num_attention_heads,
    #     head_dim,
    #     max_angle=1.0,
    #     dtype=torch.float32,
    #     device=device,
    # )
#    R = dense_orthogonal_R(model.config.hidden_size, dtype=torch.float32, device="cuda", seed=42)

    if model is not None:
        # Apply rotation to all model weights in-place
        _rotate_model_parameters_with_R(model, R)

        # Extract masked weights W' from the same layer
        Wq_p = layer.self_attn.q_proj.weight.data.clone()
        Wk_p = layer.self_attn.k_proj.weight.data.clone()
        Wv_p = layer.self_attn.v_proj.weight.data.clone()
        Wo_p = layer.self_attn.o_proj.weight.data.clone()
    else:
        # Stand-alone layer: compute W' directly
        # Ensure all tensors live on the same device as R
        dev = R.device
        Wq_noisy = Wq_noisy.to(dev)
        Wk_noisy = Wk_noisy.to(dev)
        Wv_noisy = Wv_noisy.to(dev)
        Wo_noisy = Wo_noisy.to(dev)
        Wq = Wq.to(dev)
        Wk = Wk.to(dev)
        Wv = Wv.to(dev)
        Wo = Wo.to(dev)

        Rt = R.t()
        Wq_p = Rt @ Wq_noisy @ R
        Wk_p = Rt @ Wk_noisy @ R
        Wv_p = Rt @ Wv_noisy @ R
        Wo_p = Rt @ Wo_noisy @ R

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
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD([R_est], lr=args.lr)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam([R_est], lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    eye = torch.eye(d, device=device, dtype=torch.float32)
    for step in range(args.recover_steps):
        optimizer.zero_grad()

        # Reconstruction loss
        rec_loss = sum((Wp - R_est.t() @ W @ R_est).pow(2).sum() for W, Wp in zip(W_list, Wp_list))

        # Orthogonality penalty  ‖RᵀR − I‖²_F
        ortho_loss = (R_est.t() @ R_est - eye).pow(2).sum()

        loss = rec_loss + args.orth_penalty * ortho_loss
        loss.backward()
        optimizer.step()

        if args.proj_every > 0 and (step + 1) % args.proj_every == 0:
            # Periodic re-projection via SVD
            with torch.no_grad():
                U, _, Vh = torch.linalg.svd(R_est, full_matrices=False)
                R_est.copy_(U @ Vh)

        # Logging
        with torch.no_grad():
            cosines = torch.abs(torch.diagonal(R_est.t() @ R))
            sim = cosines.mean().item()
        print(
            f"Step {step+1}/{args.recover_steps} | total={loss.item():.4e} "
            f"rec={rec_loss.item():.4e} ortho={ortho_loss.item():.4e} | sim={sim:.6f}"
        )


    # Final projection to ensure perfectly orthogonal output
    with torch.no_grad():
        U, _, Vh = torch.linalg.svd(R_est, full_matrices=False)
        R_est.copy_(U @ Vh)
        cosines = torch.abs(torch.diagonal(R_est.t() @ R))
        sim = cosines.mean().item()
    print(f"Final recovered R similarity (average |cos|): {sim:.6f}")
    out['R_est'] = R_est.detach()

    # Determine output path and save all data
    out_path = args.output or f"recover_layer{args.layer}_noise{int(args.noise_ratio*100)}pct.pt"
    torch.save(out, out_path)
    print(f"Saved masked weight pairs and recovered R to {out_path}")


if __name__ == '__main__':
    main()

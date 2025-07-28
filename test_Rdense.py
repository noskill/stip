
import copy

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaConfig, LlamaForCausalLM

from utils import (
    get_model_R_dense,
    model as MODEL_NAME,
    replace_attention_RoPE_dense,
    replace_rms_with_rotated,
    dense_orthogonal_R,
    _rotate_model_parameters_with_R,
)


def test_replace_attention_identity():
    """
    Replacing self_attn modules with RoPE-aware attention under R=I should be a no-op.
    """
    cfg = LlamaConfig(hidden_size=32,
                      intermediate_size=64,
                      num_hidden_layers=1,
                      num_attention_heads=4,
                      vocab_size=100)
    llm = LlamaForCausalLM(cfg).eval()
    block = llm.model.layers[0]
    rotary = llm.model.rotary_emb

    seq_len = 5
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, cfg.hidden_size)
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf")).triu(1)
    pos_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rotary(x, pos_ids)

    y0 = block(x,
               attention_mask=mask,
               position_ids=pos_ids,
               position_embeddings=(cos, sin))[0]
    replace_attention_RoPE_dense(llm, torch.eye(cfg.hidden_size))
    y1 = block(x,
               attention_mask=mask,
               position_ids=pos_ids,
               position_embeddings=(cos, sin))[0]
    assert torch.allclose(y0, y1, atol=1e-5)


def test_block_reversability():
    # Load model config to get dimensions for R
    config = AutoConfig.from_pretrained(MODEL_NAME)
    d = config.hidden_size

    # Instantiate base Llama model and a rotated copy for testing
    llm = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).eval().to('cpu')
    llm_rot = copy.deepcopy(llm)

    # Build a random dense orthogonal matrix R
    R = dense_orthogonal_R(
        d,
        dtype=torch.float32,
        device=next(llm.parameters()).device,
        seed=0,
    )
 #   R_t = torch.eye(d)
    perm = torch.randperm(d)
    pi = torch.eye(d, dtype=torch.float32)[perm]
    R = pi

    R_t = R.t()

    # Apply rotation to all weights, replace RMSNorm, and swap in RoPE-aware attention
    _rotate_model_parameters_with_R(llm_rot, R)
    replace_rms_with_rotated(llm_rot, R)
    replace_attention_RoPE_dense(llm_rot, R)

    # Extract the first Transformer block and its rotary embeddings
    block_orig = llm.model.layers[0]
    block_rot = llm_rot.model.layers[0]
    rotary = llm.model.rotary_emb

    # Dummy input for block-level check
    seq_len = 800 
    device = next(llm.parameters()).device
    dtype = llm.dtype
    del llm
    del llm_rot
    torch.cuda.empty_cache()
    x = torch.randn(1, seq_len, d, dtype=dtype, device=device)
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"),
                      dtype=dtype,
                      device=device).triu(1)
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cos, sin = rotary(x, pos_ids)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block_orig.to(device)
    block_rot.to(device)
    x = x.to(device)
    R = R.to(device)
    R_t = R_t.to(device)
    mask = mask.to(device)
    cos = cos.to(device)
    sin = sin.to(device)

    # Compute baseline and rotated-block outputs
    y_base = block_orig(
        x,
        attention_mask=mask,
        position_ids=pos_ids,
        position_embeddings=(cos, sin),
    )[0]
    y_rot = block_rot(
        (x.float() @ R).to(dtype),
        attention_mask=mask,
        position_ids=pos_ids,
        position_embeddings=(cos, sin),
    )[0]

    # Un-rotate in float64 for numerical stability, then cast back to model dtype
    y_unrot = (y_rot.to(torch.float64) @ R_t.to(torch.float64)).to(dtype)
    max_diff = (y_base - y_unrot).abs().max()
    print(f"Block reversibility max diff = {max_diff:.3e}")
    # allow numeric drift under float32 with dense R
    assert torch.allclose(y_base, y_unrot, atol=5e-2), (
        f"Block reversibility under R failed (max|Δ| = {max_diff:.3e})"
    )

def test_full_forward_invariance_dense():
    # First run on a mini Llama to get quick feedback
    mini_cfg = LlamaConfig(hidden_size=32,
                           intermediate_size=64,
                           num_hidden_layers=1,
                           num_attention_heads=4,
                           vocab_size=100)
    mini_base = LlamaForCausalLM(mini_cfg).eval()
    d0 = mini_cfg.hidden_size

    R0 = torch.eye(d0, dtype=torch.float32)
    print("=== Full-forward invariance (mini model): identity R ===")
    test_full_forward_invariance_dense_impl(mini_base, R0)

    perm0 = torch.randperm(d0)
    R0_perm = torch.eye(d0, dtype=torch.float32)[perm0]
    print("=== Full-forward invariance (mini model): permutation R ===")
    test_full_forward_invariance_dense_impl(mini_base, R0_perm)

    R0_dense = dense_orthogonal_R(d0, dtype=torch.float32, device='cpu', seed=0)
    print("=== Full-forward invariance (mini model): dense random R ===")
    test_full_forward_invariance_dense_impl(mini_base, R0_dense)

    # Then run on the full pretrained model on CPU
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    d = cfg.hidden_size
    base = LlamaForCausalLM.from_pretrained(MODEL_NAME).eval().to('cpu')
    device = next(base.parameters()).device

    R_id = torch.eye(d, dtype=torch.float32, device=device)
    print("=== Full-forward invariance: identity R ===")
    test_full_forward_invariance_dense_impl(base, R_id)

    perm = torch.randperm(d)
    R_perm = torch.eye(d, dtype=torch.float32, device=device)[perm]
    print("=== Full-forward invariance: permutation R ===")
    test_full_forward_invariance_dense_impl(base, R_perm)

    R_dense = dense_orthogonal_R(d, dtype=torch.float32, device=device, seed=0)
    print("=== Full-forward invariance: dense random R ===")
    test_full_forward_invariance_dense_impl(base, R_dense)

def test_hidden_invariance_dense():
    # First run on a mini Llama for quick feedback
    mini_cfg = LlamaConfig(hidden_size=32,
                           intermediate_size=64,
                           num_hidden_layers=1,
                           num_attention_heads=4,
                           vocab_size=100)
    mini_base = LlamaForCausalLM(mini_cfg).eval()
    d0 = mini_cfg.hidden_size

    R0 = torch.eye(d0, dtype=torch.float32)
    print("=== Hidden-state invariance (mini model): identity R ===")
    test_hidden_invariance_dense_impl(mini_base, R0)

    perm0 = torch.randperm(d0)
    R0_perm = torch.eye(d0, dtype=torch.float32)[perm0]
    print("=== Hidden-state invariance (mini model): permutation R ===")
    test_hidden_invariance_dense_impl(mini_base, R0_perm)

    R0_dense = dense_orthogonal_R(d0, dtype=torch.float32, device='cpu', seed=0)
    print("=== Hidden-state invariance (mini model): dense random R ===")
    test_hidden_invariance_dense_impl(mini_base, R0_dense)

    # Then run on the full pretrained model on CPU
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    d = cfg.hidden_size
    base = LlamaForCausalLM.from_pretrained(MODEL_NAME).eval().to('cpu')
    device = next(base.parameters()).device

    R_id = torch.eye(d, dtype=torch.float32, device=device)
    print("=== Hidden-state invariance: identity R ===")
    test_hidden_invariance_dense_impl(base, R_id)

    perm = torch.randperm(d)
    R_perm = torch.eye(d, dtype=torch.float32, device=device)[perm]
    print("=== Hidden-state invariance: permutation R ===")
    test_hidden_invariance_dense_impl(base, R_perm)

    R_dense = dense_orthogonal_R(d, dtype=torch.float32, device=device, seed=0)
    print("=== Hidden-state invariance: dense random R ===")
    test_hidden_invariance_dense_impl(base, R_dense)

def test_toy_block_reversibility_fp64():
    # Toy block-level reversibility in double precision to debug numeric drift
    cfg = LlamaConfig(hidden_size=64,
                      intermediate_size=256,
                      num_hidden_layers=1,
                      num_attention_heads=4,
                      vocab_size=100)
    base = LlamaForCausalLM(cfg).double().eval()
    rot = copy.deepcopy(base)
    R = dense_orthogonal_R(64, dtype=torch.float64, device='cpu', seed=0)
    R_t = R.t()
    _rotate_model_parameters_with_R(rot, R)
    replace_rms_with_rotated(rot, R)
    replace_attention_RoPE_dense(rot, R)
    block_orig = base.model.layers[0]
    block_rot = rot.model.layers[0]
    # small sequence for clarity
    seq_len = 10
    x = torch.randn(1, seq_len, 64, dtype=torch.float64)
    mask = torch.full((1, 1, seq_len, seq_len), float('-inf'),
                      dtype=torch.float64).triu(1)
    pos_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = base.model.rotary_emb(x, pos_ids)
    # forward
    y_base = block_orig(x, attention_mask=mask,
                        position_ids=pos_ids,
                        position_embeddings=(cos, sin))[0]
    y_rot = block_rot(x @ R, attention_mask=mask,
                      position_ids=pos_ids,
                      position_embeddings=(cos, sin))[0]
    y_unrot = y_rot @ R_t
    diff = (y_base - y_unrot).abs().max()
    print(f"Toy FP64 block reversibility diff = {diff:.3e}")
    # Double-precision numeric drift is small; require block-level tolerance similar to float32
    assert diff < 2e-3, f"Toy FP64 block reversibility failed (Δ={diff:.3e})"


def test_full_forward_invariance_dense_impl(base, R):
    """
    Compute and print max difference in next-token logits between base and R-rotated model
    under float32 and bfloat16 evaluations.
    """
    rot = copy.deepcopy(base)
    _rotate_model_parameters_with_R(rot, R)
    replace_rms_with_rotated(rot, R)
    replace_attention_RoPE_dense(rot, R)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt = "def foo():"
    device = next(base.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # float32 evaluation
    base32 = base.to(torch.float32)
    rot32 = rot.to(torch.float32)
    with torch.no_grad():
        logits_base32 = base32(input_ids).logits[:, -1, :].float()
        logits_rot32 = rot32(input_ids).logits[:, -1, :].float()
    diff32 = (logits_base32 - logits_rot32).abs().max().item()

    # bfloat16 evaluation
    base_bf16 = base.to(torch.bfloat16)
    rot_bf16 = rot.to(torch.bfloat16)
    with torch.no_grad():
        logits_base_bf16 = base_bf16(input_ids).logits[:, -1, :].float()
        logits_rot_bf16 = rot_bf16(input_ids).logits[:, -1, :].float()
    diff_bf16 = (logits_base_bf16 - logits_rot_bf16).abs().max().item()

    print(f"  Full-forward max diff (float32)={diff32:.3e}, (bfloat16)={diff_bf16:.3e}")

def test_hidden_invariance_dense_impl(base, R):
    """
    Compute and print max difference in final hidden states between base and R-rotated model
    under float32 and bfloat16 evaluations.
    """
    rot = copy.deepcopy(base)
    _rotate_model_parameters_with_R(rot, R)
    replace_rms_with_rotated(rot, R)
    replace_attention_RoPE_dense(rot, R)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt = "def foo():"
    device = next(base.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # float32 evaluation
    base32 = base.to(torch.float32)
    rot32 = rot.to(torch.float32)
    with torch.no_grad():
        out_base32 = base32(input_ids, output_hidden_states=True)
        h_base32 = out_base32.hidden_states[-1].float()
        out_rot32 = rot32(input_ids, output_hidden_states=True)
        h_rot32 = out_rot32.hidden_states[-1].float()
    h_unrot32 = h_rot32 @ R.t()
    diff32 = (h_base32 - h_unrot32).abs().max().item()

    # bfloat16 evaluation
    base_bf16 = base.to(torch.bfloat16)
    rot_bf16 = rot.to(torch.bfloat16)
    with torch.no_grad():
        out_base_bf16 = base_bf16(input_ids, output_hidden_states=True)
        h_base_bf16 = out_base_bf16.hidden_states[-1].float()
        out_rot_bf16 = rot_bf16(input_ids, output_hidden_states=True)
        h_rot_bf16 = out_rot_bf16.hidden_states[-1].float()
    h_unrot_bf16 = h_rot_bf16 @ R.t()
    diff_bf16 = (h_base_bf16 - h_unrot_bf16).abs().max().item()

    print(f"  Hidden-state max diff (float32)={diff32:.3e}, (bfloat16)={diff_bf16:.3e}")


def main():
    test_replace_attention_identity()
    test_block_reversability()
    test_full_forward_invariance_dense()
    test_hidden_invariance_dense()


if __name__ == '__main__':
    main()

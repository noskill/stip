import torch
from utils import get_model_pi
from transformers import LlamaConfig, LlamaForCausalLM
from utils import _permute_model_parameters
import copy



def test_identity_permutation():
    prompt = 'import socket\n\ndef ping_exponential_backoff(host: str):'

    # Prepare pipelines: base (pi=None) vs identity-pi
    pipeline_base  = get_model(pi=None)
    d              = pipeline_base.model.config.hidden_size
    pi             = torch.eye(d)
    pipeline_ident = get_model(pi=pi)

    # Shared generation settings
    params = dict(
        do_sample=False,
        top_k=1,
        top_p=1.0,
        num_return_sequences=1,
        eos_token_id=pipeline_base.tokenizer.eos_token_id,
        max_length=20,
    )

    out_base  = pipeline_base(prompt,  **params)[0]['generated_text']
    out_ident = pipeline_ident(prompt, **params)[0]['generated_text']
    assert out_ident == out_base, (
        f"Identity-permutation pipeline differs:\n"
        f"BASE : {out_base!r}\n"
        f"IDENT: {out_ident!r}" )
    print("✅ Identity-permutation consistency (base vs identity-pi)")

def test_block_reversibility():
    # Test that a single Transformer block permutes reversibly under π


    # Small model
    cfg = LlamaConfig(hidden_size=16, intermediate_size=64, num_hidden_layers=1,
                       num_attention_heads=4, vocab_size=100)
    llm = LlamaForCausalLM(cfg)
    llm.eval()
    block = llm.model.layers[0]
    rotary = llm.model.rotary_emb

    # Random π
    d = cfg.hidden_size
    perm = torch.randperm(d)
    pi = torch.eye(d)[perm]
    pi_t = pi.t()

    # Dummy input
    seq_len = 8
    x = torch.randn(1, seq_len, d)
    # causal mask
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
    mask = mask.triu(diagonal=1)
    pos_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rotary(x, pos_ids)

    # Original block output (unpack tuple if returned)
    y_out = block(x, attention_mask=mask, position_ids=pos_ids,
                  position_embeddings=(cos, sin))
    y = y_out[0] if isinstance(y_out, tuple) else y_out

    # Permute model weights, then permute input
    llm2 = copy.deepcopy(llm)
    llm2.eval()
    _permute_model_parameters(llm2, pi)
    block2 = llm2.model.layers[0]
    x2 = x @ pi
    # Permuted block output (unpack tuple if returned)
    y2_out = block2(x2, attention_mask=mask, position_ids=pos_ids,
                     position_embeddings=(cos, sin))
    y2 = y2_out[0] if isinstance(y2_out, tuple) else y2_out

    # Un-permute output and compare
    y2u = y2 @ pi_t
    assert torch.allclose(y, y2u, atol=1e-6), (
        f"Block irreversibility: y vs y2u differ by { (y - y2u).abs().max() }"
    )
    print("✅ Transformer block reversibility under π holds.")


# -------------------------------------------------------------------------
# 3.  Unit test  (single block reversibility under R)
# -------------------------------------------------------------------------
def test_block_reversibility_R(llm):
    """
    Test that a single Transformer block permutes reversibly under R for the given Llama model.
    Verifies RMSNorm-only change, rotation-only breakage, and full rotation+RMSNormRotated reversibility.
    """
    from utils import _rotate_model_parameters_with_R, rope_preserving_R, replace_rms_with_rotated, better_rope_preserving_R

    llm = llm.eval()
    block, rotary = llm.model.layers[0], llm.model.rotary_emb
    cfg = llm.config
    device = next(llm.parameters()).device
    dtype = llm.dtype

    # generate RoPE-preserving rotation
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    R = better_rope_preserving_R(n_heads, head_dim, dtype=torch.float32, device=device)
    R_t = R.t()

    # dummy input
    seq_len = 8
    x = torch.randn(1, seq_len, cfg.hidden_size, dtype=dtype, device=device)
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"),
                      dtype=dtype, device=device).triu(1)
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cos, sin = rotary(x, pos_ids)

    # baseline
    y_baseline = block(x, attention_mask=mask,
                       position_ids=pos_ids,
                       position_embeddings=(cos, sin))[0]

    # free GPU memory by offloading original model before cloning
    llm = llm.cpu()

    # RMSNormRotated only should change outputs
    llm_norm = copy.deepcopy(llm.cpu()).eval().to(device)
    replace_rms_with_rotated(llm_norm, R)
    y_norm = llm_norm.model.layers[0](x, attention_mask=mask,
                                      position_ids=pos_ids,
                                      position_embeddings=(cos, sin))[0]
    delta_norm = (y_baseline - y_norm).abs().max()
    assert delta_norm > 0, f"Replacing RMSNorm alone should change outputs, but max|Δ| = {delta_norm}"
    print(f"✅ RMSNormRotated alone changes block outputs (max|Δ| = {delta_norm})")
    del llm_norm
    torch.cuda.empty_cache()

    # rotation only should break reversibility
    llm_rot_only = copy.deepcopy(llm.cpu()).eval().to(device)
    _rotate_model_parameters_with_R(llm_rot_only, R)

    y_rot_only = llm_rot_only.model.layers[0]((x.to(R) @ R).to(dtype), attention_mask=mask,
                                              position_ids=pos_ids,
                                              position_embeddings=(cos, sin))[0]
    y_unrot_only = y_rot_only.to(R_t) @ R_t
    delta_rot = (y_baseline - y_unrot_only).abs().max()
    assert not torch.allclose(y_baseline.to(torch.float32), y_unrot_only.to(torch.float32), atol=1e-6), \
        f"Rotation without replacing RMSNorm should break reversibility, but max|Δ| = {delta_rot}"
    print(f"✅ Rotation without replacing RMSNorm breaks block reversibility (max|Δ| = {delta_rot})")
    del llm_rot_only
    torch.cuda.empty_cache()

    # full rotation + RMSNormRotated should restore reversibility
    llm_full = copy.deepcopy(llm.cpu()).eval().to(device)
    _rotate_model_parameters_with_R(llm_full, R)
    replace_rms_with_rotated(llm_full, R)
    y_rot = llm_full.model.layers[0]((x.to(R) @ R).to(dtype), attention_mask=mask,
                                     position_ids=pos_ids,
                                     position_embeddings=(cos, sin))[0]
    y_unrot = y_rot.to(R_t) @ R_t
    delta = (y_baseline - y_unrot).abs().max()
    assert delta < 1e-3, f"Transformer block reversibility under R failed, max|Δ| = {delta}"
    print(f"✅ Transformer block reversibility under R holds (max|Δ| = {delta})")
    del llm_full
    torch.cuda.empty_cache()
    

def test_model_reversibility():
    # Test that the full LlamaModel permuted under π can be un-permuted by πᵀ
    from transformers import LlamaConfig, LlamaForCausalLM
    from utils import _permute_model_parameters
    import copy

    # Small model config
    cfg = LlamaConfig(hidden_size=16, intermediate_size=64, num_hidden_layers=1,
                       num_attention_heads=4, vocab_size=100)
    llm = LlamaForCausalLM(cfg)
    llm.eval()

    # Random permutation π and its transpose
    d = cfg.hidden_size
    perm = torch.randperm(d)
    pi = torch.eye(d)[perm]
    pi_t = pi.t()

    # Dummy token input and mask
    seq_len = 8
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len))
    attention_mask = torch.ones_like(input_ids)

    # Original model output hidden states
    out = llm.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    y = out.last_hidden_state

    # Permute model weights, then run permuted model
    llm2 = copy.deepcopy(llm)
    llm2.eval()
    _permute_model_parameters(llm2, pi)
    out2 = llm2.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    y2 = out2.last_hidden_state

    # Un-permute output and compare to original
    y2u = y2 @ pi_t
    assert torch.allclose(y, y2u, atol=1e-6), (
        f"Model irreversibility: y vs y2u differ by { (y - y2u).abs().max() }"
    )
    print("✅ LlamaModel reversibility under π holds.")

def test_block_reversibility_R_and_Delta():
    cfg = LlamaConfig(hidden_size=16,
                      intermediate_size=64,
                      num_hidden_layers=1,
                      num_attention_heads=4,
                      vocab_size=100)
    llm  = LlamaForCausalLM(cfg).eval()
    block, rotary = llm.model.layers[0], llm.model.rotary_emb

    # ---------- inject strong dense ΔW ----------------------------------
    _add_dense_noise(llm, rho=0.30)

    # ---------- baseline output -----------------------------------------
    seq = 8
    x   = torch.randn(1, seq, cfg.hidden_size)
    mask = torch.full((1,1,seq,seq), float("-inf")).triu(1)
    pos  = torch.arange(seq).unsqueeze(0)
    cos, sin = rotary(x, pos)
    y_base = block(x, attention_mask=mask,
                   position_ids=pos,
                   position_embeddings=(cos, sin))[0]

    # ---------- build rotation & rotated model --------------------------
    n_h     = cfg.num_attention_heads
    h_dim   = cfg.hidden_size // n_h
    R       = rope_preserving_R(n_h, h_dim, dtype=x.dtype, device=x.device)
    llm2    = copy.deepcopy(llm).eval()          # already contains ΔW
    _rotate_model_parameters_with_R(llm2, R)

    block2  = llm2.model.layers[0]
    x_rot   = x @ R
    y_rot   = block2(x_rot, attention_mask=mask,
                     position_ids=pos,
                     position_embeddings=(cos, sin))[0]
    y_unrot = y_rot @ R.T

    assert torch.allclose(y_base, y_unrot, atol=3e-4), (
        f"Mismatch: max|Δ| = {(y_base - y_unrot).abs().max()}"
    )
    print("✅  Reversibility with rotation R + dense ΔW holds.")


def test_block_reversibility_R_real():
    """
    Run block reversibility under R on a real pretrained Llama model.
    """
    import torch
    from utils import model as MODEL_NAME
    from transformers import LlamaForCausalLM

    # load model to GPU if available (deep-copies happen on CPU inside the block test)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    llm = LlamaForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()

    test_block_reversibility_R(llm)


if __name__ == '__main__':
    test_block_reversibility_R_real()

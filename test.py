import torch
from utils import get_model




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
    from transformers import LlamaConfig, LlamaForCausalLM
    from utils import _permute_model_parameters
    import copy

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


if __name__ == '__main__':
    test_block_reversibility()
    test_model_reversibility()
    test_identity_permutation()

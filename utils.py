import torch
from transformers import AutoTokenizer
import transformers

from transformers.models.llama.modeling_llama import LlamaForCausalLM


model = "codellama/CodeLlama-7b-hf"

def _permute_model_parameters(model, pi):
    """In-place permute parameters of a Llama-based model according to a permutation matrix pi,
    following the transformations in readme.txt.

    The single pi is applied across all layers. Embeddings (and the tied lm_head), attention,
    feedforward (SwiGLU), and LayerNorm weights/biases are permuted."""
    pi = torch.as_tensor(pi)
    # STIP uses the following convention: PyTorch Linear layers store weight as Wᵀ (out_features × in_features).
    # So for a STIP transform W' = πᵀ·W, we implement weight @ π;
    # and for a transform W' = W·π, we implement πᵀ @ weight.
    # Only Llama-based models with model.model.layers support permutation; otherwise no-op
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        return
    if pi.ndim != 2 or pi.size(0) != pi.size(1):
        raise ValueError(f"Permutation matrix pi must be square, got shape {tuple(pi.shape)}")
    d = model.model.config.hidden_size
    if pi.size(0) != d:
        raise ValueError(f"pi size {tuple(pi.shape)} does not match model hidden size {d}")
    # Prepare permutation matrix: ensure dtype matches model, device will be set per-layer
    embed = model.model.embed_tokens
    pi = pi.to(dtype=embed.weight.dtype)

    for name, module in model.model.named_modules():
        if hasattr(module, 'bias') and module.bias is not None:
            raise RuntimeError(f"Unexpected bias in module '{name}'")
    with torch.no_grad():
        # Permute input embedding and LM head weight (W_e, W_lm): shape [vocab_size, d]
        #   W' = W · π  (permute embedding/​head dimensions, matching readme's x·π)
        pi_embed = pi.to(device=embed.weight.device)
        embed.weight.copy_(embed.weight @ pi_embed)
        head = model.lm_head
        pi_head = pi.to(device=head.weight.device)
        # head.weight = Wᵀ
        # weight = head.weight @ pi_head  equivalent to W′ᵀ = Wᵀ · π
        # logits = h · (Wᵀ)ᵀ for unmodified transformer
        # logits = h π · (W′ᵀ)ᵀ = h π · (Wᵀ · π)ᵀ = h π ·πᵀ W = h W for permuted transformer
        head.weight.copy_(head.weight @ pi_head)

        # Permute each Transformer block on its own device
        for layer in model.model.layers:
            dev = layer.self_attn.q_proj.weight.device
            pi_layer = pi.to(device=dev)
            pi_t = pi_layer.t()

            # Attention projections:
            # - q_proj/k_proj/v_proj.weight stores Wqᵀ/Wkᵀ/Wvᵀ, so to implement W' = πᵀ·W (readme), we do weight @ π
            # - o_proj.weight stores Woᵀ, so to implement W' = W·π (readme), we do πᵀ @ weight
            layer.self_attn.q_proj.weight.copy_(layer.self_attn.q_proj.weight @ pi_layer)
            layer.self_attn.k_proj.weight.copy_(layer.self_attn.k_proj.weight @ pi_layer)
            layer.self_attn.v_proj.weight.copy_(layer.self_attn.v_proj.weight @ pi_layer)
            layer.self_attn.o_proj.weight.copy_(pi_t @ layer.self_attn.o_proj.weight)

            # Feedforward (SwiGLU): gate_proj (W1) and up_proj (W3) parameters have shape (m×d), so
            #   W1' = πᵀ·W1 and W3' = πᵀ·W3      → weight @ π
            # down_proj parameter stores W2ᵀ (shape d×m) for W2∈ℝ^{m×d}, so
            #   W2' = W2·π      ⟹ (W2·π)ᵀ = πᵀ·W2ᵀ → πᵀ @ weight
            # After permuting down_proj (W2) and up_proj (W3) weights, users will need to perform permuted-features -> logits transformation
            # (see readme.txt line 65)
            layer.mlp.gate_proj.weight.copy_(layer.mlp.gate_proj.weight @ pi_layer)
            layer.mlp.up_proj.weight.copy_(layer.mlp.up_proj.weight @ pi_layer)
            layer.mlp.down_proj.weight.copy_(pi_t @ layer.mlp.down_proj.weight)

            # LayerNorm parameters: γ',β' = γ·π, β·π
            ln1 = layer.input_layernorm
            ln2 = layer.post_attention_layernorm
            ln1.weight.copy_(ln1.weight @ pi_layer)
            if hasattr(ln1, 'bias'):
                ln1.bias.copy_(ln1.bias @ pi_layer)
            ln2.weight.copy_(ln2.weight @ pi_layer)
            if hasattr(ln2, 'bias'):
                ln2.bias.copy_(ln2.bias @ pi_layer)

        # Final model-level normalization (model.model.norm)
        final_ln = model.model.norm
        # align pi to the norm's device
        pi_fn = pi.to(device=final_ln.weight.device)
        final_ln.weight.copy_(final_ln.weight @ pi_fn)
        if hasattr(final_ln, 'bias'):
            final_ln.bias.copy_(final_ln.bias @ pi_fn)

def get_model(pi=None):
    """
    Build a text-generation pipeline for the specified model.

    If a permutation matrix `pi` (shape [d, d]) is provided, the model's parameters
    are permuted in memory according to the procedure in readme.txt before inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    if pi is not None:
        model_obj = LlamaForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _permute_model_parameters(model_obj, pi)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_obj,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    return pipeline





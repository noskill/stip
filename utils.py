import torch
from transformers import AutoTokenizer
import transformers
import math
from transformers.models.llama.modeling_llama import LlamaForCausalLM


model = "codellama/CodeLlama-7b-hf"


# -------------------------------------------------------------------------
# 2.  Add dense high-rank ΔW  (≈ 30 % spectral norm of W)
# -------------------------------------------------------------------------
def _add_dense_noise(model, rho: float = 0.30):
    """In-place:   W ← W + ΔW   with  ‖ΔW‖₂ ≈ rho · ‖W‖₂."""
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 2:                                 # linear weight
                g = torch.randn_like(p)
                g *= (rho * p.norm(2) / g.norm(2) + 1e-12)
                p.add_(g)


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
            


# -------------------------------------------------------------------------
# 1.  Build a block-orthogonal rotation that keeps every RoPE pair intact
# -------------------------------------------------------------------------
def rope_preserving_R(n_heads: int, head_dim: int = 128,
                      dtype=torch.float32, device=None) -> torch.Tensor:
    """
    Return a (d × d) orthogonal matrix R that is block-diagonal per head
    and, inside every 2-element RoPE pair, a planar rotation.

        d        = n_heads * head_dim
        head_dim = 128   (64 RoPE pairs) in Llama-2

    Such an R commutes with apply_rotary_pos_emb and allows us to keep
    RMS-Norm diagonal (because it only re-orders / rotates within pairs).
    """
    if head_dim % 2:
        raise ValueError("head_dim must be even")

    pairs = head_dim // 2
    d     = n_heads * head_dim
    R     = torch.zeros((d, d), dtype=dtype, device=device)

    for h in range(n_heads):
        base = h * head_dim
        for j in range(pairs):
            theta  = torch.rand(1, device=device) * 2 * math.pi
            c, s   = torch.cos(theta), torch.sin(theta)
            rot2   = torch.tensor([[c, -s],
                                   [s,  c]], dtype=dtype, device=device)
            row    = base + 2 * j
            R[row:row + 2, row:row + 2] = rot2
    return R

                
from torch import nn
class RMSNormRotated(nn.Module):
    """
    RMSNorm that assumes its inputs are *already* right-multiplied by a
    fixed orthogonal matrix R. It uses mixed-precision to ensure stability.
    """
    def __init__(self, d, eps=1e-5, R=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))  # γ
        assert R is not None and R.shape == (d, d)
        self.register_buffer("R", R, persistent=False)  # no grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # Cast to float32 for stable calculations
        x_fp32 = x.to(torch.float32)

        # (1) Calculate variance and normalize in float32
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        xhat_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)

        # (2) Un-rotate in float32
        # Ensure R is also in float32 for the matmul
        R_fp32 = self.R.to(torch.float32).to(x.device)
        x_unrot_fp32 = torch.matmul(xhat_fp32, R_fp32.t())

        # (3) Apply scaling gamma in float32
        x_scaled_fp32 = x_unrot_fp32 * self.weight.to(torch.float32).to(x_unrot_fp32.device)

        # (4) Re-rotate back to R-space in float32
        y_fp32 = torch.matmul(x_scaled_fp32, R_fp32)

        # (5) Cast back to the original dtype before returning
        return y_fp32.to(input_dtype)

from transformers.models.llama.modeling_llama import LlamaRMSNorm

def replace_rms_with_rotated(model_obj, R):
    d = model_obj.config.hidden_size
    R = R.to(model_obj.dtype).to(model_obj.device)

    for layer in model_obj.model.layers:
        for attr in ("input_layernorm", "post_attention_layernorm"):
            old_ln = getattr(layer, attr)
            new_ln = RMSNormRotated(d, old_ln.variance_epsilon, R)
            new_ln.weight.data.copy_(old_ln.weight.data)   # keep γ
            setattr(layer, attr, new_ln)

    # final norm at the top of the stack
    old_top = model_obj.model.norm
    model_obj.model.norm = RMSNormRotated(d, old_top.variance_epsilon, R)
    model_obj.model.norm.weight.data.copy_(old_top.weight.data)


# -------------------------------------------------------------------------
# 2.  In-place rotation of *all* parameters that live in ℝd
# -------------------------------------------------------------------------
def _rotate_model_parameters_with_R(model, R: torch.Tensor):
    """
    Fold an orthogonal rotation R into every Llama weight.
    – square  (d×d)   :  W′ = Rᵀ W R
    – tall    (d×m)   :  W′ = Rᵀ W
    – wide    (m×d)   :  W′ = W R
    Embedding and lm_head are post-multiplied by R.
    RMS-Norm scales γ are *not* modified – the new RMSNormRotated takes
    care of the rotation at run time.
    """
    d = model.config.hidden_size
    assert R.shape == (d, d)
    R = R.to(torch.float32)                 # master

    emb  = model.model.embed_tokens.weight      # (V, d)
    head = model.lm_head.weight                # (V, d)  tied

    with torch.no_grad():
        # embeddings + lm_head
        emb .copy_((emb .float() @ R.to(emb .device)).to(emb .dtype))
        head.copy_((head.float() @ R.to(head.device)).to(head.dtype))

        # transformer blocks
        Rt_cpu = R.T
        for layer in model.model.layers:
            # self-attention (square)
            for proj in ("q_proj", "k_proj", "v_proj"):
                w = getattr(layer.self_attn, proj).weight   # (d,d)
                Rt = Rt_cpu.to(w.device)
                w.copy_((Rt @ w.float() @ R.to(w.device)).to(w.dtype))

            w = layer.self_attn.o_proj.weight               # (d,d)
            Rt = Rt_cpu.to(w.device)
            w.copy_((Rt @ w.float() @ R.to(w.device)).to(w.dtype))

            # MLP (rectangular)
            up   = layer.mlp.up_proj.weight     # (11008 , 4096)
            gate = layer.mlp.gate_proj.weight   # (11008 , 4096)
            down = layer.mlp.down_proj.weight   # ( 4096 , 11008)

            up  .copy_((up  .float() @ R.to(up  .device)).to(up  .dtype))
            gate.copy_((gate.float() @ R.to(gate.device)).to(gate.dtype))

            Rt = Rt_cpu.to(down.device)
            down.copy_((Rt @ down.float()).to(down.dtype))




def get_model_R():

    tokenizer = AutoTokenizer.from_pretrained(model)
    # load model on a single device in half precision
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_obj = LlamaForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
    ).to(device)

    # rho = 0.1
    # print(f'adding noise {rho}')
    # _add_dense_noise(model_obj, rho)
    cfg = model_obj.config
    R    = rope_preserving_R(cfg.num_attention_heads,
                            cfg.hidden_size // cfg.num_attention_heads,
                            dtype=torch.float32,
                            device=next(model_obj.parameters()).device)
    _rotate_model_parameters_with_R(model_obj, R)
    replace_rms_with_rotated(model_obj, R)
    
    # build a text-generation pipeline on the same device
    device_id = device.index if device.type == "cuda" else -1
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_obj,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device=device_id,
    )
    return pipeline



def get_model_pi(pi):

    tokenizer = AutoTokenizer.from_pretrained(model)
    # load model on a single device in half precision
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_obj = LlamaForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
    ).to(device)

    cfg = model_obj.config

    _permute_model_parameters(model_obj, pi)
    
    # build a text-generation pipeline on the same device
    device_id = device.index if device.type == "cuda" else -1
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_obj,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device=device_id,
    )
    return pipeline


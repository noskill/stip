import torch
from typing import Optional
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


def better_add_dense_noise(model, noise_ratio: float = 0.05, seed: int | None = None):
    """
    Добавляет к каждой двумерной матрице модели плотный гауссов шум.

    noise_ratio:  относительная величина шума (0.05 ≈ 5 % Frobenius-нормы).
    seed:         при необходимости фиксирует ГСЧ для воспроизводимости.
    """
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        for W in model.parameters():
            if W.ndim != 2:          # пропускаем векторы RMSNorm, bias и т.п.
                continue

            frob_norm = W.norm(p='fro')            # ‖W‖_F
            n_elem    = W.numel()                  # m · n
            sigma     = noise_ratio * frob_norm / n_elem**0.5
            noise     = torch.randn_like(W) * sigma
            W.add_(noise)  


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


def better_rope_preserving_R(n_heads: int,
                      head_dim: int = 128,
                      max_angle: float = 0.20,        # radians ≈ 11°
                      dtype=torch.float32,
                      device=None) -> torch.Tensor:
    """
    Same layout as rope_preserving_R but θ ∼ U(-max_angle, max_angle).
    That keeps each feature within ≈ cos(11°)=0.98 correlation
    with the original axis → quality drop is usually <0.5 pp.
    """
    import math, torch
    if head_dim % 2:
        raise ValueError("head_dim must be even")
    d      = n_heads * head_dim
    pairs  = head_dim // 2
    R      = torch.zeros((d, d), dtype=torch.float32, device=device)
    for h in range(n_heads):
        base = h * head_dim
        for j in range(pairs):
            theta = (torch.rand(1, device=device) * 2 - 1) * max_angle
            c, s  = torch.cos(theta), torch.sin(theta)
            rot   = torch.stack((torch.cat([c, -s]),
                                 torch.cat([s,  c])))
            row   = base + 2 * j
            R[row:row+2, row:row+2] = rot
    return R.to(dtype)


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
        R_fp32 = self.R.to(x.device)
        x_unrot_fp32 = torch.matmul(xhat_fp32, R_fp32.t())

        # (3) Apply scaling gamma in float32
        x_scaled_fp32 = x_unrot_fp32 * self.weight.to(torch.float32).to(x_unrot_fp32.device)

        # (4) Re-rotate back to R-space in float32
        y_fp32 = torch.matmul(x_scaled_fp32, R_fp32)

        # (5) Cast back to the original dtype before returning
        return y_fp32.to(input_dtype)




def rope_incompatible_head_R(
    n_heads: int,
    head_dim: int = 128,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device | str] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Return a *head-local* but *RoPE-incompatible* orthogonal matrix R ∈ SO(d).

    •  “head-local”:  R is block-diagonal with one head_dim×head_dim block
       per attention head.  Nothing is mixed across heads, so you can still
       `view(b,s,n_heads,head_dim)` without breaking shapes.

    •  “RoPE-incompatible”:  inside every head we use a dense orthogonal
       matrix (QR of a Gaussian) instead of independent 2×2 rotations, so
       the original (x0,x1), (x2,x3) pairs are not preserved.

    Parameters
    ----------
    n_heads : number of attention heads.
    head_dim: hidden width per head (must be even for Llama-style models).
    dtype   : result dtype (default fp32, change to fp16/fp64 if needed).
    device  : torch.device or "cuda", "cpu", etc.
    seed    : optional RNG seed for deterministic output.

    Returns
    -------
    R : torch.Tensor of shape (d, d) where d = n_heads * head_dim.
        R is orthogonal (R @ R.T == I) and det(R) = +1.
    """
    if head_dim % 2:
        raise ValueError("head_dim must be even (got %d)" % head_dim)
    if seed is not None:
        torch.manual_seed(seed)

    d = n_heads * head_dim
    R = torch.zeros((d, d), dtype=dtype, device=device)

    for h in range(n_heads):
        # 1.  Random Gaussian block, cast to float32 for numerical stability
        g = torch.randn(head_dim, head_dim, device=device, dtype=torch.float32)

        # 2.  QR → orthogonal matrix q  (g = q r)
        q, _ = torch.linalg.qr(g)

        # 3.  Make determinant +1  (q in SO(d), not O(d))
        if torch.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]

        # 4.  Insert the block on the diagonal
        base = h * head_dim
        R[base : base + head_dim, base : base + head_dim] = q.to(dtype)

    return R


from transformers.models.llama.modeling_llama import LlamaRMSNorm

def replace_rms_with_rotated(model_obj, R):
    d = model_obj.config.hidden_size
    R = R.to(torch.float32).to(model_obj.device)

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


from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig, Cache, Unpack, FlashAttentionKwargs, apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS


class MyAttn(LlamaAttention):
    def __init__(self, config, layer_idx, R: torch.Tensor):
        super().__init__(config, layer_idx)
        self.register_buffer("R", R)               # [d, d]
        self.num_attention_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim

    # ⬇ helper – быстрее, чем torch.matmul 2-D-к-3-D
    def _mm(self, x, M):
        # Batched matrix-multiply and cast back to original dtype.
        # Preserve higher precision if x is float64 or lower if float32/16.
        return (x.to(M.dtype) @ M).to(x.dtype)

    def forward(
        self,
        hidden_states,                # (B,S,d)           уже содержит R
        position_embeddings,          # (cos, sin)
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        #return super().forward(hidden_states,  position_embeddings, attention_mask=attention_mask,
        # past_key_value=past_key_value,
        # cache_position=cache_position,
        # **kwargs)
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        B, S, _ = hidden_states.shape
        d      = self.config.hidden_size

        # 1) linear projection still in  R-space
        q = self.q_proj(hidden_states)             # (B,S,d)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)             # v оставляем как есть

        # 2) revert R               (B,S,d)
        q = self._mm(q, self.R.T)       #           q′ = (h R) @ (Rᵀ Wq R) …  ─────▶  q = q′ Rᵀ
        k = self._mm(k, self.R.T)         #           k = k′ Rᵀ
        
        qshape = q.shape
        kshape = k.shape

        # 3) attention heads
        q = q.view(hidden_shape).transpose(1, 2)  # B, S, num_attention_heads, head_dim
        k = k.view(hidden_shape).transpose(1, 2)
        v = v.view(hidden_shape).transpose(1, 2)
 
        # 4) RoPE on un-rotated q/k, then update key/value cache if used
        cos, sin = position_embeddings
        # 4) RoPE on un-rotated q/k
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 5) merge heads and re-apply rotation R to q/k; v remains in R-space
        q = q.transpose(1, 2).contiguous().view(B, S, d)
        _, _, k_seq, _ = k.size()
        k = k.transpose(1, 2).contiguous().view(B, k_seq, d)
        # rotate again
        q = self._mm(q, self.R)                  # q` ← q · R
        k = self._mm(k, self.R)                  # k`  ← k · R

        # 6) update KV cache (key may grow beyond current S)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        # 7) proceed with usual attention over heads
        q = q.view(hidden_shape).transpose(1, 2)
        _, k_seq, _ = k.size()
        k = k.view((B, k_seq, -1, self.head_dim)).transpose(1, 2)

        attention_fn = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_fn(
            self,
            q, k, v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(B, S, d).contiguous()
        attn_output = self.o_proj(attn_output)     # o_proj уже повернут оф-лайн
        return attn_output, attn_weights
    

def replace_attention_RoPE_dense(model_obj, R):
    for layer in model_obj.model.layers:

        old_attn = getattr(layer, 'self_attn', None)
        if old_attn is None:
            return
        with torch.no_grad():    
            new_attn = MyAttn(
                old_attn.config,
                getattr(old_attn, "layer_idx", 0),
                R
            ).to(old_attn.q_proj.weight.device, old_attn.q_proj.weight.dtype)
            

            # 2. копируем веса; bias может отсутствовать
            new_attn.q_proj.weight.copy_(old_attn.q_proj.weight)
            new_attn.k_proj.weight.copy_(old_attn.k_proj.weight)
            new_attn.v_proj.weight.copy_(old_attn.v_proj.weight)
            new_attn.o_proj.weight.copy_(old_attn.o_proj.weight)

            if old_attn.q_proj.bias is not None:
                new_attn.q_proj.bias.copy_(old_attn.q_proj.bias)
                new_attn.k_proj.bias.copy_(old_attn.k_proj.bias)
                new_attn.v_proj.bias.copy_(old_attn.v_proj.bias)
                new_attn.o_proj.bias.copy_(old_attn.o_proj.bias)

            # 3. если в MyAttn есть rotary_emb или другие буферы — скопируем
            for name, buf in old_attn.named_buffers():
                if name in dict(new_attn.named_buffers()):
                    getattr(new_attn, name).copy_(buf)

            # 4. подменяем в слое
            layer.self_attn = new_attn
            new_attn.R = R
    return model_obj


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
    # Ensure R master is float32 on CPU for consistent per-weight dispatch
    R_master = R.to(torch.float32).cpu()

    emb  = model.model.embed_tokens.weight      # (V, d)
    head = model.lm_head.weight                # (V, d) 

    with torch.no_grad():
        # ensure no biases in linear projections for rotation invariance
        for layer in model.model.layers:
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                bias = getattr(layer.self_attn, proj).bias
                assert bias is None, f"Expected no bias in self_attn.{proj}, but got {bias}"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                bias = getattr(layer.mlp, proj).bias
                assert bias is None, f"Expected no bias in mlp.{proj}, but got {bias}"
        untied = False
        if head.data_ptr() != emb.data_ptr():
            untied = True
        # embeddings + lm_head
        emb.copy_((emb.float() @ R_master.to(emb.device)).to(emb.dtype))
        if untied:
            head.copy_((head.float() @ R_master.to(head.device)).to(head.dtype))

        # transformer blocks
        Rt_cpu = R_master.T
        for layer in model.model.layers:
            # self-attention (square)
            for proj in ("q_proj", "k_proj", "v_proj"):
                w = getattr(layer.self_attn, proj).weight   # (d,d)
                Rt = Rt_cpu.to(w.device)
                w.copy_((Rt @ w.float() @ R_master.to(w.device)).to(w.dtype))

            w = layer.self_attn.o_proj.weight               # (d,d)
            w.copy_((Rt @ w.float() @ R_master.to(w.device)).to(w.dtype))

            # MLP (rectangular)
            up   = layer.mlp.up_proj.weight     # (11008 , 4096)
            gate = layer.mlp.gate_proj.weight   # (11008 , 4096)
            down = layer.mlp.down_proj.weight   # ( 4096 , 11008)

            # (A·B·C)ᵀ = Bᵀ·Aᵀ·Cᵀ
            # W₁` = (Rᵀ W₁)
            # W₁ᵀ = (Rᵀ W₁)ᵀ =W₁ᵀRᵀᵀ = W₁ᵀR  
            up.copy_((up.float() @ R_master.to(up.device)).to(up.dtype))
            gate.copy_((gate.float() @ R_master.to(gate.device)).to(gate.dtype))

            Rt = Rt_cpu.to(down.device)
            # W₂′ = W₂ R 
            down.copy_((Rt @ down.float()).to(down.dtype))


def get_model_R_dense():
    tokenizer = AutoTokenizer.from_pretrained(model)
    # load model on a single device in half precision
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_obj = LlamaForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
    ).to(device)

    
    cfg = model_obj.config
    R    = dense_orthogonal_R(cfg.hidden_size,
                            dtype=torch.float32,
                            device=next(model_obj.parameters()).device)

#    R = torch.eye(cfg.hidden_size)
#    R    = better_rope_preserving_R(cfg.num_attention_heads,
#                            cfg.hidden_size // cfg.num_attention_heads,
#                            max_angle=0.8,
#                            dtype=torch.float32,
#                            device=next(model_obj.parameters()).device)
#
    _rotate_model_parameters_with_R(model_obj, R)
    replace_rms_with_rotated(model_obj, R)
    replace_attention_RoPE_dense(model_obj, R)
    model_obj.to(device)
    
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


def get_model_R():
    tokenizer = AutoTokenizer.from_pretrained(model)
    # load model on a single device in half precision
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_obj = LlamaForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
    ).to(device)

    rho = 0.1
    # print(f'adding noise {rho}')
    better_add_dense_noise(model_obj)
    
    cfg = model_obj.config
    R    = better_rope_preserving_R(cfg.num_attention_heads,
                            cfg.hidden_size // cfg.num_attention_heads,
                            max_angle=0.8,
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



def dense_orthogonal_R(
    d: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device | str] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generates a random orthogonal matrix R size D × D.
    
    • The matrix is dense: each dimension is mixed with everyone else.
    • No structure of “pairing coordinates” (as in Rope) is preserved.
    • DET (R) = +1, i.e. Belongs SO (D).
    
    Parameters
    ----------
    D: Dimension of hidden space (for example, 4096 for LLAMA-2-13B).
    Dtype: Torch.float32 | Torch.float16 | ...
    Device: "Cuda", "Cuda: 1", "CPU", etc.
    SEED: if necessary, fixes the generator of random numbers.
    
    Returns
    ----------
    R: torch.tensor shape (d, d), orthogonal (r @ r.t == i).
        """
    if seed is not None:
        torch.manual_seed(seed)

    # 1. random Gaussian matrix
    g = torch.randn(d, d, dtype=dtype, device=device)

    # 2. QR-decomposition  (g = Q R)
    #    Q – orthonormal, R – upper triangular
    q, r = torch.linalg.qr(g, mode='reduced')

    # 3. make det(Q) = +1
    #    (multiply each column Q by diagonal element of R)
    diag = torch.diagonal(r)
    phase = diag.sign()
    q *= phase

    return q

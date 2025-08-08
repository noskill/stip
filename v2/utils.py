"""Utility helpers used across the project.

This file introduces a custom attention layer – `MyAttn` – that follows the
idea described in *readme.txt*:

    1.  Inside a Trusted Execution Environment (TEE) we would multiply the
        projections `Q`, `K` and `V` by a secret orthogonal matrix **R**.
    2.  The expensive matrix multiplication `Q'K'^T` and the soft-max would be
        executed on an (un-trusted) accelerator.  Because **R** is orthogonal
        the attention scores are identical to a vanilla run.
    3.  Finally we would multiply the attended values by **Rᵀ** so that the
        remainder of the network stays unchanged.

For the purpose of this repository we do **all** computations on a single GPU
or on CPU.  Nevertheless `MyAttn` still performs the two extra rotations so
that we can later separate the trusted / un-trusted parts by simply moving the
relevant blocks of code.

Two helper functions are exposed to generate valid orthogonal matrices **R**:

* ``dense_random_orthogonal`` – draws a random orthogonal matrix using QR
  decomposition.
* ``random_permutation_matrix`` – returns a random permutation matrix (a special
  orthogonal matrix whose rows/columns form a permutation of the identity).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import transformers
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    Cache,
    TransformersKwargs,
    LlamaAttention,
    Unpack,
    apply_rotary_pos_emb,
    eager_attention_forward,
    LlamaForCausalLM
)

__all__ = [
    "MyAttn",
    "dense_random_orthogonal",
    "random_permutation_matrix",
    "MyAttnBlock",
    "dense_block_orthogonal",
    "permutation_block_matrix",
]

# ---------------------------------------------------------------------------
# Helpers to create orthogonal matrices R
# ---------------------------------------------------------------------------


def dense_random_orthogonal(dim: int, *, device: torch.device | None = None, dtype=torch.float32) -> torch.Tensor:
    """Return a random orthogonal matrix *R* of shape ``[dim, dim]``.

    The matrix is obtained from the *Q* factor of a QR-decomposition which
    guarantees orthogonality: ``R @ R.T = I``.
    """

    # QR-decomposition is not available for float16/bfloat16 on CUDA.  We work
    # around this limitation by computing the orthogonal matrix in float32 and
    # casting afterwards if necessary.

    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

    a = torch.randn((dim, dim), device=device, dtype=compute_dtype)
    q, r = torch.linalg.qr(a, mode="reduced")

    d = torch.diag(r).sign()
    q *= d

    if q.dtype != dtype:
        q = q.to(dtype)

    return q


def random_permutation_matrix(dim: int, *, device: torch.device | None = None, dtype=torch.float32) -> torch.Tensor:
    """Return a random *dim × dim* permutation matrix.

    A permutation matrix is orthogonal because each row/column contains a
    single *1* and otherwise *0*s.
    """

    perm = torch.randperm(dim, device=device)
    mat = torch.zeros((dim, dim), device=device, dtype=dtype)
    mat[torch.arange(dim, device=device), perm] = 1.0
    return mat


# ---------------------------------------------------------------------------
#               Block–diagonal variants (unique R per attention head)
# ---------------------------------------------------------------------------


def _block_diag(mats: list[torch.Tensor]) -> torch.Tensor:
    """Create a block-diagonal matrix from a list of square matrices."""

    return torch.block_diag(*mats)


def dense_block_orthogonal(
    num_heads: int,
    head_dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return **block-diagonal** random orthogonal matrix of size ``hidden_size``.

    A fresh random orthogonal block is drawn for every attention head.
    """

    blocks = [dense_random_orthogonal(head_dim, device=device, dtype=dtype) for _ in range(num_heads)]
    return _block_diag(blocks).to(dtype)


#######################################################################
#                       INTERNAL ATTENTION UTILS                      #
#######################################################################


def _attention_in_chunks(
    module: LlamaAttention,
    attention_interface: Callable,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    chunk_size: int,
    kwargs: dict,
):
    """Apply *attention_interface* sequentially over head-chunks and concat.

    This emulates sending disjoint sets of heads to distinct accelerators.
    All tensors must have shape ``[B, H, S, D]`` before calling this helper.
    Returns concatenated attn_output (B, S, H, D) and attn_weights
    (Optionally concatenated along head dimension).
    """

    outputs: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []

    num_heads = query_states.shape[1]

    for start in range(0, num_heads, chunk_size):
        end = min(start + chunk_size, num_heads)

        qs = query_states[:, start:end]
        ks = key_states[:, start:end]
        vs = value_states[:, start:end]

        out, w = attention_interface(
            module,
            qs,
            ks,
            vs,
            attention_mask,
            dropout=0.0 if not module.training else module.attention_dropout,
            scaling=module.scaling,
            **kwargs,
        )  # out: (B, S, end-start, D)

        outputs.append(out)
        if w is not None:
            weights.append(w)

    attn_output = torch.cat(outputs, dim=2)  # concatenate along head dim inside (B, S, heads, D)

    attn_weights_cat = None
    if weights:
        attn_weights_cat = torch.cat(weights, dim=1)  # (B, heads, S_q, S_k)

    return attn_output, attn_weights_cat


def permutation_block_matrix(
    num_heads: int,
    head_dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return **block-diagonal** permutation matrix (orthogonal)."""

    blocks = [random_permutation_matrix(head_dim, device=device, dtype=dtype) for _ in range(num_heads)]
    return _block_diag(blocks).to(dtype)


# ---------------------------------------------------------------------------
#                   Custom attention layer with secret rotation
# ---------------------------------------------------------------------------


# NOTE: `MyAttn` behaves identically to `LlamaAttention` **numerically**.  The
# only difference is that (1) the *query*, *key* and *value* tensors are first
# multiplied by a secret orthogonal matrix **R** and (2) the attended values
# are multiplied by **Rᵀ** before the final output projection.  Thanks to the
# orthogonality of **R**, the attention scores are unchanged.  Consequently the
# complete forward pass is functionally equivalent to the vanilla
# implementation – this will be verified in the accompanying unit-tests.


class MyAttn(LlamaAttention):
    """Attention layer that hides intermediate tensors behind a random rotation.

    Parameters
    ----------
    config:
        ``transformers`` *LlamaConfig* instance.
    layer_idx:
        Index of the layer inside the model (only needed for the built-in
        caching mechanism).
    R:
        Orthogonal matrix of shape ``[head_dim, head_dim]`` that will be used
        to rotate *Q*, *K* and *V*.
    """

    def __init__(self, config, layer_idx: int, R: torch.Tensor):
        super().__init__(config, layer_idx)

        if R.shape != (config.head_dim, config.head_dim):
            raise ValueError(
                f"R must be of shape [head_dim, head_dim] == "
                f"[{config.head_dim}, {config.head_dim}], but got {list(R.shape)}"
            )

        # Register as a (non-persistent) buffer so that *R* is moved together
        # with the module between devices but is **not** saved when calling
        # ``state_dict``.
        self.register_buffer("R", R, persistent=False)

        # Keep local copies for quick access (avoids repeated attribute lookup)
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        # ------------------------------------------------------------------
        # 1.  Standard projections – identical to `LlamaAttention`
        # ------------------------------------------------------------------

        input_shape = hidden_states.shape[:-1]  # (batch, seq_len)
        hidden_shape = (*input_shape, -1, self.head_dim)  # -> (B, S, H, D)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # ------------------------------------------------------------------
        # 2.  Rotary position embedding (RoPE)
        # ------------------------------------------------------------------

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ------------------------------------------------------------------
        # 3.  Rotate Q, K, V by the secret orthogonal matrix  ----------------
        # ------------------------------------------------------------------

        # Shapes:
        #   query_states : (B, H, S, D)
        #   self.R       : (D, D)
        # We rely on ``torch.matmul`` to perform the rotation along the last
        # dimension while broadcasting across the leading ones.
        def _rotate_tensor(t: torch.Tensor, transpose: bool = False) -> torch.Tensor:
            original_dtype = t.dtype
            R_mat = self.R.T if transpose else self.R
            tmp = torch.matmul(t.to(R_mat.dtype), R_mat)
            return tmp.to(original_dtype)

        query_states = _rotate_tensor(query_states)
        key_states = _rotate_tensor(key_states)
        value_states = _rotate_tensor(value_states)

        # ------------------------------------------------------------------
        # 4.  Past-key-value cache (if any) – nothing changes except that the
        #     cached tensors are now rotated as well.
        # ------------------------------------------------------------------

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # ------------------------------------------------------------------
        # 5.  Attention computation (possibly Flash-Attention)  -------------
        # ------------------------------------------------------------------

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # ------------------------------------------------------------------
        #  Split heads into chunks to emulate distributed untrusted devices
        # ------------------------------------------------------------------

        chunk_size = 4  # default #heads per "device" (can be tuned)

        attn_output, attn_weights = _attention_in_chunks(
            self,
            attention_interface,
            query_states,
            key_states,
            value_states,
            attention_mask,
            chunk_size=chunk_size,
            kwargs=kwargs,
        )  # -> (B, S, H, D)

        # ------------------------------------------------------------------
        # 6.  Undo the rotation so that the remainder of the model receives
        #     the *standard* representation.
        # ------------------------------------------------------------------

        attn_output = _rotate_tensor(attn_output, transpose=True)  # undo rotation

        # ------------------------------------------------------------------
        # 7.  Final output projection (unchanged)         -------------------
        # ------------------------------------------------------------------

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


# ---------------------------------------------------------------------------
#                     Block-diagonal rotation variant
# ---------------------------------------------------------------------------


class MyAttnBlock(LlamaAttention):
    """Attention layer with **block-diagonal** secret rotation (one per head).

    R_big has the same dimensionality as *q_proj* (hidden_size × hidden_size).
    It consists of *num_heads* independent orthogonal blocks of size
    ``head_dim × head_dim``.  Consequently each individual attention head is
    rotated by its own matrix.
    """

    def __init__(self, config, layer_idx: int, R_big: torch.Tensor):
        super().__init__(config, layer_idx)

        hidden_size = config.hidden_size
        if R_big.shape != (hidden_size, hidden_size):
            raise ValueError(f"R_big must be [{hidden_size}, {hidden_size}] but got {list(R_big.shape)}")

        self.register_buffer("R_big", R_big, persistent=False)

        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim

    # helper ---------------------------------------------------------------

    def _rotate(self, tensor: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        """Rotate (or un-rotate) *tensor* with block-diag matrix.

        tensor shape: (B, H, S, D)
        R_big shape:  (hidden_size, hidden_size)
        If *transpose* is True we use R_bigᵀ.
        """

        B, H, S, D = tensor.shape
        hidden_size = H * D
        t = tensor.permute(0, 2, 1, 3).reshape(B, S, hidden_size)

        R = self.R_big.T if transpose else self.R_big

        original_dtype = t.dtype
        t_rot = torch.matmul(t.to(R.dtype), R).to(original_dtype)

        t_rot = t_rot.reshape(B, S, H, D).permute(0, 2, 1, 3)
        return t_rot

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # --- rotate per head --------------------------------------------
        query_states = self._rotate(query_states)
        key_states = self._rotate(key_states)
        value_states = self._rotate(value_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        import pdb;pdb.set_trace()
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        chunk_size = 4

        # Apply attention per-head chunks (query/key/value shapes B,H,S,D)
        attn_output, attn_weights = _attention_in_chunks(
            self,
            attention_interface,
            query_states,
            key_states,
            value_states,
            attention_mask,
            chunk_size=chunk_size,
            kwargs=kwargs,
        )  # (B, S, H, D)

        # reorder to (B,H,S,D) for rotation removal
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = self._rotate(attn_output, transpose=True)
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


def get_model_R(model: str, *, r_dtype: torch.dtype = torch.float32):
    """Load a Llama-based model and protect every attention layer with MyAttnBlock.

    A fresh, *block-diagonal* orthogonal matrix **Rᵢ** is sampled per decoder
    layer so that **each attention head gets its own secret rotation**.  The
    model is returned as an HF *text-generation* pipeline ready for use.
    """

    # ------------------------------------------------------------------
    # 1.  Load model & tokenizer
    # ------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prefer float16/bfloat16 on GPU and float32 on CPU to avoid overflows.
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model_obj = LlamaForCausalLM.from_pretrained(model, torch_dtype=dtype).to(device)

    # Ensure we always have a valid attention implementation flag.
    if getattr(model_obj.config, "_attn_implementation", None) is None:
        model_obj.config._attn_implementation = "sdpa"

    # ------------------------------------------------------------------
    # 2.  Replace every self-attention module with MyAttnBlock
    # ------------------------------------------------------------------

    hidden_size = model_obj.config.hidden_size
    num_heads = model_obj.config.num_attention_heads
    head_dim = model_obj.config.head_dim

    for layer_idx, layer in enumerate(model_obj.model.layers):
        attn: LlamaAttention = layer.self_attn  # type: ignore[attr-defined]

        # Create block-diagonal orthogonal matrix (one block per head)
        # Generate block-diagonal rotation in the *requested* dtype (default
        # fp32) regardless of model precision.
        R_big = dense_block_orthogonal(
            num_heads,
            head_dim,
            device=device,
            dtype=r_dtype,
        )

        # Build wrapped attention layer and copy parameters
        wrapped = MyAttnBlock(model_obj.config, layer_idx, R_big=R_big)
        wrapped.load_state_dict(attn.state_dict(), strict=False)

        # Preserve training mode & device
        wrapped.to(device=device, dtype=attn.q_proj.weight.dtype)
        if not model_obj.training:
            wrapped.eval()

        layer.self_attn = wrapped

    # ------------------------------------------------------------------
    # 3.  Return a text-generation pipeline (same device)
    # ------------------------------------------------------------------

    device_id = device.index if device.type == "cuda" else -1
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_obj,
        tokenizer=tokenizer,
        torch_dtype=dtype,
        device=device_id,
    )

    return pipeline




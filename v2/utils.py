


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
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

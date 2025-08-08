import torch


from utils import (
    MyAttn,
    MyAttnBlock,
    dense_random_orthogonal,
    random_permutation_matrix,
    dense_block_orthogonal,
    permutation_block_matrix,
)


# ---------------------------------------------------------------------------
# Helper utilities for the tests
# ---------------------------------------------------------------------------


def _get_dummy_inputs(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
):
    """Return random hidden states together with *identity* rotary embeddings."""

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)

    # Identity rotary embedding (cos = 1, sin = 0) so that the projection is a
    # no-op yet retains the expected shape.
    head_dim = hidden_size // num_heads
    cos = torch.ones(batch_size, seq_len, head_dim, dtype=dtype, device=device)
    sin = torch.zeros_like(cos)

    return hidden_states, (cos, sin)


def _run_single_test(R_generator, atol: float = 1e-5, rtol: float = 1e-5):
    """Compare the output of *LlamaAttention* with *MyAttn* for a given R."""

    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig

    # --- tiny config -------------------------------------------------------
    config = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
    )

    # The currently installed *transformers* release uses a private attribute
    # ``_attn_implementation`` to select between different backend
    # implementations (eager, sdpa, flash, ...).  The default (`None`) causes
    # a run-time *KeyError*.  We explicitly pick the vanilla scaled-dot-product
    # attention (SDPA) variant.
    config._attn_implementation = "sdpa"

    baseline_attn = LlamaAttention(config, layer_idx=0)

    head_dim = config.head_dim  # == hidden_size // num_heads
    R = R_generator(head_dim, device=baseline_attn.q_proj.weight.device, dtype=baseline_attn.q_proj.weight.dtype)

    my_attn = MyAttn(config, layer_idx=0, R=R)

    # make sure both modules share the *same* weights so that we only test the
    # difference introduced by the rotations.
    my_attn.load_state_dict(baseline_attn.state_dict(), strict=False)

    baseline_attn.eval()
    my_attn.eval()

    # ---------------------------------------------------------------------
    batch_size, seq_len = 2, 5
    hidden_states, position_embeddings = _get_dummy_inputs(
        batch_size, seq_len, config.hidden_size, config.num_attention_heads
    )

    with torch.no_grad():
        out_baseline, _ = baseline_attn(hidden_states, position_embeddings, attention_mask=None)
        out_my, _ = my_attn(hidden_states, position_embeddings, attention_mask=None)

    assert torch.allclose(out_baseline, out_my, atol=atol, rtol=rtol), (
        "MyAttn does not match baseline attention for the given rotation matrix "
        f"generator {R_generator.__name__}."
    )


# ---------------------------------------------------------------------------
# Block-diagonal variant helpers
# ---------------------------------------------------------------------------


def _run_block_test(R_block_generator, atol: float = 1e-5, rtol: float = 1e-5):
    """Compare baseline vs MyAttnBlock using per-head rotation."""

    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig

    config = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
    )
    config._attn_implementation = "sdpa"

    baseline_attn = LlamaAttention(config, layer_idx=0)

    hidden_size = config.hidden_size
    head_dim = config.head_dim
    num_heads = config.num_attention_heads

    R_big = R_block_generator(num_heads, head_dim, device=baseline_attn.q_proj.weight.device,
                              dtype=baseline_attn.q_proj.weight.dtype)

    my_attn = MyAttnBlock(config, layer_idx=0, R_big=R_big)

    my_attn.load_state_dict(baseline_attn.state_dict(), strict=False)

    baseline_attn.eval()
    my_attn.eval()

    batch_size, seq_len = 2, 5
    hidden_states, position_embeddings = _get_dummy_inputs(
        batch_size, seq_len, hidden_size, num_heads
    )

    with torch.no_grad():
        ref, _ = baseline_attn(hidden_states, position_embeddings, attention_mask=None)
        out, _ = my_attn(hidden_states, position_embeddings, attention_mask=None)

    assert torch.allclose(ref, out, atol=atol, rtol=rtol), (
        "MyAttnBlock does not match baseline attention for rotation generator "
        f"{R_block_generator.__name__}."
    )


def test_dense_random_orthogonal():
    _run_single_test(dense_random_orthogonal)


def test_random_permutation_matrix():
    _run_single_test(random_permutation_matrix)


# block-diagonal -----------------------------------------------


def test_block_dense_random_orthogonal():
    _run_block_test(dense_block_orthogonal)


def test_block_random_permutation():
    _run_block_test(permutation_block_matrix)


# ---------------------------------------------------------------------------
#  Precision robustness tests
# ---------------------------------------------------------------------------


def _supports_dtype(dtype: torch.dtype) -> bool:
    if dtype == torch.float32:
        return True
    if dtype == torch.bfloat16:
        return torch.cuda.is_available() or torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    if dtype == torch.float16:
        return torch.cuda.is_available()
    return False


def test_rotation_precision():
    """Ensure MyAttn is numerically stable when model & R share dtype."""

    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig

    dtypes = [torch.float32, torch.bfloat16, torch.float16]

    for dtype in dtypes:
        if not _supports_dtype(dtype):
            print(f"[SKIP] dtype {dtype} not supported")
            continue

        device = torch.device("cuda:0" if torch.cuda.is_available() and dtype != torch.float32 else "cpu")

        config = LlamaConfig(hidden_size=32, num_attention_heads=4, intermediate_size=64, num_hidden_layers=1)
        config._attn_implementation = "eager"

        baseline = LlamaAttention(config, layer_idx=0).to(device=device, dtype=dtype)

        R = dense_random_orthogonal(config.head_dim, device=device, dtype=dtype)
        my_attn = MyAttn(config, layer_idx=0, R=R).to(device=device, dtype=dtype)
        my_attn.load_state_dict(baseline.state_dict(), strict=False)

        hidden_states, position_embeddings = _get_dummy_inputs(
            2, 64, config.hidden_size, config.num_attention_heads, dtype=dtype, device=device
        )

        with torch.no_grad():
            ref, _ = baseline(hidden_states, position_embeddings, attention_mask=None)
            out, _ = my_attn(hidden_states, position_embeddings, attention_mask=None)

        atol = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        assert torch.allclose(ref, out, atol=atol, rtol=1e-3), f"Mismatch for dtype {dtype}"


def test_bf16_model_various_R():
    """Model in bfloat16, compare R in fp32, bf16, fp16."""

    if not _supports_dtype(torch.bfloat16):
        print("[SKIP] bfloat16 not supported")
        return

    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = LlamaConfig(hidden_size=32, num_attention_heads=4, intermediate_size=64, num_hidden_layers=1)
    config._attn_implementation = "eager"

    baseline = LlamaAttention(config, layer_idx=0).to(device=device, dtype=torch.bfloat16)

    for r_dtype in [torch.float32, torch.bfloat16, torch.float16]:
        if not _supports_dtype(r_dtype):
            print(f"[SKIP] R dtype {r_dtype} unsupported")
            continue

        # Build MyAttn *after* baseline is already in bf16 (option 1). We keep
        # the module's parameters in bf16 but restore R to its own precision.

        R = dense_random_orthogonal(config.head_dim, device=device, dtype=r_dtype)

        my_attn = MyAttn(config, layer_idx=0, R=R).to(device=device)  # keep params float32 for a moment

        # Load identical weights and convert parameters to bf16 (without touching R)
        my_attn.load_state_dict(baseline.state_dict(), strict=False)
        my_attn.to(dtype=torch.bfloat16)
        my_attn.R.data = R  # restore R with its original dtype

        # Use a longer sequence to expose possible numeric divergence.
        hidden_states, position_embeddings = _get_dummy_inputs(
            batch_size=2,
            seq_len=64,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            dtype=torch.bfloat16,
            device=device,
        )

        with torch.no_grad():
            ref, _ = baseline(hidden_states, position_embeddings, attention_mask=None)
            out, _ = my_attn(hidden_states, position_embeddings, attention_mask=None)

        diff = (ref - out).abs().max().item()
        print(f"R dtype {r_dtype}: max|Δ| = {diff:.5f}")

        assert torch.allclose(ref, out, atol=5e-3, rtol=1e-3), f"Mismatch with R dtype {r_dtype}"


# ---------------------------------------------------------------------------
# Real-world config: CodeLlama-7B ------------------------------------------------
# ---------------------------------------------------------------------------


def test_codellama_attention():
    """Check that *MyAttn* stays numerically identical using CodeLlama-7B config.

    The test only depends on the configuration file (small download).  If the
    environment is offline or the HuggingFace hub is not reachable we skip the
    test gracefully.
    """

    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("codellama/CodeLlama-7b-hf", trust_remote_code=True)
    except Exception as err:  # noqa: BLE001 – broad to catch offline errors
        print(f"[SKIPPED] Could not load CodeLlama config – {err}")
        return

    # We only need a single attention layer, therefore we can keep the default
    # hidden size (4096) and #heads (32).  Memory usage stays reasonable.
    # Ensure a valid attention implementation flag.
    config._attn_implementation = "sdpa"

    from transformers.models.llama.modeling_llama import LlamaAttention

    baseline_attn = LlamaAttention(config, layer_idx=0)

    head_dim = config.head_dim
    # Standard (shared) rotation check ------------------------------------

    R = dense_random_orthogonal(head_dim, device=baseline_attn.q_proj.weight.device,
                                dtype=baseline_attn.q_proj.weight.dtype)

    from utils import MyAttn, MyAttnBlock, dense_block_orthogonal

    my_attn = MyAttn(config, layer_idx=0, R=R)

    # Copy weights so that both modules are identical except for the rotation.
    my_attn.load_state_dict(baseline_attn.state_dict(), strict=False)

    baseline_attn.eval()
    my_attn.eval()

    batch_size, seq_len = 2, 8
    hidden_states, position_embeddings = _get_dummy_inputs(
        batch_size, seq_len, config.hidden_size, config.num_attention_heads,
        dtype=baseline_attn.q_proj.weight.dtype,
        device=baseline_attn.q_proj.weight.device,
    )

    with torch.no_grad():
        ref, _ = baseline_attn(hidden_states, position_embeddings, attention_mask=None)
        out1, _ = my_attn(hidden_states, position_embeddings, attention_mask=None)

    assert torch.allclose(ref, out1, atol=1e-5, rtol=1e-5), "Mismatch for CodeLlama shared R."

    # Block–diagonal variant ---------------------------------------------

    R_big = dense_block_orthogonal(config.num_attention_heads, head_dim,
                                   device=baseline_attn.q_proj.weight.device,
                                   dtype=baseline_attn.q_proj.weight.dtype)

    my_attn_block = MyAttnBlock(config, layer_idx=0, R_big=R_big)
    my_attn_block.load_state_dict(baseline_attn.state_dict(), strict=False)
    my_attn_block.eval()

    with torch.no_grad():
        out2, _ = my_attn_block(hidden_states, position_embeddings, attention_mask=None)

    assert torch.allclose(ref, out2, atol=1e-5, rtol=1e-5), "Mismatch for CodeLlama block-diag R."


# ---------------------------------------------------------------------------
# When executed as a standalone script run the two tests so that the file does
# not depend on *pytest* being available in the environment.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    test_dense_random_orthogonal()
    test_random_permutation_matrix()
    test_block_dense_random_orthogonal()
    test_block_random_permutation()
    test_codellama_attention()
    test_rotation_precision()
    test_bf16_model_various_R()
    print("All MyAttn tests passed.")

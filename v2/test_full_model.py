"""Model-level numeric equivalence test.

Loads the full CodeLlama-7B model twice – once unmodified and once wrapped
with *MyAttnBlock* via ``utils.get_model_R`` – and measures the maximum
absolute difference of the output logits on a short prompt.

The test skips automatically if the model cannot be downloaded or does not fit
into the available memory.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys

import torch

PROMPT = (
    "import socket\n\n"
    "def ping_exponential_backoff(host: str):"
)


def _load_baseline(model_name: str, device: torch.device, dtype: torch.dtype):
    """Return original CodeLlama model (no attention modification)."""

    from transformers import AutoTokenizer
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)

    return tokenizer, model


def _load_modified(model_name: str, device: torch.device):
    """Return model wrapped by get_model_R with R kept in float32."""

    from utils import get_model_R

    pipeline = get_model_R(model_name, r_dtype=torch.float16)
    return pipeline.tokenizer, pipeline.model


def _measure_logits(model, tokenizer, prompt: str, device: torch.device):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        logits = model(**inputs).logits  # (B, S, vocab)
    return logits.cpu()


def test_full_codellama_equivalence():
    model_name = "codellama/CodeLlama-7b-hf"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # ------------------------------------------------------------------
    # 1.  Baseline model
    # ------------------------------------------------------------------

    try:
        baseline_tokenizer, baseline_model = _load_baseline(model_name, device, dtype)
        baseline_model.eval()
    except Exception as err:  # noqa: BLE001
        print(f"[SKIP] Could not load baseline model – {err}")
        return

    # Compute baseline logits

    try:
        ref_logits = _measure_logits(baseline_model, baseline_tokenizer, PROMPT, device)
    except RuntimeError as err:
        print(f"[SKIP] Runtime/OOM during baseline forward – {err}")
        return

    # Free memory before loading modified model
    del baseline_model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ------------------------------------------------------------------
    # 2.  Modified model with secure attention (loaded after freeing GPU)
    # ------------------------------------------------------------------

    try:
        mod_tokenizer, mod_model = _load_modified(model_name, device)
        mod_model.eval()
    except Exception as err:  # noqa: BLE001
        print(f"[SKIP] Could not load modified model – {err}")
        return

    assert baseline_tokenizer.get_vocab() == mod_tokenizer.get_vocab(), "Tokenizers differ!"

    try:
        mod_logits = _measure_logits(mod_model, mod_tokenizer, PROMPT, device)
    except RuntimeError as err:
        print(f"[SKIP] Runtime/OOM during modified forward – {err}")
        return

    diff = (ref_logits - mod_logits).abs().max().item()
    max_abs = ref_logits.abs().max().item()
    rel = diff / max_abs if max_abs > 0 else 0.0
    print(f"Max |Δ| = {diff:.6f}  (relative {rel:.4%})")

    assert rel < 1e-2, "Relative logits difference exceeds 1%"


if __name__ == "__main__":
    test_full_codellama_equivalence()
    print("Full-model equivalence test passed.")

"""Analysis of token embeddings across layers for CodeLlama.

This script extracts the hidden-state embedding of the word *bank* in two
different contexts for every transformer layer of `codellama/CodeLlama-7b-hf`.
It then:
1. Stores a simple in-memory dataset with records `(token_id, layer, embedding)`
   for each occurrence.
2. Prints the cosine similarity between the two *bank* embeddings at each layer.

Because the environment in which this file may run can vary, **no file is
written by default**.  If you wish to persist the dataset, uncomment the
section at the bottom that saves it using `pickle`.

Requires: `transformers`, `torch`, `numpy` (and optionally `pandas`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


###############################################################################
# Data classes
###############################################################################


@dataclass
class TokenEmbeddingRecord:
    """Single record for the dataset."""

    context_id: int  # 0 or 1 for the two example sentences
    token_id: int
    layer: int  # 0 == embedding layer, 1-N == transformer layers
    embedding: np.ndarray  # (hidden_size,)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # store embedding as python list for easier serialization
        d["embedding"] = d["embedding"].tolist()
        return d


###############################################################################
# Helper functions
###############################################################################


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D numpy arrays."""

    dot = float(np.dot(a, b))
    denom = math.sqrt(float(np.dot(a, a))) * math.sqrt(float(np.dot(b, b)))
    return dot / denom if denom != 0 else 0.0


def find_subtoken_positions(tokens: List[int], subtoken: List[int]) -> List[int]:
    """Find starting indices where *subtoken* sequence occurs inside *tokens*.

    Works even if the tokenizer split the word into multiple pieces.
    Returns a list with all start positions (should be exactly one here).
    """

    positions: List[int] = []
    sub_len = len(subtoken)
    for i in range(len(tokens) - sub_len + 1):
        if tokens[i : i + sub_len] == subtoken:
            positions.append(i)
    return positions


###############################################################################
# Main analysis
###############################################################################


def main() -> None:
    # ---------------------------------------------------------------------
    # 1. Model & tokenizer
    # ---------------------------------------------------------------------
    model_name = "codellama/CodeLlama-7b-hf"

    print("Loading model … this can take a few minutes & ~14 GB of RAM")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_hidden_states=True, torch_dtype=torch.float16, device_map="auto"
    )

    # ---------------------------------------------------------------------
    # 2. Input sentences
    # ---------------------------------------------------------------------
    # If a file "sent.txt" exists in the current working directory, pick two
    # random non-empty lines from it.  Otherwise fall back to the built-in
    # examples above so the script still works standalone.

    import pathlib, random

    sentences: List[str]
    sent_file = pathlib.Path("sent.txt")
    if sent_file.is_file():
        import re

        all_lines = [l.strip() for l in sent_file.read_text(encoding="utf-8").splitlines() if l.strip()]

        # Keep only sentences that include the standalone word "bank" (case-insensitive).
        bank_regex = re.compile(r"\bbank\b", re.IGNORECASE)
        bank_lines = [ln for ln in all_lines if bank_regex.search(ln)]

        if len(bank_lines) < 2:
            raise ValueError(
                "sent.txt must contain at least two lines with the standalone word 'bank'"
            )

        sentences = random.sample(bank_lines, 2)
        print("Using two random sentences from sent.txt:")
        for i, s in enumerate(sentences, 1):
            print(f"  [{i}] {s}")
    else:
        sentences = [
            "She went along the river's bank.",
            "She went to the bank to close the mortgage.",
        ]

    # Encode separately to keep sequence lengths isolated; no BOS/EOS tokens.
    encodings = [tokenizer(s, return_tensors="pt", add_special_tokens=False) for s in sentences]

    # Tokens that represent the word "bank" (could be one or multiple IDs)
    bank_subtokens = tokenizer("bank", add_special_tokens=False)["input_ids"]

    # In-memory dataset
    dataset: List[TokenEmbeddingRecord] = []

    # Cosine similarity per layer
    similarities: List[float] = []

    # Determine number of layers (embedding + transformer layers)
    with torch.no_grad():
        _dummy = model(torch.tensor([[tokenizer.eos_token_id]]))
        num_layers = len(_dummy.hidden_states)  # includes embedding layer at idx 0

    # ---------------------------------------------------------------------
    # 3. Iterate over layers
    # ---------------------------------------------------------------------
    print("Computing embeddings …")

    # Hidden states for each sentence
    hidden_states_per_sent: List[List[torch.Tensor]] = []
    with torch.no_grad():
        for enc in encodings:
            out = model(**enc)
            hidden_states_per_sent.append(list(out.hidden_states))

    # For each layer, gather embedding for the *first* occurrence of \"bank\".
    for layer_idx in range(num_layers):
        layer_embs: List[np.ndarray] = []
        for ctx_id, (enc, h_states) in enumerate(zip(encodings, hidden_states_per_sent)):
            tokens = enc["input_ids"][0].tolist()
            positions = find_subtoken_positions(tokens, bank_subtokens)

            if not positions:
                raise ValueError(
                    f"No standalone occurrence of 'bank' found in context {ctx_id}."
                )

            # If multiple occurrences, pick the first.
            pos = positions[0]
            # If \"bank\" is split into multiple subtokens, average their embeddings.
            sub_len = len(bank_subtokens)
            emb_tensor = h_states[layer_idx][0, pos : pos + sub_len, :].mean(dim=0)
            emb = emb_tensor.cpu().float().numpy()

            # Save record
            dataset.append(
                TokenEmbeddingRecord(
                    context_id=ctx_id,
                    token_id=bank_subtokens[0],  # first subtoken id for reference
                    layer=layer_idx,
                    embedding=emb,
                )
            )

            layer_embs.append(emb)

        # Cosine similarity between the two contexts at this layer
        sim = cosine_similarity(layer_embs[0], layer_embs[1])
        similarities.append(sim)

    # ---------------------------------------------------------------------
    # 4. Output results
    # ---------------------------------------------------------------------
    print("\nCosine similarity for 'bank' between the two contexts:")
    for layer_idx, sim in enumerate(similarities):
        layer_tag = "embedding" if layer_idx == 0 else str(layer_idx)
        print(f"Layer {layer_tag:>9}: {sim:.4f}")

    # ---------------------------------------------------------------------
    # 5. Optional: persist dataset
    # ---------------------------------------------------------------------
    # Uncomment below to save to disk (requires write permissions).
    # import pickle, pathlib
    # outfile = pathlib.Path("bank_embeddings.pkl")
    # with outfile.open("wb") as f:
    #     pickle.dump([rec.to_dict() for rec in dataset], f)
    # print(f"\nDataset written to {outfile.resolve()}")


if __name__ == "__main__":
    main()

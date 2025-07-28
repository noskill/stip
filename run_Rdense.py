
import copy

import torch
from transformers import AutoConfig, LlamaConfig, LlamaForCausalLM

from utils import (
    get_model_R_dense,
    model as MODEL_NAME,
    replace_attention_RoPE_dense,
    replace_rms_with_rotated,
    dense_orthogonal_R,
    _rotate_model_parameters_with_R,
)

    
def main():
    # Demonstrate text generation with the rotated model pipeline
    pipeline = get_model_R_dense()
    prompt = "import socket\n\n" \
             "def ping_exponential_backoff(host: str):"
    outputs = pipeline(
        prompt,
        max_length=200,
        do_sample=True,
        temperature=0.1,
        top_k=10,
        top_p=0.95,
    )
    print("Decoded generation:", outputs[0]["generated_text"])

if __name__ == '__main__':
    main()

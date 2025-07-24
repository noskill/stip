
from utils import get_model_R_dense, model as MODEL_NAME, replace_attention_RoPE_dense
import torch
from transformers import AutoConfig


def test_block_reversability():
    # Build a random permutation matrix π
    config = AutoConfig.from_pretrained(MODEL_NAME)
    d = config.hidden_size

    # Create a text-generation pipeline with permuted parameters
    pipeline = get_model_R_dense()

    # Generate text
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
    test_block_reversability()

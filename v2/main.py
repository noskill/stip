from utils import get_model_R
import torch
from transformers import AutoConfig

if __name__ == "__main__":
    # Build a random permutation matrix π
    model = "codellama/CodeLlama-7b-hf"
    config = AutoConfig.from_pretrained(model)

    pipeline = get_model_R(model)

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



from utils import get_model, model as MODEL_NAME
import torch
from transformers import AutoConfig

if __name__ == "__main__":
    # Build a random permutation matrix π
    config = AutoConfig.from_pretrained(MODEL_NAME)
    d = config.hidden_size
    perm = torch.randperm(d)
    pi = torch.eye(d, dtype=torch.float32)[perm]

    # Create a text-generation pipeline with permuted parameters
    pipeline = get_model(pi=pi)

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
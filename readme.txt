# model transformation

Let x ∈ R n×d denote the input where n is the sequence length (e.g., the number of tokens) and d is the model dimension.
We define a Transformer block as a function fθ : R^n×d → R^n×d with trainable parameters θ. Then the Transformer
inference, i.e., fθ(x) = y, is computed as follows:
    Q = x Wq, K = x Wk, V = x Wv, where Wq, Wk, Wv ∈ R^d×d

    u = softmax (Q K' / √k + M) V Wo, where M ∈ R^n×n, Wo ∈ R^d×d
 
    v = LayerNorm(u + x; γ1, β1), where γ1, β1 ∈ R^d

    z = ReLU(v W1)W2, W1 ∈ R where d×m, W2 ∈ R^m×d

    y = LayerNorm(z + v; γ2, β2), where γ1, β1 ∈ R^d

where k is a constant equal to d divided by the number of attention heads, M denotes the mask which is an all-zero
matrix in the encoder and a matrix whose upper right corner (not including the diagonal) is negative infinity in
the decoder. The parameter θ consists of attention weights (Wq, Wk, Wv, Wo), feedforward weights (W1, W2) and
LayerNorm weights (γ, β).

Let π ∈ {0, 1} d×d denote a permutation matrix. We transform the parameters θ as follows:
Wq′ = π' Wq 
Wk′ = π' Wk
Wv′ = π' Wv 
W1′ = π' W1
Wo′ = Wo π
W2′ = W2 π
γ1′ = γ1 π
β1′ = β1 π 
γ2′ = γ2 π 
β2′ = β2 π


# inference

at inference it works as follows:

1. transform x to x′ = x π
2. compute fθ′(x′) = y′
3. compute y = y′ π'

we need to emulate the situation when the host where llm inference is taking place don't have access to π or original θ,
but currently last layer of LlamaForCausalLM performs reverse transformation, so the "host" has access to llm output

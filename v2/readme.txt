All operations except attention are executed in TEE. Q @ K.T product and softmax is computed on untrusted device.

Let

 h ∈ ℝⁿˣᵈ   (original mini-batch of hidden states, n = seq-len)
 R ∈ ℝᵈˣᵈ   secret orthogonal matrix(might just be permutation matrix) (R Rᵀ = I)

New R is generated on each run.

 ──────────────────────────── Attention with rotation ───────────────────────
Compute in TEE

Queries:
Q = h Wq (2)

Keys K = h Wk (3)

Values V′ = h Wv R = h Wv R = V R (4)

apply RoPE
Q_rope = apply_rope(Q)
K_rope = apply_rope(K)
apply R:

Q′ = Q_rope R
K′ = K_rope R
V′ = V R

Move to untrusted GPU.

Scaled scores per attention head:

S′ = (Q′ K′ᵀ)/√dₖ
= (Q R)(K R)ᵀ/√dₖ
= Q (R Rᵀ) Kᵀ/√dₖ
= Q Kᵀ/√dₖ = S (5)

So S′ is the same scores that we would get in unmodified llm execution.

For security Q′ K′ V′ are split in blocks by n attention heads, for example 4.

This step significanlty emproves security. It might be hard for untrusted GPUs(one or many) to 
gues where they are computing equation (5) from the same prompt or from different prompts.

Attention prob A′ = softmax(S′) = softmax(S) = A (6)


U′ = A V′ = A (V R) = U R  (7)
 
Move to TEE:

gather attention heads into U′

Output proj. O = U′ Wo′ = (U R)(Rᵀ Wo ) = U Wo (8)


Residual h_out = h + O

─────────────────────── end ───────────────────────



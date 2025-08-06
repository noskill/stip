All operations except attention are executed in TEE. Q @ K.T product and softmax is computed on untrusted device.

Let

 h ∈ ℝⁿˣᵈ   (original mini-batch of hidden states, n = seq-len)
 R ∈ ℝᵈˣᵈ   secret orthogonal matrix(might just be permutation matrix) (R Rᵀ = I)

We build the modified model inside the TEE:

 Wq′ = Wq R   Wk′ = Wk R   Wv′ = Wv R   Wo′ = Rᵀ Wo (1)

 ──────────────────────────── Attention with rotation ───────────────────────
Compute in TEE

Queries:
Q′ = h Wq R = Q R (2)

Keys K′ = h Wk R = K R (3)

Values V′ = h Wv R = h Wv R = V R (4)

Move to untrusted GPU:

Scaled scores per attention head:

S′ = (Q′ K′ᵀ)/√dₖ
= (Q R)(K R)ᵀ/√dₖ
= Q (R Rᵀ) Kᵀ/√dₖ
= Q Kᵀ/√dₖ = S (5)

So S′ is the same scores that we would get in unmodified llm execution.

Attention prob A′ = softmax(S′) = softmax(S) = A (6)


U′ = A V′ = A (V R) = U R  (7)
 
Move to TEE:

Output proj. O = U′ Wo′ = (U R)(Rᵀ Wo ) = U Wo = O (8)


Residual h_out = h + O

─────────────────────── end ───────────────────────



Pytorch linear layer implementation stores internally Wᵀ and implements y = xW + b 

as

y = x(Wᵀ)ᵀ + b
given that 
(A·B)ᵀ = Bᵀ·Aᵀ

in derivation we have
W` = W·R
then W`ᵀ = (W·R)ᵀ = Rᵀ Wᵀ

update in code:
weight = Rᵀ @ weight


and for Rᵀ Wo

Wo′ = Rᵀ Wo

then Wo′ᵀ = (Rᵀ Wo)ᵀ = Woᵀ (Rᵀ)ᵀ = Woᵀ R

update in code:
weight =  weight @ R

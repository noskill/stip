Let

 h ∈ ℝⁿˣᵈ   (original mini-batch of hidden states, n = seq-len)
 R ∈ ℝᵈˣᵈ   user-secret orthogonal matrix (R Rᵀ = I)

We build the “masked’’ model inside the TEE:

 Wq′ = Rᵀ Wq R   Wk′ = Rᵀ Wk R   Wv′ = Rᵀ Wv R   Wo′ = Rᵀ Wo R (1)

and rotate every activation that leaves / enters the un-trusted GPUs

 h′ = h R. (2)
 
 
 
 ──────────────────────────── Attention with rotation ───────────────────────

Queries:
Q′ = h′Wq′ = (h R) (Rᵀ Wq R) = h Wq R = Q R (3)

Keys K′ = h′ Wk′ = h Wk R = K R (4)

Values V′ = h′ Wv′ = h Wv R = V R (5)

Scaled scores:

S′ = (Q′ K′ᵀ)/√dₖ
= (Q R)(K R)ᵀ/√dₖ
= Q (R Rᵀ) Kᵀ/√dₖ
= Q Kᵀ/√dₖ = S (6)

Attention prob A′ = softmax(S′) = softmax(S) = A (7)


U′ = A′ V′ = A (V R) = U R  (8)
 
 
Output proj. O′ = U′ Wo′ = (U R)(Rᵀ Wo R) = U Wo R = O R


Residual h_out′ = h′ + O′ = (h+O) R


Thus every intermediate that leaves the block carries the rotation on the
right:

 Q′ = Q R, K′ = K R, V′ = V R, U′ = U R, h_out′ = h_out R. (11)

In compact form

 f′(h R) = f(h) R (12)
 
 
 ──────────────────────────── RMS-Norm ───────────────────────

inputs are left-multiplied by R, so we unrotate by multiplying by Rᵀ
 
h = h′ Rᵀ
then run rmsnorm:
r = √(mean(h² )) 
ŷ = h / r 
y = γ ⊙ ŷ 

then rotate with R again:

y′ = yR



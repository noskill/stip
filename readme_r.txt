Let

 h ∈ ℝⁿˣᵈ   (original mini-batch of hidden states, n = seq-len)
 R ∈ ℝᵈˣᵈ   user-secret orthogonal matrix (R Rᵀ = I)

We build the “masked’’ model inside the TEE:

 Wq′ = Rᵀ Wq R   Wk′ = Rᵀ Wk R   Wv′ = Rᵀ Wv R   Wo′ = Rᵀ Wo R (1)

and rotate every activation that leaves / enters the un-trusted GPUs

 h′ = h R. (2)
 
 
Pytorch linear layer implements stores internally Wᵀ and implements y = xW + b 

as

y = x(Wᵀ)ᵀ + b
given that 
(A·B·C)ᵀ = Bᵀ·Aᵀ·Cᵀ

in derivation we have
W` = Rᵀ·W·R
then W`ᵀ = (Rᵀ·W·R)ᵀ = Rᵀ Wᵀ (Rᵀ)ᵀ =  Rᵀ Wᵀ R

update in code:
weight = Rᵀ @ weight @ R


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
 
──────────────────────────── SwiGLU ──────────────────────────── 
 Below is the short algebra that shows the feed-forward / SwiGLU block
behaves exactly the same way as the attention block once every weight
matrix has been rewritten with the rotation (or permutation) R.

Notation – original model
 x   : (batch,seq,d) hidden state that enters the MLP
 W₁ , W₃ ∈ ℝ^{d×m}  gate-proj / up-proj
 W₂  ∈ ℝ^{m×d}    down-proj
 σ(·) : SiLU (≈ x · sigmoid x)
 ⊙ : element-wise product

 FFN(x) = ( (x W₁) · σ(x W₁) ⊙ (x W₃) ) W₂ 

Transformation we apply once off-line

 x′ = x R (2)
 W₁′ = Rᵀ W₁ , W₃′ = Rᵀ W₃ , W₂′ = W₂ R 

(the same rule we already use for q/k/v/o).

Step-by-step inside the rotated model
1 Gate / up projections
 x′ W₁′ = (x R)(Rᵀ W₁) = x W₁
 x′ W₃′ = (x R)(Rᵀ W₃) = x W₃ 

 → the vectors that the non-linearities see are identical to the
  baseline ones.

2 SwiGLU non-linearity (SiLU + ⊙)
 All operations are element-wise, so the result is the same tensor that
 the baseline would produce:

 g = (x W₁) · σ(x W₁) ⊙ (x W₃) 
 g′ (x′ computed in rotated model) = g 

3 Down projection
 y′ = g′ W₂′ = g (W₂ R) = ( g W₂ ) R = y R (7)
 
 ──────────────────────────── RMS-Norm ───────────────────────

inputs are left-multiplied by R, so we unrotate by multiplying by Rᵀ
 
h = h′ Rᵀ
then run rmsnorm:
r = √(mean(h² )) 
ŷ = h / r 
y = γ ⊙ ŷ 

then rotate with R again:

y′ = yR

───────────────────────────── RoPE ─────────────────────────────

The rotation matrix R can be chosen in two flavours:

1. RoPE–compatible R (block 2 × 2 inside every head)
    • keeps the (x0 , x1), (x2 , x3)… pairs that Rotary Positional
    Embedding expects;
    • RoPE can run on the un-trusted GPUs exactly where it was in the
    original model;
    • drawback – R is easy to recover (row / Procrustes attack).
2. RoPE–incompatible R (dense, mixes coordinates across heads)
    • maximises secrecy: recovering R from W and W′ is practically
    impossible;
    • RoPE must then be executed inside the TEE, because outside GPUs see
    vectors already rotated by R and the original “pair” assumption is
    gone.
    Both variants keep the algebra of the main README intact; only the place
    where RoPE is applied changes.
    
in this case addition transformation from R space back to original space is needed:
GPU side (R-space)              TEE side

q′ = (h R) @ (Rᵀ Wq R) …  ─────▶  q = q′ Rᵀ
k = k′ Rᵀ
(q,k) = apply_rotary_pos_emb(q,k)
q′ = q R
k′ = k R  ─────▶ back to GPU
softmax(q′ k′ᵀ /√d) …


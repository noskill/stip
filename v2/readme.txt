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

Below is a back-of-the-envelope breakdown of how much compute (FLOPs) lands in the                          [190/1981]
TEE vs. on the un-trusted accelerator for CodeLlama-7B when processing a single prompt                                
of length L ∈ {8, 16, 32, 64, 128, 256, 512} tokens.                                                                  
                                                                                                                      
Assumptions                                                                                                           
• CodeLlama-7B: 32 decoder layers, hidden = 4096, heads = 32, head-dim = 128, FFN dim = 11                            
008.                                                                                          
• 1 multiply-add = 2 FLOPs.                                                                                           
• Only „big“ ops counted (layer norms, bias adds, activations, etc. ignored).                                         
• TEE executes every operation except the two matrix multiplications inside                                           
  the attention soft-max block (Q Kᵀ and A V). Those two go to the un-trusted device.                                 
                                                                                                                      
Per-layer FLOPs                                                                     
                                                                                                                      
  TEE                                                                                                                 
    • Q/K/V/O projections: 4 · 4096 × 4096  →  67 M FLOPs / token                                                     
    • Feed-forward (gate, up, down): 3 · 4096 × 11 008  → 135 M FLOPs / token                                         
    ⇒ 202 M FLOPs per token → 0.202 · L B FLOPs per layer                                                             
                                                                                                                      
  Un-trusted accelerator                                                                                              
    • Q Kᵀ : 32 · L² · 128                                                                                            
    • A V  : 32 · L² · 128                                                                                            
    ⇒ 8 192 · L² FLOPs ≈ 0.008192 · L² B FLOPs per layer 

When amount of compute is equal:
TEE                     untrusted
32 * 0.202 * L  = 32 * 0.008192 * L**2
6.464 L =  0.262144 L**2
0.262144 L**2 - 6.464 L =  0

D = B**2 - 4a c
D = 41.78329600000001

L = 6.464 +- 6.464 / (2 * 0.262144) = 24.658203125000004


            ┌───────  TEE  ──────────┐                       ┌──── Un-trusted ──────────┐                     
L (tokens)    6.464 · L  (B FLOPs)   |   Un-trusted / TEE    |   0.262144 · L² (B FLOPs)                          
─────────   ─────────────────────────┼───────────────────────┼───────────────────────────                         
    8                 51.7 B         |       0.04 ×          |     2.1 B                                     
    16                103.4 B        |       0.16 ×          |     16.8 B                                     
    32                206.9 B        |       1.30 ×          |     268.4 B                                     
    64                413.7 B        |       2.60 ×          |     1 074  B                                      
    128               827.3 B        |       5.20 ×          |     4 298  B                                      
    256               1 654.6 B      |      10.40 ×          |     17 193  B
    512               3 309.1 B      |      20.80 ×          |     68 719  B

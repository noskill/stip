Good enough llm obsfurcation(GELO) algorithm

We adress the situation when we have trusted execution environment
which handles LLM inference for users, but we would like to offload some computations to
the untrusted device.

Offloading Q, K, V projections

We accumulate and build a batch from different users to decrease correlation of embeddings in the batch.


          tokens × dim         dim × proj-out
    H  ∈  ℝ^{500 × 4096},   W ∈ ℝ^{4096 × 4096}

We want, at the end, the usual projection

          Q = H · W   ∈ ℝ^{500 × 4096}.               (1)

To hide the individual rows of H we multiply **on the left** by an
orthogonal 500 × 500 matrix A that is known only to the trusted side.

Steps batch-by-batch
────────────────────────────────────────
Trusted side
1.  draw a fresh dense orthogonal A  ∈ ℝ^{500 × 500}
2.  compute                U =  A · H          ⟨––– cost  d·n²
                          (shape 500 × 4096)
3.  send U to the un-trusted GPU

Un-trusted GPU
4.  compute                Y =  U · W = A · H · W           ⟨––– cost  d·n·p
                          (shape 500 × 4096)
5.  return Y

Trusted side
6.  compute          Q =  A^{-1} · Y =  A^{-1} · A · H · W         ⟨––– cost  d·n²
                          (because A^{-1}A = I)
7.  Q is exactly H·W as in (1).

Checks
────────────────────────────────────────
•  Dimensions line up: (500×500)(500×4096)=500×4096, then
   (500×500)^{-1}(500×4096)=500×4096.

•  FLOPs per 500-token batch (Llama-2-7B/13B numbers)

      mix  A·H   :  d·n² = 4096 · 500²   ≈ 1.0 B
      main GEMM  :  d·n·p = 4096 · 500 · 4096 ≈ 8.4 B
      un-mix     :  another d·n² ≈ 1.0 B
      inverse    :  n³ = 500³  (0.125 B) once per A

   Mixing + un-mixing is ≈ 2 B FLOPs, about 24 % of the projection cost,
   and they run on the trusted CPU/GPU; the heavy 8.4 B FLOPs stay on the
   un-trusted device.

•  Privacy: the accelerator never sees a single h-row; every row of U is a
   dense ±-weighted combination of all 500 tokens.  Current blind source
   separation methods need far more than d=4096 independent samples to
   recover a 500×500 mixing matrix, so with *one fresh A per batch* the
   hidden states cannot be isolated.

Things to keep in mind
────────────────────────────────────────
1.  Use a new A (or at least re-sign / permute its columns) every batch; if
    you re-use the same dense A for many batches the attacker eventually
    gathers enough samples to invert it.

2.  Make A well conditioned (orthogonal or near-orthogonal) so that
    A^{-1} does not amplify numerical noise.

3.  If 24 % extra trusted-side FLOPs is still too much you can
       – split the 500 tokens into blocks of 128–256, apply an independent
         dense A per block (inverse is tiny, cost drops to ≈ 5–10 %), or
       – replace the dense A with a fast Hadamard-block; cost goes down
         further but privacy becomes “ICA-hard” rather than “ICA-infeasible”.



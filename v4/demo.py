"""
Demo of the “mix / un-mix” protocol with a dense orthogonal matrix A
--------------------------------------------------------------------
Setting
    n = 500          # number of tokens in the batch
    d = 4096         # hidden-state width  (Llama-2-7B/13B)
    p = 4096         # output width of Wq

Protocol
    U   =  A  @ H          (trusted side, mix)
    Y   =  U  @ W          (un-trusted accelerator)
    Q   =  A^{-1} @ Y      (trusted side, un-mix)

The code checks that Q == H @ W (within numerical precision).
"""

import numpy as np

# reproducibility
np.random.seed(0)

# --------------------------------------------
# 1. dimensions
# --------------------------------------------
n = 500          # tokens (rows)
d = 4096         # hidden-state width
p = 4096         # projection width (Wq)

# --------------------------------------------
# 2. make random H and W  (float32 to save RAM)
# --------------------------------------------
H = np.random.randn(n, d).astype(np.float32)
W = np.random.randn(d, p).astype(np.float32)

# --------------------------------------------
# 3. build a dense orthogonal A (500 × 500)
#    QR on a random gaussian matrix, keep Q
# --------------------------------------------
A_random  = np.random.randn(n, n).astype(np.float32)
Q, _      = np.linalg.qr(A_random)        # Q is orthonormal
A         = Q.astype(np.float32)          # rename for clarity
A_inv     = A.T                           # inverse of an orthogonal matrix

# --------------------------------------------
# 4. reference result on the trusted side
# --------------------------------------------
Q_ref = H @ W                   # shape (n, p)

# --------------------------------------------
# 5. protocol
# --------------------------------------------
# trusted side – mix
import pdb;pdb.set_trace()
U = A @ H                       # (500 × 4096)

# un-trusted accelerator – heavy GEMM
Y = U @ W                       # (500 × 4096)

# trusted side – un-mix
Q_rec = A_inv @ Y               # should equal Q_ref

# --------------------------------------------
# 6. verify correctness
# --------------------------------------------
abs_err   = np.max(np.abs(Q_rec - Q_ref))
rel_err   = abs_err / np.max(np.abs(Q_ref))

print(f"max absolute error : {abs_err:.3e}")
print(f"max relative error : {rel_err:.3e}")

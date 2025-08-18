import numpy as np
from sklearn.decomposition import FastICA
from scipy.spatial.distance import cdist

np.random.seed(0)

# ------------------------------------------------------------
# 1.  dimensions
# ------------------------------------------------------------
n = 100          # tokens in the batch  (also number of sources)
d = 4096         # width of hidden state
p = 4096         # output width of W (unused in the attack)

# ------------------------------------------------------------
# 2.  create ground-truth hidden states (H)  and orthogonal mix (A)
# ------------------------------------------------------------
H = np.random.randn(n, d).astype(np.float32)          # (500 × 4096)

A_rand = np.random.randn(n, n).astype(np.float32)
A, _   = np.linalg.qr(A_rand)                         # dense orthogonal (500 × 500)

# mixed data seen by the attacker
U = A @ H                                            # (500 × 4096)

# centre the data along the sample axis (as ICA expects)
U_c   = U - U.mean(axis=1, keepdims=True)

# samples for ICA must be (n_samples, n_features) so transpose
X = U_c.T                                            # (4096 samples × 500 features)

# ------------------------------------------------------------
# 3.  attempt Blind Source Separation with FastICA
# ------------------------------------------------------------
ica = FastICA(n_components=n, whiten="unit-variance", max_iter=2000, tol=1e-4)
S_est = ica.fit_transform(X)         # (4096 × 500)   estimated sources
H_est = S_est.T                      # reshape like H   (500 × 4096)

# ------------------------------------------------------------
# 4.  match each estimated column to its nearest true column
# ------------------------------------------------------------
# cosines between every pair  (500 × 500)
cos = 1 - cdist(H_est, H, metric="cosine")

best_match   = cos.argmax(axis=1)        # index of closest true column
best_cosines = cos[np.arange(n), best_match]

print(f"Best-match cosine statistics over {n} tokens")
print("  mean  :", best_cosines.mean())
print("  median:", np.median(best_cosines))
print("  90-th :", np.quantile(best_cosines, 0.9))
print("  max   :", best_cosines.max())

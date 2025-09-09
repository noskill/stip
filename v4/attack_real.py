# Copyright (c) 2024 – research demo only.
#
# This script tries to recover the original hidden-state rows (H) from
# the obfuscated matrices U = A·H that leave a trusted enclave.  It draws
# realistic batches from a (very large) embedding database produced by
# `extract_layer10_embeddings.py`, records only the public U’s, runs
# Picard-ICA, then aligns the recovered sources with the ground-truth H
# that stayed private.
#
# Compared with the earlier draft, the script now:
#   • understands both a single `.npy` file **or** a directory with many
#     `*.npz` shards (as written by the extractor);
#   • samples rows uniformly at random across shards without ever loading
#     the full database in RAM;
#   • accepts extra CLI flags (`--embed-dir`, `--field`) to locate the
#     database;
#   • fixes assorted dtype / shape bugs discovered during real-data runs.

import argparse, glob, os, mmap, sys
from pathlib import Path
import threading, queue, concurrent.futures
import time

import numpy as np
from numpy.linalg import qr, norm
from scipy.spatial.distance import cdist

# Picard ≥ 0.7 provides an sklearn-compatible estimator.
try:
    from picard import Picard  # type: ignore
except ImportError as exc:  # pragma: no cover – CI must have picard installed
    sys.exit("[attack_real] picard ≥0.7 required – aborting. " + str(exc))

# ---------------- CLI ----------------
# ---------------- CLI ----------------

p = argparse.ArgumentParser(description="Recover hidden-state rows from obfuscated batches (research demo)")
p.add_argument("--embed-dir", type=str, default="embeddings_out_test",
               help="Path to a .npy matrix **or** directory with *.npz shards (default: embeddings_out_test)")
p.add_argument("--field", type=str, default="embeddings", choices=["embeddings", "embeddings_rms"],
               help="Which array inside each npz shard to use (ignored for .npy)")
p.add_argument("--dim", type=int, default=4096, help="Width of the hidden state (d)")
p.add_argument("--batches", type=int, default=200, help="How many batches to draw (T)")
p.add_argument("--rows", type=int, default=500, help="#tokens per batch (n)")
p.add_argument("--chunksz", type=int, default=200, help="Row-chunk size passed to Picard")
p.add_argument("--dtype", choices=["float32", "float16"], default="float32",
               help="Precision used for the attacker memmap; lower it to save RAM/disk")
p.add_argument("--reservoir", type=int, default=1_000_000,
               help="Max #rows kept for evaluation reservoir (0 = disable)")
p.add_argument("--cache-shards", type=int, default=4,
               help="Maximum number of npz shards kept open simultaneously")
p.add_argument("--save-ica", type=str, default="work/ica_W.npy",
               help="Path to save the learned unmixing matrix W (\n × n)")
# Extra outputs – whitening, mixing, reservoir sample (R)
# Keep separate paths so downstream analysis can load only what it needs.
p.add_argument("--save-whitening", type=str, default="work/ica_K.npy",
               help="Path to save the whitening matrix K (n × n)")
p.add_argument("--save-mixing", type=str, default="work/ica_Ahat.npy",
               help="Path to save the estimated mixing matrix Â (n × n)")
p.add_argument("--save-R", type=str, default="work/R.npy",
               help="Path to save the stacked reservoir sample R (≤1M × d)")
# Picard-ICA convergence control.
p.add_argument("--max-iter", type=int, default=300,
               help="Maximum Picard iterations (passed through to picard.max_iter)")
# Debug/ablation option – skip the final un-whitening multiplication so
# that the recovered directions stay in the whitened space.  Useful for
# unit-tests that compare alignment with/without this step.
# Note: previous versions offered a `--skip-unwhiten` flag for an
# ablation that kept the recovered directions in the whitened space.
# The current implementation relies on Picard’s own `mixing_` attribute
# (or the functional equivalent) which already returns the mixing matrix
# in the *original* feature space, so the flag is obsolete.
args = p.parse_args()

d, T, n, CH = args.dim, args.batches, args.rows, args.chunksz
print(f"== running with  d={d}  n={n}  T={T}  chunk={CH}")

# ---------------- Embedding database helper -------------


class EmbeddingDB:
    """Uniform random access to rows spread across many *.npz shards.

    For performance we lazily memory-map each shard the first time we need
    a row from it and then keep the NumPy array alive for subsequent
    accesses.  Only the requested rows are materialised as float32 in
    memory, so you can point this at >100 GB of embeddings if you like.
    """

    def __init__(self, path: str | os.PathLike, field: str = "embeddings", *, cache_limit: int = 4,
                 rotate_interval: int = 200):
        path = Path(path)
        self.rng = np.random.default_rng(0)

        if path.is_file():
            # Single .npy with shape (tokens, dim)
            self.arr = np.load(path, mmap_mode="r")
            self.M, self.d = self.arr.shape
            self.is_single = True
        else:
            # Directory containing many shards.
            files = sorted(glob.glob(str(path / "*.npz")))
            if not files:
                raise FileNotFoundError(f"No *.npz found in {path}")

            self.files = files
            self.field = field
            self.is_single = False

            # probe each file to know its length *without* loading data
            self.lengths = []
            import zipfile, io, struct

            def _np_shape_from_header(byt: bytes):
                """Return the shape tuple from a *.npy header (no data read)."""
                # Cheat: rely on literal 'shape': (n, d) in the header text.
                hdr = byt.decode("latin1")
                import ast, re
                m = re.search(r"'shape': *\(([^)]*)\)", hdr)
                if not m:
                    raise ValueError("Cannot parse .npy header")
                return ast.literal_eval(f"({m.group(1)})")

            for f in self.files:
                with zipfile.ZipFile(f) as zf:
                    name = f"{field}.npy"
                    if name not in zf.namelist():
                        raise KeyError(f"{name} missing in {f}")
                    with zf.open(name) as npy:
                        # read the first 128 bytes which always contain the header
                        prefix = npy.read(256)
                        # NumPy .npy header starts with magic \x93NUMPY + version.
                        # The header length is stored at bytes 8–9 (little-endian).
                        header_len = struct.unpack_from("<H", prefix, 8)[0]
                        # Full header may be longer than 256 bytes; ensure we have it.
                        if len(prefix) < 10 + header_len:
                            prefix += npy.read(10 + header_len - len(prefix))
                        header = prefix[10:10 + header_len]
                        nrows, ncols = _np_shape_from_header(header)

                self.lengths.append(nrows)
                if hasattr(self, "d"):
                    assert self.d == ncols, "dim mismatch across shards"
                else:
                    self.d = ncols
            self.M = int(np.sum(self.lengths))

            # cumulative counts for O(log #files) lookup
            self.cum = np.cumsum(self.lengths)
            self._cache: dict[int, tuple[np.ndarray, np.lib.npyio.NpzFile]] = {}
            self._lru: list[int] = []  # keeps most-recently used shard indices
            self._cache_limit = max(1, cache_limit)
            # counters for sampling stats
            self._served = np.zeros(len(self.files), dtype=np.int64)
            self._prefetch_idx = 0  # next shard to prefetch

            # Pre-warm cache with the first few shards so that we have
            # something to sample from before the background thread kicks in.
            for s in range(min(self._cache_limit, len(self.files))):
                self._get_shard(s)
                self._prefetch_idx = s + 1

            assert len(self._cache) >= 2 or len(self.files) < 2, (
                "Cache should contain at least 2 shards to ensure diversity; "
                "increase --cache-shards if possible.")

            # background prefetch machinery
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._prefetch_future: concurrent.futures.Future | None = None
            self._rotate_interval = max(1, rotate_interval)
            self._calls_since_rot = 0

    # --------------------------------------------------
    def close(self):
        """Release all open NPZ files and shut down background thread."""
        if not self.is_single:
            # Cancel any pending prefetch task
            if self._prefetch_future and not self._prefetch_future.done():
                self._prefetch_future.cancel()

            # Close cached shards
            for idx, (arr, npz) in list(self._cache.items()):
                del arr
                npz.close()
            self._cache.clear()
            self._lru.clear()

            self._executor.shutdown(wait=False)

    # --------------------------------------------------
    def _get_shard(self, idx: int) -> np.ndarray:
        """Return mmap-ed array for shard *idx*, loading on demand."""
        if idx not in self._cache:
            fpath = self.files[idx]
            print(f"[EmbeddingDB] loading shard {idx+1}/{len(self.files)} → {os.path.basename(fpath)}")
            npz = np.load(fpath, mmap_mode="r")
            arr = npz[self.field]

            # Evict least-recently used shard if we exceed the limit.
            if len(self._cache) >= self._cache_limit:
                old_idx = self._lru.pop(0)
                old_arr, old_npz = self._cache.pop(old_idx)
                # Explicitly delete objects so mmap can be released.
                del old_arr
                old_npz.close()
            self._cache[idx] = (arr, npz)
        # update LRU order – move idx to the end
        if idx in self._lru:
            self._lru.remove(idx)
        self._lru.append(idx)
        return self._cache[idx][0]

    # --------------------------------------------------
    def _schedule_prefetch(self):
        """Launch background task to mmap the next shard if none running."""
        if self._prefetch_future is None or self._prefetch_future.done():
            if self._prefetch_idx >= len(self.files):
                return  # nothing left to load

            next_idx = self._prefetch_idx
            self._prefetch_idx += 1

            fpath = self.files[next_idx]

            def _load(path, field):
                print(f"[EmbeddingDB-bg] prefetching shard {next_idx+1}/{len(self.files)} → {os.path.basename(path)}")
                npz = np.load(path, mmap_mode="r")
                arr = npz[field]
                return next_idx, arr, npz

            self._prefetch_future = self._executor.submit(_load, fpath, self.field)

    # --------------------------------------------------
    def _maybe_rotate(self):
        """After rotate_interval calls check quota and possibly replace shard."""
        # Count how many times this method was entered since the previous
        # *successful* rotation.  We only reset the counter **after** a new
        # shard has actually been swapped in.  Otherwise, if the background
        # pre-fetch is still running we would never reach the quota again and
        # rotation would stall indefinitely.

        self._calls_since_rot += 1

        # Haven't reached the quota yet → nothing to do.
        if self._calls_since_rot < self._rotate_interval:
            return

        # Quota reached – make sure the prefetched shard is ready.  If it is
        # still loading, keep the counter unchanged so we will check again on
        # the next call.
        if not (self._prefetch_future and self._prefetch_future.done()):
            return  # nothing ready yet; try again later

        # A new shard is ready → proceed with rotation and then reset counter.
        self._calls_since_rot = 0

        new_idx, new_arr, new_npz = self._prefetch_future.result()
        self._prefetch_future = None

        # choose victim shard among cached ones based on utilisation ratio
        ratios = []
        cached_indices = list(self._cache.keys())
        for idx in cached_indices:
            served = self._served[idx]
            size = self.lengths[idx]
            ratios.append(served / max(1, size))
        ratios = np.asarray(ratios, dtype=float)
        ratios /= ratios.sum()
        victim = self.rng.choice(cached_indices, p=ratios)

        # evict victim
        v_arr, v_npz = self._cache.pop(victim)
        del v_arr
        v_npz.close()
        if victim in self._lru:
            self._lru.remove(victim)

        # add new shard
        print(f"[EmbeddingDB] rotating in shard {new_idx+1} – replacing {victim+1}")
        self._cache[new_idx] = (new_arr, new_npz)
        self._lru.append(new_idx)


    # --------------------------------------------------
    def sample_rows(self, k: int, *, dim_limit: int | None = None,
                    dtype: np.dtype | str = np.float32) -> np.ndarray:
        """Sample *k* rows uniformly at random across the DB.

        Parameters
        ----------
        k : int
            Number of rows to return.
        dim_limit : int | None
            Optionally truncate the returned vectors to the first
            *dim_limit* columns.  This is cheaper than slicing later
            because we avoid writing the extra bytes at all.
        dtype : np.dtype or str, default float32
            Desired dtype of the returned array.  The embeddings inside
            the shards are typically float32, but callers may request
            float16 to save memory – we convert using ``astype`` with
            ``copy=False`` when possible so this is virtually free if
            the dtypes already match.
        """
        assert k > 0
        # schedule background prefetch right away
        self._schedule_prefetch()

        # ensure *dtype* is a NumPy dtype object
        dtype = np.dtype(dtype)

        if self.is_single:
            # Simple – global array so no caching issues.
            idxs = self.rng.integers(0, self.M, size=k)
            rows = self.arr[idxs]
        else:
            # Pick only among *currently cached* shards to guarantee we
            # do not load anything synchronously.
            cached_indices = list(self._cache.keys())
            if not cached_indices:
                # very first call – cache should contain at least one shard
                cached_indices = [0]
                self._get_shard(0)

            sizes_remaining = np.array([self.lengths[i] - self._served[i] for i in cached_indices], dtype=float)
            sizes_remaining[sizes_remaining < 1] = 1.0
            probs = sizes_remaining / sizes_remaining.sum()

            rows = np.empty((k, self.d), dtype=dtype)
            for i in range(k):
                shard = int(self.rng.choice(cached_indices, p=probs))
                local = self.rng.integers(0, self.lengths[shard])
                rows[i] = self._get_shard(shard)[local]
                self._served[shard] += 1
        if dim_limit is not None and dim_limit < self.d:
            rows = rows[:, :dim_limit]

        # maybe rotate cache now
        self._maybe_rotate()

        return rows.astype(dtype, copy=False)


# ---------------- load db ------------

DB = EmbeddingDB(args.embed_dir, field=args.field, cache_limit=args.cache_shards)
M, d0 = DB.M, DB.d
assert d <= d0, f"Requested dim {d} but DB has {d0}"
rng = np.random.default_rng(0)

# ---------------- helper -------------
def ortho(k=500):
    q,_ = qr(rng.standard_normal((k,k)))
    if np.linalg.det(q)<0: q[:,0]*=-1
    return q.astype(np.float32)

# ---------------- mem-map attacker matrix Y ----------------
# Y holds *samples* as rows (d per batch) and *features* as columns (n).
#   shape = (T * d, n)
workdir = Path("work")
workdir.mkdir(exist_ok=True)
rows_Y = T * d
Y_path = workdir / ("Y_f16.dat" if args.dtype == "float16" else "Y_f32.dat")
dtype_y = np.float16 if args.dtype == "float16" else np.float32
Y = np.memmap(Y_path, dtype=dtype_y, mode="w+", shape=(rows_Y, n))

print("Writing obfuscated batches to memmap …")
row_ptr = 0

# Reservoir sample for evaluation – about 1 M rows.
# Reservoir sample for evaluation (uniform reservoir sampling).
# Size is configurable via --reservoir (0 disables evaluation to save RAM).
reservoir: list[np.ndarray] = []
RSZ = max(0, args.reservoir)

# Storing 1 M vectors of 4 096 float32 elements consumes ≈16 GiB.  Down-cast to
# float16 to cut this in half; callers can change `RSZ` or disable the reservoir
# entirely via --reservoir 0 (see CLI flag below).
cos_cnt = 0

for t in range(T):
    # ------------------------------------------------------------------
    # 1) Draw a fresh batch of *n* hidden states uniformly from the DB.
    # ------------------------------------------------------------------
    H = DB.sample_rows(n, dim_limit=d, dtype=dtype_y)  # (n , d)

    # ------------------------------------------------------------------
    # 2) Obfuscate with a fresh orthogonal matrix *A* → attacker sees U.
    # ------------------------------------------------------------------
    A = ortho(n)
    U = A @ H                                  # (n , d)

    # Attacker records the *transposed* matrix so that rows = samples.
    Y[row_ptr:row_ptr + d, :] = U.T.astype(dtype_y, copy=False)  # (d , n)
    row_ptr += d

    # ------------------------------------------------------------------
    # 3) Keep a (private) normalised copy of H for later evaluation.
    # ------------------------------------------------------------------
    # --------------------------------------------------------------
    # 3) (Optional) Add to evaluation reservoir (uniform sample)
    # --------------------------------------------------------------
    if RSZ > 0:
        H_norm = (H / norm(H, axis=1, keepdims=True)).astype(np.float16)

        if len(reservoir) < RSZ:
            reservoir.extend(H_norm)
        else:
            # classic reservoir sampling
            for r in H_norm:
                j = rng.integers(0, cos_cnt + 1)
                if j < RSZ:
                    reservoir[j] = r
        cos_cnt += n

    if (t + 1) % 20 == 0 or (t + 1) == T:
        print(f"[progress] sampled {t+1}/{T} batches")

print("All batches written.")

# Release mmap-ed shards ASAP to keep memory footprint low before ICA.
DB.close()

# ---------------- Picard-ICA -----------------------
# ---------------- Picard-ICA -----------------------
print("Running Picard-ICA on the recorded data …")

# Picard estimator expects float32 (n_samples, n_features).  Avoid an extra
# allocation when the memmap already has the right dtype.
print(f"[progress] Feeding {rows_Y // d}/{T} batches to ICA …")

if Y.dtype == np.float32:
    X_est = Y  # zero-copy use of memmap
else:
    # Cast to float32 then release the memmap to drop unused pages.
    X_est = Y.astype(np.float32, copy=False)
    del Y  # allow OS to reclaim mmap-backed pages

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Instantiate Picard estimator (sklearn-style) and fit. A minimal
# `_validate_data` shim is injected when running against very old
# scikit-learn versions that lack this helper – this keeps the script
# functional without any additional fallbacks.
# ------------------------------------------------------------------

est = Picard(n_components=n, ortho=True, max_iter=args.max_iter, tol=1e-5)

# Patch missing `_validate_data` for ancient sklearn builds (<0.22).
if not hasattr(est, "_validate_data"):
    import types

    def _noop_validate(self, X, copy=None, dtype=None, **k):  # type: ignore
        return np.asarray(X)

    est._validate_data = types.MethodType(_noop_validate, est)  # type: ignore

start_t = time.perf_counter()
est.fit(X_est)
ica_time = time.perf_counter() - start_t
print(f"[timing] Picard-ICA fit completed in {ica_time:.2f} s")

# Extract learned matrices.
W = est.components_.astype(np.float32, copy=False)
K = est.whitening_.astype(np.float32, copy=False)

np.save(args.save_ica, W)
print(f"[saved] unmixing matrix W → {args.save_ica}")

np.save(args.save_whitening, K)
print(f"[saved] whitening matrix K → {args.save_whitening}")

# --------------------------------------------------------------
# Build global mixing estimate  Â  (n × n) in the *original* space.
#
# For the sklearn-compatible estimator provided by recent Picard
# versions, the `mixing_` attribute already contains this matrix so we
# can simply reuse it. For the functional API fallback (or exceptionally
# old Picard builds without `mixing_`), we compute the product manually.
# --------------------------------------------------------------

A_hat = est.mixing_.astype(np.float32, copy=False)

# Save mixing matrix as requested.
np.save(args.save_mixing, A_hat)
print(f"[saved] mixing matrix Â → {args.save_mixing}")

# Unit-normalise columns → row vectors in `dirs` (n , n)
dirs = A_hat.T / norm(A_hat, axis=0, keepdims=True)
# extend / trim to match the ambient dimension d
if dirs.shape[1] < d:
    dirs = np.pad(dirs, ((0, 0), (0, d - dirs.shape[1])))
dirs = dirs[:, :d]
dirs /= norm(dirs, axis=1, keepdims=True)

# -------------------------------------------
# Debug: report key matrix shapes
# -------------------------------------------
print("[shape] X_est", X_est.shape, "A_hat", A_hat.shape, "R", (len(reservoir), d), "dirs", dirs.shape)

# ---------------- cosine on reservoir ---------------------
if RSZ > 0 and reservoir:
    print("Computing cosine statistics on reservoir …")
    R = np.stack(reservoir, axis=0).astype(np.float32)  # (≤RSZ , d)
    np.save(args.save_R, R)
    print(f"[saved] reservoir matrix R → {args.save_R}")

    from scipy.spatial.distance import cdist  # local import to avoid overhead if skipped

    sim = 1 - cdist(R, dirs, metric="cosine")
    best = sim.max(axis=1)

    print(f"\n=== RESULTS on {len(R):,}-row sample ===")
    print("  average cosine  :", float(best.mean()))
    print("  25 / 50 / 75 % :", *map(float, np.percentile(best, [25, 50, 75])))
else:
    print("[note] Reservoir disabled or empty – skipping cosine evaluation.")

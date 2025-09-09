"""Efficient excess-kurtosis test on saved embeddings.

Loads a **single pass** over each `.npz` file (at most once) and draws a
uniform random sample of vectors, then reports max / median excess kurtosis
over random projection directions.

Example (raw vs RMS-norm):
    python kurtosis_test.py --emb-dir ./embeddings_out --field embeddings_rms -M 50000 -K 1024
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Sampling helper (two-pass, file-wise loading ≤1 time)
# ---------------------------------------------------------------------------


def uniform_sample(dir_path: pathlib.Path, field: str, M: int, rng: np.random.Generator) -> np.ndarray:
    """Return `M` rows sampled uniformly across all NPZ files in *dir_path*.

    Strategy:
    1. Cheap metadata pass: read shapes with `mmap_mode='r'` → counts.
    2. Choose `M` distinct global indices without replacement.
    3. Second pass: load only files that contain at least one selected index
       (each at most once) and gather those rows.
    This guarantees uniform sampling while eliminating repeated loads.
    """

    files, counts, total = [], [], 0
    # First lightweight metadata pass (no verbose output)
    for fp in sorted(dir_path.glob("*.npz")):
        with np.load(fp, mmap_mode="r") as data:
            arr_name = field if field in data else ("embeddings" if field == "embeddings_rms" else None)
            if arr_name is None or arr_name not in data:
                continue  # skip file without requested array
            n = data[arr_name].shape[0]
            files.append((fp, arr_name))
            counts.append(n)
            total += n

    if total < M:
        raise ValueError(f"Only {total} rows available but sample_size={M}")

    # Select global indices and sort for sequential access
    global_idx = rng.choice(total, size=M, replace=False)
    global_idx.sort()

    sample = np.empty((M, 4096), dtype=np.float32)
    ptr = 0
    start = 0
    print(files)
    for (fp, arr_name), cnt in zip(files, counts):
        end = start + cnt
        mask = (global_idx >= start) & (global_idx < end)
        if not mask.any():
            start = end
            continue

        local = global_idx[mask] - start  # positions inside this file
        with np.load(fp) as data:
            # verbose load message for actual heavy load
            print(f"loading {fp.name}")
            arr = data[arr_name].astype(np.float32)
            sample[ptr : ptr + len(local)] = arr[local]
            ptr += len(local)

        start = end

    return sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Excess kurtosis on embedding sample")
    p.add_argument("--emb-dir", required=True, type=pathlib.Path, help="Directory with *.npz files")
    p.add_argument("--field", choices=["embeddings", "embeddings_rms"], default="embeddings")
    p.add_argument("-M", "--sample-size", type=int, default=50000, help="# vectors to sample")
    p.add_argument("-K", "--directions", type=int, default=1024, help="# random projection directions")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Sampling {args.sample_size} rows of '{args.field}' …")
    H = uniform_sample(args.emb_dir, args.field, args.sample_size, rng)

    # Standardise each dimension
    H -= H.mean(axis=0, keepdims=True)
    H /= H.std(axis=0, keepdims=True) + 1e-8

    # Random directions
    dirs = rng.normal(size=(args.directions, H.shape[1])).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    proj = H @ dirs.T  # (M, K)
    kurt = (proj ** 4).mean(0) / (proj ** 2).mean(0) ** 2
    excess = kurt - 3.0

    print(f"max excess kurtosis:    {excess.max():.4f}")
    print(f"median excess kurtosis: {np.median(excess):.4f}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(
            "Example: python kurtosis_test.py --emb-dir ./embeddings_out --field embeddings_rms -M 50000 -K 1024",
            file=sys.stderr,
        )
    main()

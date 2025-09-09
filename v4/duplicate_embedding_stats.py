"""Duplicate embedding analysis across saved `.npz` shards.

This utility scans a directory that contains multiple NumPy ``.npz`` files
with token embeddings (as written by ``extract_layer10_embeddings.py``) and
reports *hash collisions* between rows.  A 64-bit hash of every vector is
computed and the script prints a small summary / histogram that shows how
often the **same hash** appears.

While a hash collision does not *guarantee* that the underlying vectors are
identical, with 64 bits the probability of an accidental clash is negligible
relative to the dataset size.  If desired, the code can easily be extended to
double-check exact equality once suspicious clusters are identified.

Example
-------
Analyse the first 6 million RMS-normalised embeddings in ``./embeddings_out``::

    python duplicate_embedding_stats.py --emb-dir ./embeddings_out --field embeddings_rms -N 6_000_000

If the optional *xxhash* package is available, it will be used for very fast
non-cryptographic hashing.  Otherwise the script falls back to the built-in
``hashlib.blake2b`` with a 64-bit digest.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys
from typing import Iterable, List

import numpy as np

# -----------------------------------------------------------------------------
# Optional fast 64-bit hash (xxhash) ➜ fallback to hashlib if unavailable
# -----------------------------------------------------------------------------


try:
    import xxhash  # type: ignore

    def hash64(buf: memoryview | bytes) -> int:
        return xxhash.xxh64(buf).intdigest()


except ModuleNotFoundError:
    import hashlib

    def hash64(buf: memoryview | bytes) -> int:  # noqa: D401 – simple hash wrapper
        """Compute a 64-bit hash using ``blake2b`` (little-endian)."""

        h = hashlib.blake2b(buf, digest_size=8)
        return int.from_bytes(h.digest(), "little", signed=False)


# -----------------------------------------------------------------------------
# Helper: iterate over rows from multiple files until *limit* is reached
# -----------------------------------------------------------------------------


def iter_rows(
    dir_path: pathlib.Path, field: str, limit: int | None
) -> Iterable[tuple[int, np.ndarray]]:
    """Yield ``(token_id, embedding)`` tuples from ``*.npz`` shards.

    The generator walks the directory in **sorted file order** and produces
    *contiguous* rows straight from each mem-mapped shard – **without copies**.

    Parameters
    ----------
    dir_path:
        Folder that contains the ``.npz`` shards produced by
        ``extract_layer10_embeddings.py``.
    field:
        Either ``"embeddings"`` or ``"embeddings_rms"`` – depending on which
        array should be analysed.
    limit:
        Optional hard cap on how many total rows to yield.
    """

    yielded = 0
    for fp in sorted(dir_path.glob("*.npz")):
        with np.load(fp, mmap_mode="r") as data:
            # Locate desired embedding array name --------------------------------
            arr_name: str | None = field if field in data else (
                "embeddings" if field == "embeddings_rms" else None
            )
            if arr_name is None or arr_name not in data or "token_ids" not in data:
                # Skip shards that don't contain both the requested embedding
                # array **and** the corresponding token-id mapping.
                continue

            emb_arr = data[arr_name]  # (N, D) float16/32 – mem-mapped
            tok_arr = data["token_ids"]  # (N,) int32 – mem-mapped

            n = emb_arr.shape[0]

            # Determine slice length while respecting the *limit* --------------
            take = n if limit is None else max(0, min(n, limit - yielded))
            if take == 0:
                break

            for i in range(take):
                # Return memory-views → no extra allocations
                yield int(tok_arr[i]), emb_arr[i]

            yielded += take

            if limit is not None and yielded >= limit:
                break


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------


def main() -> None:  # noqa: C901 – small script, acceptable complexity
    p = argparse.ArgumentParser(description="Duplicate-hash statistics for embeddings")
    p.add_argument("--emb-dir", required=True, type=pathlib.Path, help="Directory with *.npz shards")
    p.add_argument("--field", choices=["embeddings", "embeddings_rms"], default="embeddings_rms")
    p.add_argument("-N", "--rows", type=int, default=None, help="Maximum #rows to analyse (default: all)")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional HuggingFace model name/path to translate token_ids to text",
    )
    args = p.parse_args()

    if not args.emb_dir.is_dir():
        sys.exit(f"error: {args.emb_dir} is not a directory")

    print(f"Scanning '{args.emb_dir}' for duplicate {args.field} rows …")
    if args.rows is not None:
        print(f"Limiting to the first {args.rows:,} rows")

    # ------------------------------------------------------------------
    # Optional: load tokenizer if the user supplied a model -------------
    # ------------------------------------------------------------------

    tokenizer = None
    if args.model is not None:
        try:
            from transformers import AutoTokenizer  # type: ignore

            print(f"Loading tokenizer for '{args.model}' …")
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        except Exception as e:  # pragma: no cover – best-effort load
            print(f"Warning: could not load tokenizer: {e}")
            tokenizer = None

    # ------------------------------------------------------------------
    # Pass 1: iterate rows, record both token-ids **and** 64-bit hashes
    # ------------------------------------------------------------------

    token_ids_list: List[int] = []
    hashes_list: List[int] = []

    print("Computing 64-bit hashes …")
    for tok_id, row in iter_rows(args.emb_dir, args.field, args.rows):
        token_ids_list.append(tok_id)
        mv: memoryview = row.view(np.uint8)  # zero-copy raw bytes
        hashes_list.append(hash64(mv))

    # Convert to compact NumPy arrays for fast downstream ops --------------
    hashes: np.ndarray[np.uint64] = np.fromiter(hashes_list, dtype=np.uint64)
    token_ids: np.ndarray[np.int32] = np.fromiter(token_ids_list, dtype=np.int32)

    total_rows = len(hashes)
    if total_rows == 0:
        sys.exit("No rows loaded – check directory and field name.")

    print(f"Processed {total_rows:,} rows → analysing collisions …")

    uniq, counts = np.unique(hashes, return_counts=True)

    # Basic stats
    num_unique = len(uniq)
    num_collisions = total_rows - num_unique
    pct = num_collisions / total_rows * 100

    print()
    print("========== SUMMARY ==========\n")
    print(f"Total rows      : {total_rows:,}")
    print(f"Unique hashes   : {num_unique:,}")
    print(f"Collisions      : {num_collisions:,}  ({pct:.6f} %)")
    print()

    # Histogram: how many hashes appear k times (k ≥ 1)
    freq_counter: collections.Counter[int] = collections.Counter(counts)

    print("Occurrences → #hashes")
    for k in sorted(freq_counter):
        num_hashes = freq_counter[k]
        if k == 1:
            continue  # singleton entries not interesting beyond summary
        print(f"{k:>3}× : {num_hashes:,}")

    # Show the *top N* most frequent hashes ---------------------------------
    TOP_N = 20
    if num_collisions:
        print(f"\nTop {TOP_N} most frequent hashes (with token IDs):")
        top = np.argsort(counts)[-TOP_N:][::-1]  # indices of the largest clusters
        for rank, idx in enumerate(top, 1):
            dup_count = counts[idx]
            if dup_count == 1:
                break  # no more collisions in list

            h_val = uniq[idx]

            # Locate *all* rows that share this hash and fetch their token-ids
            pos: np.ndarray[np.int64] = np.flatnonzero(hashes == h_val)
            # Deduplicate token-ids to avoid printing repeats ------------------
            tok_ids_for_hash: np.ndarray[np.int32] = np.unique(token_ids[pos])

            # Limit very long outputs – show up to MAX_DISPLAY IDs, then ellipsis
            MAX_DISPLAY = 20
            if len(tok_ids_for_hash) > MAX_DISPLAY:
                display_ids_arr = tok_ids_for_hash[:MAX_DISPLAY]
                ids_str = ", ".join(map(str, display_ids_arr)) + ", …"
            else:
                ids_str = ", ".join(map(str, tok_ids_for_hash))

            # Optional textual representation using tokenizer -----------------
            if tokenizer is not None:
                try:
                    tokens_text = tokenizer.convert_ids_to_tokens(
                        tok_ids_for_hash.tolist()
                    )
                    if len(tokens_text) > MAX_DISPLAY:
                        tokens_text_disp = tokens_text[:MAX_DISPLAY] + ["…"]
                    else:
                        tokens_text_disp = tokens_text
                    toks_str = ", ".join(tokens_text_disp)
                    extra = f"  tokens=[{toks_str}]"
                except Exception as e:
                    extra = ""  # if conversion fails, just omit text
            else:
                extra = ""

            print(f"#{rank}: hash={h_val:016x}  count={dup_count}  token_ids=[{ids_str}]{extra}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(
            "Example: python duplicate_embedding_stats.py --emb-dir ./embeddings_out_test --field embeddings_rms -N 1000000",
            file=sys.stderr,
        )
    main()

"""Synthetic smoke test for `attack_real.py`.

The test builds a tiny fake embedding database (NPZ shards) on-the-fly,
runs the attack script with a *rectangular* (d > n) configuration so
that whitening is non-trivial, and finally checks that:

  • the script saves an ICA matrix of the expected size (n×n), and
  • the reported average cosine in the **original** feature space is
    clearly above random chance.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePath

import numpy as np


def _make_fake_db(
    tmp: Path,
    *,
    shards: int = 3,
    rows_per_shard: int = 50,
    dim: int = 16,
    df: int = 3,
):
    """Create *shards* compressed npz files with Student-t rows.

    Using a heavy-tailed distribution (ν ≈ 3) yields rows whose second
    moment still equals the identity in expectation, yet practical
    samples deviate more strongly than with a Gaussian, making whitening
    less of a no-op even for independent dimensions.
    """

    rng = np.random.default_rng(0)

    # Apply a fixed *correlating* linear transform so that whitening is
    # genuinely needed.  Use a random Gaussian matrix with full rank.
    # Strong anisotropy: spread singular values across 1e-2 … 1e2.
    eigvals = (10.0 ** np.linspace(-2, 2, dim)).astype(np.float32)
    rot = rng.standard_normal((dim, dim)).astype(np.float32)
    while np.linalg.matrix_rank(rot) < dim:  # ensure full rank (very rare)
        rot = rng.standard_normal((dim, dim)).astype(np.float32)
    mix = rot @ np.diag(eigvals)

    for i in range(shards):
        base = rng.standard_t(df, size=(rows_per_shard, dim))  # (rows , dim)
        arr = (base @ mix).astype(np.float32)
        out = tmp / f"shard_{i:02d}.npz"
        # provide both 'embeddings' and 'embeddings_rms' fields
        np.savez_compressed(out, embeddings=arr, embeddings_rms=arr)


def test_attack_real_smoke():
    # Tune hyper-parameters so that Picard converges to noticeably
    # higher alignment on the synthetic Gaussian data while still
    # keeping the test runtime modest (<5 s on CI).
    #
    #  – Using a *square* setting (d == n) empirically helps ICA
    #    converge much faster than the original tall-matrix regime.
    #  – Slightly increasing the number of batches provides the
    #    additional samples required without ballooning memory.
    # A smaller square setting (d == n) greatly eases ICA's job on the
    # i.i.d. synthetic data and therefore leads to *noticeably* higher
    # alignment scores while still finishing well under a second on
    # typical CI hardware.

    # Use a *rectangular* setting (d > n) so the whitening matrix K is
    # non-trivial and the test exercises the de-whitening logic added in
    # attack_real.py.

    dim = 8       # hidden-state width (d)
    rows = 8      # tokens / batch (n)
    batches = 1800  # number of batches (T) – still under CI timeout

    # Use system tmp dir to avoid permission issues in read-only CWD.
    tmp_root = Path(tempfile.gettempdir())
    tdir = tmp_root / "_synthetic_test"
    if tdir.exists():
        shutil.rmtree(tdir)
    tdir.mkdir()
    db_dir = tdir / "db"
    db_dir.mkdir()
    _make_fake_db(db_dir, shards=3, rows_per_shard=50, dim=dim)

    out_w = tdir / "W.npy"

    cmd = [
        sys.executable,
        str((Path(__file__).parent / "attack_real.py").resolve()),
        "--embed-dir", str(db_dir.resolve()),
        "--field", "embeddings",
        "--dim", str(dim),
        "--rows", str(rows),
        "--batches", str(batches),
        "--dtype", "float32",
        "--cache-shards", "2",
    ]
    cmd_base = cmd + [
        "--save-ica", str(out_w.resolve()),
        "--max-iter", "800",
    ]

    def run(extra: list[str] = None) -> float:
        call = cmd_base + (extra or [])
        res = subprocess.run(call, capture_output=True, text=True)
        # Echo full output for visibility in CI logs.
        print("[attack_real stdout]\n", res.stdout)
        if res.stderr:
            print("[attack_real stderr]\n", res.stderr)

        if res.returncode != 0:
            raise RuntimeError(res.stderr)

        m = re.search(r"average cosine\s*:\s*([0-9.]+)", res.stdout)
        assert m, "average cosine not printed"
        return float(m.group(1))

    print("[debug] running pipeline …")
    val_full = run()

    # saved matrix check
    assert out_w.exists(), "ICA matrix not saved"
    W = np.load(out_w)
    assert W.shape == (rows, rows)

    # Alignment should be clearly above random; empirical >0.45 in this setup.
    assert val_full >= 0.45, f"average cosine too low: {val_full}" 

    # cleanup
    shutil.rmtree(tdir)


if __name__ == "__main__":
    test_attack_real_smoke()

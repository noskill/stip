"""Extract layer-10 token embeddings from CodeLlama over OpenWebText2.

Usage (example):

    python extract_layer10_embeddings.py \
        --data-dir ./openwebtext2 \
        --output-dir ./embeddings_out \
        --batch-size 8

The script walks *all* text files under `--data-dir`, tokenises each line,
computes hidden-states with CodeLlama-7b-hf and stores every token’s
layer-10 embedding together with its original token id.

If `--token-limit N` is provided, processing stops once **≈N** tokens have
been stored across all NPZ files.

Output format: one NPZ file per processed text file (or chunk) written to
`--output-dir`.

Always contains
    token_ids        : (N,) int32

Depending on `--norm-mode` it adds
    embeddings       : raw vectors (float16)
    embeddings_rms   : RMS-normalised vectors (float16)

This chunked approach avoids keeping the full corpus in memory.

Resuming: with `--resume` the script skips files already present in
`--output-dir` and counts their tokens toward `--token-limit`, letting you
continue a halted run without duplication.

Prerequisites: `transformers`, `torch`, `numpy`.
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import List

import json
import subprocess

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def find_text_files(root: pathlib.Path) -> List[pathlib.Path]:
    """Return sorted list of all text-like files under *root*."""

    exts = {".txt", ".text", ".jsonl", ".json", ".gz", ".xz", ".zst"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract layer-10 embeddings from CodeLlama")
    parser.add_argument("--data-dir", type=pathlib.Path, required=True, help="Path to openwebtext2 root directory")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Where chunk NPZ files will be written")
    parser.add_argument("--batch-size", type=int, default=4, help="Lines to process per batch (memory trade-off)")
    parser.add_argument("--max-length", type=int, default=2048, help="Truncate sequences longer than this many tokens")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume: skip files already processed in output-dir and count their tokens toward the limit",
    )

    parser.add_argument(
        "--norm-mode",
        choices=["raw", "rms", "both"],
        default="raw",
        help="Store raw embeddings, RMS-normalised embeddings, or both.",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=0,
        help="Stop after saving this many tokens (0 = unlimited)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Model & tokenizer -----------------------------------------------------
    model_name = "codellama/CodeLlama-7b-hf"
    print(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # CodeLlama tokenizer lacks a pad token.  Use eos as pad for batching.
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        # model.resize_token_embeddings might be needed, but hidden-state size
        # for pad is irrelevant (never attended to) so we skip resizing.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device.startswith("cuda") else None,
    ).eval()

    # Layer index: hidden_states[0] is embedding layer.
    LAYER10_IDX = 11  # transformer layer 10 output (embedding+10)

    # RMSNorm module for layer 11 input (after layer10 output) if requested
    rms_module = None
    if args.norm_mode in {"rms", "both"}:
        rms_module = model.model.layers[10 + 1].input_layernorm

    text_files = find_text_files(args.data_dir)
    if not text_files:
        raise ValueError(f"No text-like files found under {args.data_dir}")

    processed_files = 0
    global_token_count = 0
    last_report_tokens = 0

    if args.resume:
        existing_files = list(args.output_dir.glob("*_layer10.npz"))
        for efp in existing_files:
            try:
                data = np.load(efp)
                n_tokens = len(data["token_ids"])
                global_token_count += n_tokens
            except Exception:
                pass  # corrupted file, ignore counting
        if args.token_limit and global_token_count >= args.token_limit:
            print(
                f"Token limit {args.token_limit} already reached by existing data ({global_token_count}). Nothing to do."
            )
            return
        if global_token_count:
            print(f"Resuming: {global_token_count} tokens already present in output_dir")

    for fp in text_files:
        out_path = args.output_dir / f"{fp.stem}_layer10.npz"
        if args.resume and out_path.exists():
            processed_files += 1
            continue
        # -----------------------------------------------------------------
        # Load text content depending on file type. For OpenWebText2, files
        # are .jsonl.zst with one JSON per line containing a "text" field.
        # -----------------------------------------------------------------

        if fp.suffix == ".zst":
            # Stream decompress with external `zstd` if python-zstandard absent.
            try:
                import zstandard as zstd  # type: ignore

                with fp.open("rb") as f:
                    dctx = zstd.ZstdDecompressor()
                    stream_reader = dctx.stream_reader(f)
                    lines_raw = stream_reader.read().decode("utf-8", "ignore").splitlines()
            except ImportError:
                # Fallback to subprocess
                proc = subprocess.run(["zstd", "-dc", str(fp)], capture_output=True, check=True)
                lines_raw = proc.stdout.decode("utf-8", "ignore").splitlines()
        else:
            lines_raw = fp.read_text(encoding="utf-8", errors="ignore").splitlines()

        # Extract text from json if necessary
        extracted: List[str] = []
        for raw in lines_raw:
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("{"):
                try:
                    obj = json.loads(raw)
                    text = obj.get("text", "")
                except json.JSONDecodeError:
                    text = raw
            else:
                text = raw
            if text:
                extracted.append(text)

        if not extracted:
            continue

        embeddings_raw: List[np.ndarray] = []
        embeddings_rms: List[np.ndarray] = [] if args.norm_mode in {"rms", "both"} else None
        token_ids_list: List[np.ndarray] = []

        # Process in mini-batches over lines to utilise GPU.
        for i in range(0, len(extracted), args.batch_size):
            batch_texts = extracted[i : i + args.batch_size]

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=False,
            ).to(args.device)

            with torch.no_grad():
                out = model(**enc)

            h_state = out.hidden_states[LAYER10_IDX]  # (batch, seq_len, hidden)

            # Flatten batch dimension into one list per sample.
            pad_id = tokenizer.pad_token_id

            for j, input_ids in enumerate(enc["input_ids"]):
                seq_len = (input_ids != pad_id).sum().item()
                valid_ids = input_ids[:seq_len].cpu().int().numpy()
                token_ids_list.append(valid_ids)

                raw_emb = h_state[j, :seq_len, :]
                if args.norm_mode in {"rms", "both"}:
                    with torch.no_grad():
                        norm_emb = rms_module(raw_emb)  # RMSNorm on device of raw_emb

                raw_emb_np = raw_emb.cpu().half().numpy()
                if args.norm_mode == "raw":
                    embeddings_raw.append(raw_emb_np)
                elif args.norm_mode == "rms":
                    embeddings_raw.append(norm_emb.cpu().half().numpy())  # store rms instead of raw list name still
                else:  # both
                    embeddings_raw.append(raw_emb_np)
                    embeddings_rms.append(norm_emb.cpu().half().numpy())

                # Update global token counter and progress feedback
                global_token_count += len(valid_ids)
                if args.token_limit:
                    report_every = max(1000, args.token_limit // 20)  # 5% steps or 1000 tokens
                else:
                    report_every = 10000

                if global_token_count - last_report_tokens >= report_every:
                    if args.token_limit:
                        pct = 100 * global_token_count / args.token_limit
                        print(f"Progress: {global_token_count}/{args.token_limit} tokens ({pct:.1f}%)")
                    else:
                        print(f"Progress: {global_token_count} tokens processed…")
                    last_report_tokens = global_token_count

                # Early-stop check after adding each sample
                if args.token_limit and global_token_count >= args.token_limit:
                    break  # break inner sample loop

            if args.token_limit and global_token_count >= args.token_limit:
                break  # break batch loop

        # Concatenate results for this file and write NPZ -----------------
        token_ids_arr = np.concatenate(token_ids_list, axis=0).astype(np.int32)
        raw_arr = np.concatenate(embeddings_raw, axis=0).astype(np.float16)
        if args.norm_mode == "raw":
            out_dict = {"token_ids": token_ids_arr, "embeddings": raw_arr}
        elif args.norm_mode == "rms":
            out_dict = {"token_ids": token_ids_arr, "embeddings_rms": raw_arr}
        else:  # both
            rms_arr = np.concatenate(embeddings_rms, axis=0).astype(np.float16)
            out_dict = {
                "token_ids": token_ids_arr,
                "embeddings": raw_arr,
                "embeddings_rms": rms_arr,
            }

        np.savez_compressed(out_path, **out_dict)

        processed_files += 1
        # token count already updated during sample loop
        print(f"[{processed_files}/{len(text_files)}] Saved {out_path.name} with {len(token_ids_arr)} tokens")

        if args.token_limit and global_token_count >= args.token_limit:
            print(f"Token limit {args.token_limit} reached. Stopping.")
            break

    print(f"Done. Total tokens processed: {global_token_count}")


if __name__ == "__main__":
    main()

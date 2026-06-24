#!/usr/bin/env python3
"""gen_narrow_base.py — generate an int8/uint8-quantized base CSV from a .fbin.

The wiki embeddings are L2-normalized float32 in ~[-1, 1], which cannot be cast
into a VECINT8 / VECUINT8 column (MO rejects fractional / out-of-range values).
To benchmark int8/uint8 *base column* storage we scale each component to the
integer range with a uniform positive scale (+ uniform offset for uint8):

    int8  : clip(round(v * 127),        -127, 127)
    uint8 : clip(round(v * 127 + 128),     0, 255)

A uniform positive scale and a uniform offset both preserve L2 nearest-neighbor
ordering (||k a + c - (k b + c)|| = k ||a - b||), so the existing L2 groundtruth
stays valid; the only new error is the int8 rounding (which is exactly what we
want to measure). Output schema / CSV dialect match gen.py so the same
LOAD DATA path imports it.

    python gen_narrow_base.py --fbin base.1M.fbin --mode int8  -o base.i8_0.csv
    python gen_narrow_base.py --fbin base.1M.fbin --mode uint8 -o base.u8_0.csv
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from gen import (
    _emb_literal,
    _iter_fbin_batches,
    _open_csv_writer,
    _read_fbin_header,
    _sanitize_content,
    FILE_ID_BASE_DEFAULT,
    DISTINCT_FILE_IDS_DEFAULT,
    PAGE_NUM_MOD_DEFAULT,
)
from generate_historical_file_blocks import _content_line, _meta_obj


def quantize(mat: np.ndarray, mode: str) -> np.ndarray:
    """Scale a float batch to integer-valued floats (so _emb_literal emits ints)."""
    if mode == "int8":
        return np.clip(np.round(mat * 127.0), -127.0, 127.0)
    if mode == "uint8":
        return np.clip(np.round(mat * 127.0 + 128.0), 0.0, 255.0)
    raise ValueError(mode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fbin", required=True, help="input base .fbin (float32)")
    ap.add_argument("--mode", required=True, choices=["int8", "uint8"])
    ap.add_argument("-o", "--output", required=True, help="output CSV path")
    ap.add_argument("--expected-dim", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=4000)
    ap.add_argument("--file-id-base", type=int, default=FILE_ID_BASE_DEFAULT)
    ap.add_argument("--distinct-file-ids", type=int, default=DISTINCT_FILE_IDS_DEFAULT)
    ap.add_argument("--page-num-mod", type=int, default=PAGE_NUM_MOD_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n, d = _read_fbin_header(args.fbin)
    if d != args.expected_dim:
        print(f"warn: dim {d} != expected {args.expected_dim}", file=sys.stderr)
    print(f"[gen_narrow_base] {args.mode}: n={n} dim={d} -> {args.output}", file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    fp, w = _open_csv_writer(args.output)
    t0 = time.perf_counter()
    written = 0
    try:
        for first_idx, mat in _iter_fbin_batches(args.fbin, args.batch_size, 0, None):
            q = quantize(mat, args.mode)
            for j in range(q.shape[0]):
                i = first_idx + j  # first_idx is already 1-based
                w.writerow([
                    i,
                    args.file_id_base + (i - 1) % args.distinct_file_ids,
                    _sanitize_content(_content_line(i, rng)),
                    _emb_literal(q[j]),
                    (i - 1) % args.page_num_mod + 1,
                    _meta_obj(i, rng),
                ])
            written += q.shape[0]
            if written % 100000 < args.batch_size:
                print(f"  {written}/{n} rows ({time.perf_counter()-t0:.0f}s)", file=sys.stderr, flush=True)
    finally:
        fp.close()
    print(f"[gen_narrow_base] done: {written} rows, {time.perf_counter()-t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

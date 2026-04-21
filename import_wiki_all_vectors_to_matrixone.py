#!/usr/bin/env python3
"""
将 cuVS Bench Wiki-all 向量文件（.fbin，见 RAPIDS 文档）流式导入 MatrixOne，
写入 embedding 列（如 VECF32(768)）；content / page_num / meta 与 generate_historical_file_blocks_1m.py 一致。
file_id **固定为 50 个不同取值**：file_id_base + (行号-1) % 50（默认 base=20000000，即 20000000..20000049）。

数据集说明与下载：
https://docs.rapids.ai/api/cuvs/nightly/cuvs_bench/wiki_all_dataset/

典型步骤：
1. 下载子集：wiki_all_1M（约 2.9GB）或 wiki_all_10M / 全量 88M。
2. 解压后在目录中找到扩展名为 .fbin 的向量库文件（常见名为 base.fbin，以你解压结果为准）。
3. 建表 historical_file_blocks_50m（embedding vecf32(768)）。
4. 运行本脚本：--fbin /path/to/xxx.fbin --table historical_file_blocks_50m ...

.fbin 布局（cuVS / RAFT 常用）：小端 uint32 num_vectors, uint32 dim，随后 num_vectors*dim 个 float32，行主序。
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from typing import Iterator

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
from generate_historical_file_blocks import (  # noqa: E402
    _content_line,
    _meta_obj,
)

# 与业务要求一致：全表仅循环使用 50 个 file_id
DISTINCT_FILE_IDS = 50

try:
    import numpy as np
except ImportError:
    print("需要 numpy: pip install numpy", file=sys.stderr)
    sys.exit(1)

try:
    import pymysql
except ImportError:
    print("需要 pymysql: pip install pymysql", file=sys.stderr)
    sys.exit(1)


def _emb_vec_component(x: float) -> str:
    if not np.isfinite(x):
        return "0"
    xf = float(np.float32(x))
    if xf == 0.0:
        return "0"
    s = f"{xf:.10f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s == "-0":
        return "0"
    return s


def _emb_literal_from_vec(v: np.ndarray) -> str:
    v = np.asarray(v, dtype=np.float32).ravel()
    return "[" + ",".join(_emb_vec_component(float(t)) for t in v) + "]"


def _read_fbin_header(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        hdr = f.read(8)
        if len(hdr) != 8:
            raise ValueError(f"文件过短，无法读取 8 字节头: {path}")
    n, d = struct.unpack("<II", hdr)
    return int(n), int(d)


def _iter_fbin_batches(
    path: str, batch_rows: int, skip_rows: int, max_rows: int | None
) -> Iterator[tuple[int, np.ndarray]]:
    """
    产出 (global_start_index_1based, ndarray shape (b, d))。
    global_start_index_1based 为本 batch 第一行在整个导出中的行号（从 1 起）。
    """
    n_total, d = _read_fbin_header(path)
    if skip_rows < 0 or skip_rows >= n_total:
        raise ValueError(f"skip-rows 非法: {skip_rows}，文件共 {n_total} 行")
    end = n_total if max_rows is None else min(n_total, skip_rows + max_rows)
    row = skip_rows
    base_off = 8 + row * d * 4
    with open(path, "rb") as f:
        f.seek(base_off)
        first_idx = row + 1  # 1-based 行号
        while row < end:
            take = min(batch_rows, end - row)
            nbytes = take * d * 4
            raw = f.read(nbytes)
            if len(raw) != nbytes:
                raise ValueError(f"期望读取 {nbytes} 字节，实际 {len(raw)}（文件截断？）")
            mat = np.frombuffer(raw, dtype=np.float32).reshape(take, d)
            yield first_idx, mat
            row += take
            first_idx += take


def main() -> None:
    ap = argparse.ArgumentParser(description="Wiki-all .fbin 向量导入 MatrixOne historical_file_blocks 类表")
    ap.add_argument(
        "--fbin",
        required=True,
        nargs="+",
        help="向量库 .fbin 路径，可指定多个（按顺序顺延全局行号 i）",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6001)
    ap.add_argument("--user", default="dump")
    ap.add_argument("--password", default="111")
    ap.add_argument("--database", default="jsp_app")
    ap.add_argument("--table", default="historical_file_blocks_50m")
    ap.add_argument("--expected-dim", type=int, default=768, help="应与表 embedding 维度一致")
    ap.add_argument("--batch-size", type=int, default=200, help="每批 executemany 行数")
    ap.add_argument("--skip-rows", type=int, default=0, help="跳过前 K 条向量（断点续导）")
    ap.add_argument("--max-rows", type=int, default=None, help="最多导入条数（默认读完整个文件可用部分）")
    ap.add_argument("--file-id-base", type=int, default=20_000_000)
    ap.add_argument("--page-num-mod", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    totals = []
    for p in args.fbin:
        n, d = _read_fbin_header(p)
        if d != args.expected_dim:
            print(
                f"错误: 文件 {p} dim={d} 与 --expected-dim={args.expected_dim} 不一致",
                file=sys.stderr,
            )
            sys.exit(2)
        totals.append(n)
    n_total = sum(totals)

    rng = np.random.default_rng(args.seed)

    sql = (
        f"INSERT INTO `{args.table}` "
        "(file_id, content, embedding, page_num, meta) VALUES (%s, %s, %s, %s, %s)"
    )

    conn = pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database or None,
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=120,
        read_timeout=600,
        write_timeout=600,
    )

    total_done = 0
    global_i = 1  # 跨文件的 1-based 全局行号
    try:
        with conn.cursor() as cur:
            for pi, path in enumerate(args.fbin):
                file_skip = args.skip_rows if pi == 0 else 0
                file_max = args.max_rows if pi == 0 and args.max_rows is not None else None
                print(
                    f"  [{pi + 1}/{len(args.fbin)}] {path}",
                    file=sys.stderr,
                    flush=True,
                )
                for _first_idx_in_file, mat in _iter_fbin_batches(
                    path, args.batch_size, file_skip, file_max
                ):
                    b, d = mat.shape
                    if d != args.expected_dim:
                        raise RuntimeError("batch 维度与头不一致")
                    batch = []
                    for j in range(b):
                        i = global_i + j
                        vec = mat[j]
                        emb_str = _emb_literal_from_vec(vec)
                        fid = args.file_id_base + (i - 1) % DISTINCT_FILE_IDS
                        pnum = (i - 1) % args.page_num_mod + 1
                        batch.append(
                            (
                                fid,
                                _content_line(i, rng),
                                emb_str,
                                pnum,
                                _meta_obj(i, rng),
                            )
                        )
                    cur.executemany(sql, batch)
                    conn.commit()
                    global_i += b
                    total_done += b
                    print(
                        f"  imported {total_done} / total_vectors={n_total}",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        conn.close()

    print(f"完成: 共写入 {total_done} 行 -> {args.database}.{args.table}", file=sys.stderr)


if __name__ == "__main__":
    main()

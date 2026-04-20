#!/usr/bin/env python3
"""
gen.py — 将 .fbin 向量文件转换为 MatrixOne LOAD DATA 可用的 CSV。

相对 INSERT / executemany 批量导入（通常 ~200 行/批），LOAD DATA INFILE 能把百万级
向量导入从数十分钟降到数分钟量级。

输出列与 historical_file_blocks_wiki 对齐（6 列，无表头，首列 \\N 触发 AUTO_INCREMENT）：
    id \\N, file_id, content, embedding, page_num, meta

字段生成规则与 import_wiki_all_vectors_to_matrixone.py / generate_historical_file_blocks.py 保持一致：
    file_id  = file_id_base + (行号-1) % distinct_file_ids
    content  = _content_line(i, rng)
    embedding= "[v1,v2,...]"（十进制小数，尾随 0 裁剪；float32）
    page_num = (行号-1) % page_num_mod + 1
    meta     = _meta_obj(i, rng)

典型用法：
    # 1) 生成 CSV
    python gen.py --fbin wiki_all_1M/base.1M.fbin -o wiki_1M.csv

    # 2) 在 MatrixOne 客户端（开启 local_infile）执行：
    LOAD DATA LOCAL INFILE '/abs/path/wiki_1M.csv'
      INTO TABLE historical_file_blocks_wiki
      FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY ''
      LINES TERMINATED BY '\\n';

或用本脚本内置的 --load 直连 MatrixOne 执行 LOAD DATA LOCAL INFILE。

也支持回读已有 CSV 并 LOAD：
    python gen.py --csv wiki_1M.csv --load --host 127.0.0.1 --port 6001 ...

注意：embedding 列含逗号与方括号，字段用双引号包裹；CSV 使用 QUOTE_MINIMAL，
content 中换行会被正确转义（LOAD DATA 的 ENCLOSED BY '"' 能处理换行）。
"""
from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
from typing import Iterator, Tuple

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

try:
    import numpy as np
except ImportError:
    print("需要 numpy: pip install numpy", file=sys.stderr)
    sys.exit(1)

from generate_historical_file_blocks import _content_line, _meta_obj  # noqa: E402


DISTINCT_FILE_IDS_DEFAULT = 50
FILE_ID_BASE_DEFAULT = 20_000_000
PAGE_NUM_MOD_DEFAULT = 800


def _emb_component(x: float) -> str:
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


def _emb_literal(v: np.ndarray) -> str:
    v = np.asarray(v, dtype=np.float32).ravel()
    return "[" + ",".join(_emb_component(float(t)) for t in v) + "]"


def _read_fbin_header(path: str) -> Tuple[int, int]:
    with open(path, "rb") as f:
        hdr = f.read(8)
    if len(hdr) != 8:
        raise ValueError(f"文件过短，无法读取 8 字节头: {path}")
    n, d = struct.unpack("<II", hdr)
    return int(n), int(d)


def _iter_fbin_batches(
    path: str, batch_rows: int, skip_rows: int, max_rows: int | None
) -> Iterator[Tuple[int, np.ndarray]]:
    n_total, d = _read_fbin_header(path)
    if skip_rows < 0 or skip_rows >= n_total:
        raise ValueError(f"skip-rows 非法: {skip_rows}，文件共 {n_total} 行")
    end = n_total if max_rows is None else min(n_total, skip_rows + max_rows)
    row = skip_rows
    with open(path, "rb") as f:
        f.seek(8 + row * d * 4)
        first_idx = row + 1  # 1-based
        while row < end:
            take = min(batch_rows, end - row)
            nbytes = take * d * 4
            raw = f.read(nbytes)
            if len(raw) != nbytes:
                raise ValueError(
                    f"期望读取 {nbytes} 字节，实际 {len(raw)}（文件截断？）"
                )
            mat = np.frombuffer(raw, dtype=np.float32).reshape(take, d)
            yield first_idx, mat
            row += take
            first_idx += take


def convert_fbin_to_csv(
    fbin_path: str,
    output_file: str,
    expected_dim: int,
    batch_size: int = 2000,
    skip_rows: int = 0,
    max_rows: int | None = None,
    file_id_base: int = FILE_ID_BASE_DEFAULT,
    distinct_file_ids: int = DISTINCT_FILE_IDS_DEFAULT,
    page_num_mod: int = PAGE_NUM_MOD_DEFAULT,
    seed: int = 42,
    progress_every: int = 50_000,
) -> int:
    n_total, d = _read_fbin_header(fbin_path)
    if d != expected_dim:
        raise ValueError(
            f"文件 dim={d} 与 --expected-dim={expected_dim} 不一致"
        )
    print(
        f"读取 {fbin_path}（n={n_total}, dim={d}），写入 CSV → {output_file}",
        file=sys.stderr,
    )

    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    written = 0

    tmp = output_file + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(
            fp,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        for first_i, mat in _iter_fbin_batches(
            fbin_path, batch_size, skip_rows, max_rows
        ):
            b = mat.shape[0]
            for j in range(b):
                i = first_i + j
                w.writerow(
                    [
                        "\\N",
                        file_id_base + (i - 1) % distinct_file_ids,
                        _content_line(i, rng),
                        _emb_literal(mat[j]),
                        (i - 1) % page_num_mod + 1,
                        _meta_obj(i, rng),
                    ]
                )
            written += b
            if written % progress_every < batch_size:
                print(f"  wrote {written} rows...", file=sys.stderr, flush=True)

    os.replace(tmp, output_file)
    st = os.stat(output_file)
    print(
        f"完成：{output_file} ({st.st_size / (1024**3):.3f} GiB, {written} 行)",
        file=sys.stderr,
    )
    return written


def load_csv_into_matrixone(
    csv_path: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    table: str,
) -> None:
    try:
        import pymysql
    except ImportError:
        print("需要 pymysql: pip install pymysql", file=sys.stderr)
        sys.exit(1)

    abs_path = os.path.abspath(csv_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(abs_path)

    # LOCAL INFILE 需要客户端启用 local_infile；pymysql 的 connect 参数 local_infile=True
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        local_infile=True,
        autocommit=False,
        connect_timeout=120,
        read_timeout=3600,
        write_timeout=3600,
    )
    sql = (
        f"LOAD DATA LOCAL INFILE %s INTO TABLE `{table}` "
        f"FIELDS TERMINATED BY ',' ENCLOSED BY '\"' ESCAPED BY '' "
        f"LINES TERMINATED BY '\\n'"
    )
    print(f"执行：{sql} (file={abs_path})", file=sys.stderr)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (abs_path,))
            affected = cur.rowcount
        conn.commit()
        print(f"LOAD DATA 完成：affected_rows={affected}", file=sys.stderr)
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="生成 LOAD DATA 兼容 CSV（从 .fbin）并可选执行 LOAD DATA LOCAL INFILE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fbin", help="输入 .fbin 路径（cuVS/RAFT 格式）")
    src.add_argument("--csv", help="已有 CSV 路径（跳过生成，直接 --load）")

    ap.add_argument("-o", "--output", help="输出 CSV 路径（与 --fbin 搭配）")
    ap.add_argument("--expected-dim", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=2000, help="读取 .fbin 每批行数")
    ap.add_argument("--skip-rows", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--file-id-base", type=int, default=FILE_ID_BASE_DEFAULT)
    ap.add_argument("--distinct-file-ids", type=int, default=DISTINCT_FILE_IDS_DEFAULT)
    ap.add_argument("--page-num-mod", type=int, default=PAGE_NUM_MOD_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)

    # 可选：生成后直接 LOAD DATA LOCAL INFILE
    ap.add_argument("--load", action="store_true", help="生成后（或 --csv 时直接）执行 LOAD DATA")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6001)
    ap.add_argument("--user", default="dump")
    ap.add_argument("--password", default="111")
    ap.add_argument("--database", default="jst_app_wiki")
    ap.add_argument("--table", default="historical_file_blocks_wiki")

    args = ap.parse_args()

    if args.fbin:
        if not args.output:
            print("错误：使用 --fbin 时需提供 --output", file=sys.stderr)
            return 2
        convert_fbin_to_csv(
            fbin_path=args.fbin,
            output_file=args.output,
            expected_dim=args.expected_dim,
            batch_size=args.batch_size,
            skip_rows=args.skip_rows,
            max_rows=args.max_rows,
            file_id_base=args.file_id_base,
            distinct_file_ids=args.distinct_file_ids,
            page_num_mod=args.page_num_mod,
            seed=args.seed,
        )
        csv_path = args.output
    else:
        csv_path = args.csv

    if args.load:
        load_csv_into_matrixone(
            csv_path=csv_path,
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            database=args.database,
            table=args.table,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

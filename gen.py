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
from typing import Iterator, List, Tuple, Union

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


def _emb_literal(v: np.ndarray) -> str:
    """向量化格式化：一次把整个 768 维向量转成 "[v1,v2,...]"，
    对等价于旧的逐元素 f"{x:.10f}" + rstrip('0').rstrip('.') 语义。
    """
    v = np.asarray(v, dtype=np.float32).ravel()
    # 非有限值 → 0
    v = np.where(np.isfinite(v), v, np.float32(0.0))
    # 整行一次性 %.10f
    parts = np.char.mod("%.10f", v)
    # 裁剪尾随 0 与孤悬小数点
    parts = np.char.rstrip(parts, "0")
    parts = np.char.rstrip(parts, ".")
    # "-0" / "" → "0"
    parts = np.where((parts == "-0") | (parts == ""), "0", parts)
    return "[" + ",".join(parts) + "]"


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


def _open_csv_writer(path: str):
    fp = open(path, "w", encoding="utf-8", newline="")
    w = csv.writer(fp, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    # Header（LOAD DATA IGNORE 1 LINES 会跳过）
    w.writerow(["id", "file_id", "content", "embedding", "page_num", "meta"])
    return fp, w


def _sanitize_content(s: str) -> str:
    """移除 content 中的换行，避免 LOAD DATA 在 PARALLEL 分片或解析时把一行拆成多行。
    用 ' | ' 替代以保留分隔语义。
    """
    return s.replace("\r\n", " | ").replace("\n", " | ").replace("\r", " | ")


def _write_row(w, i: int, vec: np.ndarray,
               file_id_base: int, distinct_file_ids: int, page_num_mod: int,
               rng: np.random.Generator) -> None:
    w.writerow(
        [
            i,
            file_id_base + (i - 1) % distinct_file_ids,
            _sanitize_content(_content_line(i, rng)),
            _emb_literal(vec),
            (i - 1) % page_num_mod + 1,
            _meta_obj(i, rng),
        ]
    )


def convert_fbin_to_csv(
    fbin_path: Union[str, List[str]],
    output_file: str | None = None,
    expected_dim: int = 768,
    batch_size: int = 2000,
    skip_rows: int = 0,
    max_rows: int | None = None,
    file_id_base: int = FILE_ID_BASE_DEFAULT,
    distinct_file_ids: int = DISTINCT_FILE_IDS_DEFAULT,
    page_num_mod: int = PAGE_NUM_MOD_DEFAULT,
    seed: int = 42,
    progress_every: int = 50_000,
    output_prefix: str | None = None,
) -> List[str]:
    """将一个或多个 .fbin 文件转换为 CSV。返回生成的 CSV 路径列表。

    - output_file：合并写入到单个 CSV。
    - output_prefix：每个 .fbin 生成一个 CSV（{prefix}0.csv、{prefix}1.csv ...）。
    两者必须二选一。

    多文件场景下全局行号 i 连续递增（shard1: 1..N1，shard2: N1+1..N1+N2 ...），
    保持 file_id / page_num / meta / content 分布与单文件语义一致。
    """
    if (output_file is None) == (output_prefix is None):
        raise ValueError("必须提供 output_file 或 output_prefix 之一")

    paths = [fbin_path] if isinstance(fbin_path, str) else list(fbin_path)
    if not paths:
        raise ValueError("fbin_path 为空")

    totals = []
    for p in paths:
        n, d = _read_fbin_header(p)
        if d != expected_dim:
            raise ValueError(f"{p} dim={d} 与 --expected-dim={expected_dim} 不一致")
        totals.append(n)

    if output_file is not None:
        print(
            f"读取 {len(paths)} 个 .fbin（总 n={sum(totals)}, dim={expected_dim}），"
            f"合并写入 CSV → {output_file}",
            file=sys.stderr,
        )
    else:
        print(
            f"读取 {len(paths)} 个 .fbin（总 n={sum(totals)}, dim={expected_dim}），"
            f"按前缀分片写入 CSV → {output_prefix}{{0..{len(paths)-1}}}.csv",
            file=sys.stderr,
        )

    rng = np.random.default_rng(seed)
    written = 0
    global_i = 1
    outputs: List[str] = []

    def _ensure_dir_for(path: str) -> None:
        d = os.path.dirname(os.path.abspath(path))
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    if output_file is not None:
        _ensure_dir_for(output_file)
        tmp = output_file + ".tmp"
        fp, w = _open_csv_writer(tmp)
        try:
            for pi, path in enumerate(paths):
                file_skip = skip_rows if pi == 0 else 0
                file_max = max_rows if pi == 0 and max_rows is not None else None
                print(f"  [{pi + 1}/{len(paths)}] {path}", file=sys.stderr, flush=True)
                for _first_idx_in_file, mat in _iter_fbin_batches(
                    path, batch_size, file_skip, file_max
                ):
                    b = mat.shape[0]
                    for j in range(b):
                        _write_row(
                            w, global_i + j, mat[j],
                            file_id_base, distinct_file_ids, page_num_mod, rng,
                        )
                    global_i += b
                    written += b
                    if written % progress_every < batch_size:
                        print(f"  wrote {written} rows...", file=sys.stderr, flush=True)
        finally:
            fp.close()
        os.replace(tmp, output_file)
        st = os.stat(output_file)
        print(
            f"完成：{output_file} ({st.st_size / (1024**3):.3f} GiB, {written} 行)",
            file=sys.stderr,
        )
        outputs.append(output_file)
    else:
        for pi, path in enumerate(paths):
            out_path = f"{output_prefix}{pi}.csv"
            _ensure_dir_for(out_path)
            tmp = out_path + ".tmp"
            fp, w = _open_csv_writer(tmp)
            file_written = 0
            try:
                file_skip = skip_rows if pi == 0 else 0
                file_max = max_rows if pi == 0 and max_rows is not None else None
                print(
                    f"  [{pi + 1}/{len(paths)}] {path} → {out_path}",
                    file=sys.stderr,
                    flush=True,
                )
                for _first_idx_in_file, mat in _iter_fbin_batches(
                    path, batch_size, file_skip, file_max
                ):
                    b = mat.shape[0]
                    for j in range(b):
                        _write_row(
                            w, global_i + j, mat[j],
                            file_id_base, distinct_file_ids, page_num_mod, rng,
                        )
                    global_i += b
                    written += b
                    file_written += b
                    if written % progress_every < batch_size:
                        print(f"  wrote {written} rows...", file=sys.stderr, flush=True)
            finally:
                fp.close()
            os.replace(tmp, out_path)
            st = os.stat(out_path)
            print(
                f"  -> {out_path} ({st.st_size / (1024**3):.3f} GiB, {file_written} 行)",
                file=sys.stderr,
            )
            outputs.append(out_path)
        print(f"完成：共 {len(outputs)} 个 CSV，总 {written} 行", file=sys.stderr)

    return outputs


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

    # 使用服务端 LOAD DATA INFILE（非 LOCAL），CSV 必须在 MatrixOne 服务端可访问路径上。
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=120,
        read_timeout=3600,
        write_timeout=3600,
    )
    tbl = f"`{table}`"
    sql = (
        f"LOAD DATA INFILE '{abs_path}' "
        f"INTO TABLE {tbl} "
        f"FIELDS TERMINATED BY ',' "
        f"ENCLOSED BY '\"' "
        f"LINES TERMINATED BY '\\r\\n' "
        f"IGNORE 1 LINES "
        f"PARALLEL 'true'"
    )
    print(f"执行：{sql}", file=sys.stderr)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
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
    src.add_argument(
        "--fbin",
        nargs="+",
        help="输入 .fbin 路径，可指定多个（cuVS/RAFT 格式，按顺序全局行号连续）",
    )
    src.add_argument("--csv", help="已有 CSV 路径（跳过生成，直接 --load）")
    src.add_argument(
        "--csv-prefix",
        help="已有 CSV 前缀（匹配 {prefix}*.csv，全部 --load）",
    )

    out_grp = ap.add_mutually_exclusive_group()
    out_grp.add_argument("-o", "--output", help="输出到单个 CSV（与 --fbin 搭配）")
    out_grp.add_argument(
        "--output-csv-prefix",
        help="输出到多个 CSV（{prefix}0.csv, {prefix}1.csv ...），每个 .fbin 对应一个 CSV",
    )
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
        if not args.output and not args.output_csv_prefix:
            print("错误：使用 --fbin 时需提供 --output 或 --output-csv-prefix", file=sys.stderr)
            return 2
        csv_paths = convert_fbin_to_csv(
            fbin_path=args.fbin,
            output_file=args.output,
            output_prefix=args.output_csv_prefix,
            expected_dim=args.expected_dim,
            batch_size=args.batch_size,
            skip_rows=args.skip_rows,
            max_rows=args.max_rows,
            file_id_base=args.file_id_base,
            distinct_file_ids=args.distinct_file_ids,
            page_num_mod=args.page_num_mod,
            seed=args.seed,
        )
    elif args.csv:
        csv_paths = [args.csv]
    else:  # args.csv_prefix
        import glob as _glob
        csv_paths = sorted(_glob.glob(f"{args.csv_prefix}*.csv"))
        if not csv_paths:
            print(f"错误：未匹配到 {args.csv_prefix}*.csv", file=sys.stderr)
            return 2
        print(f"发现 {len(csv_paths)} 个 CSV：", file=sys.stderr)
        for p in csv_paths:
            print(f"  - {p}", file=sys.stderr)

    if args.load:
        for p in csv_paths:
            load_csv_into_matrixone(
                csv_path=p,
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

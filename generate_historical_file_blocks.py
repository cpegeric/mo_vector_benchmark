#!/usr/bin/env python3
"""
按 jsp_app.historical_file_blocks 结构生成大 CSV（默认 100 万行），用于本地 MatrixOne 导入测试。
默认输出 **6 列**（id, file_id, content, embedding, page_num, meta），无表头，首列为 \\N 以匹配 MatrixOne 表顺序与 AUTO_INCREMENT。
embedding 为方括号列表，数值采用与原表相同的**十进制小数**（无科学计数法），如 -0.028884888、0.03186035。
可选 --five-column 生成旧版 5 列+表头；LOAD 时需 OPTIONALLY ENCLOSED BY '\"' ESCAPED BY ''。
file_id 默认仅在 **50** 个取值间循环（base..base+49），便于过滤/热点场景；可通过 --distinct-file-ids 调整个数。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Any

_ROW_SUFFIX_MULT: dict[str, int] = {
    "k": 10**3,
    "m": 10**6,
    "g": 10**9,
    "t": 10**12,
}


def _parse_rows(value: str) -> int:
    """
    解析行数：正整数、可选千分位下划线，或后缀 k/m/g/t（十进制，如 1m=1_000_000）。
    示例：1000、1_000_000、1k、500k、50m、1.5m（小数仅在与后缀连用时允许）。
    """
    s = str(value).strip().replace("_", "")
    if not s:
        raise argparse.ArgumentTypeError("行数不能为空")
    low = s.lower()
    mult = 1
    if len(low) >= 2 and low[-1] in _ROW_SUFFIX_MULT:
        mult = _ROW_SUFFIX_MULT[low[-1]]
        low = low[:-1]
        if not low:
            raise argparse.ArgumentTypeError(f"无效行数写法: {value!r}")
        if "." in low:
            try:
                coef = float(low)
            except ValueError as e:
                raise argparse.ArgumentTypeError(f"无效行数: {value!r}") from e
            if coef <= 0:
                raise argparse.ArgumentTypeError("行数必须为正数")
            n = int(round(coef * mult))
        else:
            try:
                n = int(low, 10) * mult
            except ValueError as e:
                raise argparse.ArgumentTypeError(f"无效行数: {value!r}") from e
        if n <= 0:
            raise argparse.ArgumentTypeError("行数换算后必须至少为 1")
        return n
    try:
        n = int(low, 10)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"无效行数: {value!r}（可用纯数字或后缀 k/m/g/t，如 50m）"
        ) from e
    if n <= 0:
        raise argparse.ArgumentTypeError("行数必须为正整数")
    return n


try:
    import numpy as np
except ImportError:
    print("需要 numpy: pip install numpy", file=sys.stderr)
    sys.exit(1)


def _emb_vec_component(x: float) -> str:
    """
    与原表 embedding 文本风格一致：十进制小数，无科学计数法。
    原表示例：-0.028884888、0.03186035、0.0064964294（尾随 0 可省略）。
    float32 用最多 10 位小数再裁剪尾随 0，与常见导出位数同量级。
    """
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


def _emb_literal(rng: np.random.Generator, dim: int = 1024) -> str:
    """与线上样本量级相近的高斯向量；字面量为方括号包裹、逗号分隔的十进制小数列表。"""
    v = rng.normal(0.0, 0.04, size=dim).astype(np.float32)
    return "[" + ",".join(_emb_vec_component(float(t)) for t in v) + "]"


def _content_line(i: int, rng: np.random.Generator) -> str:
    h = hashlib.sha256(f"{i}:{rng.integers(0, 2**32)}".encode()).hexdigest()
    # 与现网块相近的数百～千余字，含换行；体积过大则生成/导入极慢
    pad = rng.integers(120, 420)
    frag = "京能电力 技术响应 EPC 采购 法定代表人 目录 供货 偏差表 售后服务 进度计划"
    body = (frag * max(1, pad // len(frag)))[:pad]
    parts = [body, f"块序号{i}", f"doc_sig_{h[:40]}", f"尾缀_{h[40:64]}"]
    return "\n".join(parts)


def _meta_obj(i: int, rng: np.random.Generator) -> str:
    o: dict[str, Any] = {
        "seq": i,
        "shard": int(i // 10_000),
        "noise": int(rng.integers(0, 1_000_000)),
    }
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def _parse_positive_int(value: str) -> int:
    try:
        v = int(str(value).strip().replace("_", ""), 10)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"无效正整数: {value!r}") from e
    if v < 1:
        raise argparse.ArgumentTypeError("必须 >= 1")
    return v


def main() -> None:
    ap = argparse.ArgumentParser(
        description="生成 historical_file_blocks 风格 CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "行数 -n 示例:\n"
            "  %(prog)s -n 10000 -o out.csv\n"
            "  %(prog)s -n 50_000_000 -o big.csv\n"
            "  %(prog)s -n 500k -o half_m.csv      # 500 × 10^3\n"
            "  %(prog)s -n 50m -o 50m.csv          # 50 × 10^6\n"
            "  %(prog)s -n 1.5m -o one_point_five_m.csv\n"
            "后缀 k/m/g/t 为十进制倍数（非 Ki/Mi）。"
        ),
    )
    ap.add_argument(
        "-n",
        "--rows",
        type=_parse_rows,
        default=_parse_rows("1m"),
        help="生成行数：正整数，或带 k/m/g/t 后缀（如 50m=5千万）",
    )
    ap.add_argument("-o", "--output", default="historical_file_blocks_1m.csv", help="输出文件")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument(
        "--distinct-file-ids",
        type=_parse_positive_int,
        default=50,
        metavar="N",
        help=(
            "file_id 仅使用 N 个不同值，按行循环：file_id = base + (行号-1) %% N。"
            "默认 50（即 20000000..20000049）。"
        ),
    )
    ap.add_argument(
        "--five-column",
        action="store_true",
        help="旧版：仅 5 列 (file_id…meta) 且写表头；不含自增 id 列",
    )
    ap.add_argument(
        "--with-header",
        action="store_true",
        help="写入表头一行（与 6 列或 5 列字段名一致）",
    )
    ap.add_argument(
        "--include-id",
        action="store_true",
        help="6 列模式下首列输出具体 id（1..n）而非 \\N；一般仅调试使用",
    )
    ap.add_argument(
        "--matrixone-load",
        action="store_true",
        help="等价默认行为（保留兼容）：6 列、无表头、首列 \\N",
    )
    args = ap.parse_args()
    if args.five_column and args.include_id:
        print("错误: --five-column 与 --include-id 不能同时使用", file=sys.stderr)
        sys.exit(2)

    rng = np.random.default_rng(args.seed)
    path = os.path.abspath(args.output)
    tmp = path + ".tmp"

    if args.five_column:
        fieldnames = ["file_id", "content", "embedding", "page_num", "meta"]
        write_header = True
        six_col = False
        use_null_id = False
    else:
        # 默认：6 列与表定义一致，便于 MatrixOne LOAD（首列 \\N 自增 id）
        six_col = True
        fieldnames = ["id", "file_id", "content", "embedding", "page_num", "meta"]
        write_header = args.with_header
        use_null_id = not args.include_id

    n = args.rows
    n_fid = args.distinct_file_ids
    base_fid = 20_000_000

    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        if write_header:
            w.writeheader()
        for i in range(1, n + 1):
            row = {
                "file_id": base_fid + (i - 1) % n_fid,
                "content": _content_line(i, rng),
                "embedding": _emb_literal(rng),
                "page_num": int((i - 1) % 800) + 1,
                "meta": _meta_obj(i, rng),
            }
            if six_col:
                if use_null_id:
                    row = {"id": "\\N", **row}
                else:
                    row = {"id": i, **row}
            w.writerow(row)
            if i % 50_000 == 0:
                print(f"  wrote {i}/{n}", file=sys.stderr, flush=True)

    os.replace(tmp, path)
    st = os.stat(path)
    print(f"完成: {path} ({st.st_size / (1024**3):.3f} GiB)", file=sys.stderr)


if __name__ == "__main__":
    main()

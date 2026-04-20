#!/usr/bin/env python3
"""
run_wiki.py — Wiki-all 基准测试子命令入口

用法：
  python run_wiki.py <command> --config cfg/xxx.json [options]

命令：
  all           顺序执行 create_table → import → create_index → recall
  create_table  仅创建表
  import        仅导入数据（默认走 .fbin INSERT；加 --csv 走 LOAD DATA LOCAL INFILE）
  create_index  仅创建向量索引
  drop_index    删除向量索引（index.name 取自 JSON；兼容旧名 idx_embedding）
  gen_csv       从 dataset.base_fbin 生成 LOAD DATA 兼容的 6 列 CSV（不连库）
  recall        仅跑召回评估

JSON 配置（cfg/*.json）示例：
  {
    "dataset": {
      "base_fbin":        "/path/to/wiki_all_1M.fbin",
      "query_fbin":       "/path/to/queries.fbin",
      "groundtruth_ibin": "/path/to/groundtruth.neighbors.ibin",
      "id_offset": 1
    }
  }

示例：
  # 全流程（INSERT 导入）
  python run_wiki.py all --config cfg/ivfpq_1M.json -n 5000 -k 100 --concurrency 32

  # 预生成 CSV 并以 LOAD DATA 走全流程
  python run_wiki.py gen_csv --config cfg/ivfpq_1M.json --output /tmp/wiki_1M.csv
  python run_wiki.py all --config cfg/ivfpq_1M.json --csv /tmp/wiki_1M.csv \
      -n 5000 -k 100 --concurrency 32

  # 只跑召回 / 只删索引
  python run_wiki.py recall     --config cfg/ivfpq_1M.json -n 5000 -k 100 --concurrency 32
  python run_wiki.py drop_index --config cfg/ivfpq_1M.json
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_vector_test import (  # noqa: E402
    _ARG_DEFAULTS,
    apply_config_to_args,
    load_index_config,
    run_eval,
    run_wiki_create_index,
    run_wiki_create_table,
    run_wiki_import,
)
from gen import convert_fbin_to_csv, load_csv_into_matrixone  # noqa: E402


COMMANDS = (
    "all",
    "create_table",
    "import",
    "create_index",
    "drop_index",
    "gen_csv",
    "recall",
)


def build_args(cli) -> SimpleNamespace:
    """从 CLI + JSON 构造一个兼容 run_vector_test 各 handler 的 args 命名空间。"""
    cfg = load_index_config(cli.config)
    if cfg is None:
        print(f"错误: --config 读取失败: {cli.config}")
        sys.exit(2)

    ns = SimpleNamespace(**_ARG_DEFAULTS)
    apply_config_to_args(ns, cfg)
    ns._index_config = cfg

    dataset = cfg.get("dataset", {}) or {}
    raw_fbin = dataset.get("base_fbin")
    if raw_fbin is None:
        ns.fbin = None
    elif isinstance(raw_fbin, str):
        ns.fbin = [raw_fbin]
    else:
        ns.fbin = list(raw_fbin)
    ns.csv = cli.csv
    ns.batch_size = cli.batch_size
    ns.file_id_base = cli.file_id_base

    ns.sql_mode = cli.sql_mode
    ns.num_queries = cli.num_queries
    ns.k = cli.k
    ns.concurrency = cli.concurrency
    ns.filter_val = None
    ns.duration = None
    ns.distribute_file_ids = False
    ns.max_distinct_file_ids = 50
    ns.skip_db_verify = True
    ns.probe = (cfg.get("env", {}) or {}).get("probe_limit")
    ns.filter_mode = None
    ns.query_fbin = None
    ns.groundtruth_ibin = None
    ns.id_offset = None

    return ns


def _resolve_input_csvs(cli) -> list[str] | None:
    """根据 CLI 返回用于 LOAD DATA 的 CSV 路径列表；若未指定 CSV 路径返回 None。"""
    if cli.csv:
        return [cli.csv]
    if cli.input_csv_prefix:
        import glob as _glob
        matched = sorted(_glob.glob(f"{cli.input_csv_prefix}*.csv"))
        return matched
    return None


def _validate_import_paths(ns: SimpleNamespace, cli) -> int:
    csvs = _resolve_input_csvs(cli)
    if csvs is not None:
        if not csvs:
            print(f"错误: --input-csv-prefix 未匹配到 {cli.input_csv_prefix}*.csv")
            return 1
        for p in csvs:
            if not os.path.exists(p):
                print(f"错误: CSV 文件不存在: {p}")
                return 1
        return 0
    # fbin 路径
    if not ns.fbin:
        print("错误: JSON 的 dataset.base_fbin 未设置，且未提供 --csv/--input-csv-prefix。")
        return 1
    for p in ns.fbin:
        if not os.path.exists(p):
            print(f"错误: base_fbin 文件不存在: {p}")
            return 1
    return 0


def _validate_recall_paths(ns: SimpleNamespace) -> None:
    ds = ns._index_config.get("dataset", {}) or {}
    qf = ds.get("query_fbin")
    gi = ds.get("groundtruth_ibin")
    if not qf or not os.path.exists(qf):
        print(
            f"警告: dataset.query_fbin 未设置或不存在: {qf!r}。"
            " 召回步骤将改走 DB 抽样 + 在线 ground truth。"
        )
    if not gi or not os.path.exists(gi):
        print(
            f"警告: dataset.groundtruth_ibin 未设置或不存在: {gi!r}。"
            " 召回步骤将改走 DB 抽样 + 在线 ground truth。"
        )


def _banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"[run_wiki] {title}")
    print("=" * 70)


def _make_import_fn(ns: SimpleNamespace, cli):
    csvs = _resolve_input_csvs(cli)
    if csvs is not None:
        def _import_via_csv(ns_):
            for p in csvs:
                print(f"[run_wiki] LOAD DATA: {p}", flush=True)
                load_csv_into_matrixone(
                    csv_path=p,
                    host=ns_.host,
                    port=ns_.port,
                    user=ns_.user,
                    password=ns_.password,
                    database=ns_.database,
                    table=ns_.table,
                )
            return 0
        suffix = f" x{len(csvs)}" if len(csvs) > 1 else ""
        return f"import (csv LOAD DATA{suffix})", _import_via_csv
    return "import", run_wiki_import


def _gen_csv(ns: SimpleNamespace, cli) -> int:
    if not ns.fbin:
        print("错误: JSON 的 dataset.base_fbin 未设置，无法生成 CSV。")
        return 1
    for p in ns.fbin:
        if not os.path.exists(p):
            print(f"错误: base_fbin 文件不存在: {p}")
            return 1
    if not cli.output and not cli.output_csv_prefix:
        print("错误: gen_csv 需要 --output 或 --output-csv-prefix 之一。")
        return 1
    if cli.output and cli.output_csv_prefix:
        print("错误: --output 与 --output-csv-prefix 不能同时指定。")
        return 1
    convert_fbin_to_csv(
        fbin_path=ns.fbin,
        output_file=cli.output,
        output_prefix=cli.output_csv_prefix,
        expected_dim=cli.expected_dim,
        batch_size=cli.gen_batch_size,
        file_id_base=cli.file_id_base,
        distinct_file_ids=cli.distinct_file_ids,
        page_num_mod=cli.page_num_mod,
        seed=cli.seed,
    )
    return 0


def _drop_index(ns: SimpleNamespace) -> int:
    import pymysql

    index_cfg = (ns._index_config.get("index") or {}) if getattr(ns, "_index_config", None) else {}
    idx_name = index_cfg.get("name", "idx_l2")

    try:
        conn = pymysql.connect(
            host=ns.host,
            port=ns.port,
            user=ns.user,
            password=ns.password,
            database=ns.database,
            autocommit=True,
        )
    except Exception as e:
        print(f"错误: 连接数据库失败: {e}")
        return 1

    try:
        with conn.cursor() as cur:
            for name in (idx_name, "idx_embedding"):
                sql = f"DROP INDEX IF EXISTS `{name}` ON `{ns.table}`"
                try:
                    cur.execute(sql)
                    print(f"  执行: {sql} -> ok")
                except Exception as e:
                    print(f"  警告: {sql} 失败（可忽略）: {e}")
        return 0
    finally:
        conn.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage=(
            "run_wiki.py {" + ",".join(COMMANDS) + "} --config CONFIG [options]"
        ),
        description="Wiki-all 基准测试子命令入口：" + " / ".join(COMMANDS),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("command", choices=COMMANDS, metavar="command", help="要执行的子命令")
    parser.add_argument("--config", required=True, help="JSON 配置文件（见 cfg/*.json）")

    # 召回评估
    parser.add_argument("-n", "--num-queries", type=int, default=1000, help="召回评估查询条数（默认: 1000）")
    parser.add_argument("-k", "--k", type=int, default=10, help="Top-K（默认: 10）")
    parser.add_argument("--concurrency", type=int, default=4, help="评估并发数（默认: 4）")
    parser.add_argument(
        "--sql-mode",
        choices=["l2_only", "l2_filter", "l2_filter_threshold"],
        default="l2_only",
        help="SQL 模式（默认: l2_only；cuVS ground truth 仅对 l2_only 有效）",
    )

    # 导入相关
    parser.add_argument("--batch-size", type=int, default=20000, help="导入批量大小（INSERT 路径用，默认: 20000）")
    parser.add_argument("--file-id-base", type=int, default=20000000, help="file_id 起始值（默认: 20000000）")
    parser.add_argument(
        "--csv",
        default=None,
        help="单个 CSV 文件，用于 LOAD DATA（替代 .fbin INSERT 路径）。",
    )
    parser.add_argument(
        "--input-csv-prefix",
        default=None,
        help="输入 CSV 前缀，匹配 {prefix}*.csv 全部 LOAD DATA（import/all 步骤）。",
    )

    # gen_csv 专用
    parser.add_argument("-o", "--output", default=None, help="gen_csv 输出单个 CSV 路径")
    parser.add_argument(
        "--output-csv-prefix",
        default=None,
        help="gen_csv 输出多个 CSV（{prefix}0.csv、{prefix}1.csv ...），每个 .fbin 对应一个",
    )
    parser.add_argument("--expected-dim", type=int, default=768, help="gen_csv 期望向量维度")
    parser.add_argument("--gen-batch-size", type=int, default=2000, help="gen_csv 读取 .fbin 每批行数")
    parser.add_argument("--distinct-file-ids", type=int, default=50, help="gen_csv file_id 循环个数")
    parser.add_argument("--page-num-mod", type=int, default=800, help="gen_csv page_num 周期")
    parser.add_argument("--seed", type=int, default=42, help="gen_csv 随机种子")

    return parser


def main() -> int:
    cli = _build_parser().parse_args()
    ns = build_args(cli)
    cmd = cli.command

    timings: dict[str, float] = {}

    def _run_step(label: str, fn) -> int:
        _banner(label)
        t0 = time.perf_counter()
        rc = fn(ns)
        timings[label] = time.perf_counter() - t0
        if rc:
            print(f"[run_wiki] 步骤失败: {label} (rc={rc})")
        return rc

    if cmd == "gen_csv":
        _banner("gen_csv")
        t0 = time.perf_counter()
        rc = _gen_csv(ns, cli)
        print(f"\n[run_wiki] gen_csv 完成，耗时 {time.perf_counter() - t0:.2f} s")
        return rc

    if cmd == "create_table":
        return _run_step("create-table", run_wiki_create_table)

    if cmd == "import":
        if _validate_import_paths(ns, cli):
            return 1
        label, fn = _make_import_fn(ns, cli)
        return _run_step(label, fn)

    if cmd == "create_index":
        return _run_step("create-index", run_wiki_create_index)

    if cmd == "drop_index":
        return _run_step("drop-index", _drop_index)

    if cmd == "recall":
        _validate_recall_paths(ns)
        return _run_step("run (recall)", run_eval)

    # cmd == "all"
    if _validate_import_paths(ns, cli):
        return 1
    _validate_recall_paths(ns)

    if _run_step("1/4  create-table", run_wiki_create_table):
        return 1
    label, fn = _make_import_fn(ns, cli)
    if _run_step(f"2/4  {label}", fn):
        return 1
    if _run_step("3/4  create-index", run_wiki_create_index):
        return 1
    rc = _run_step("4/4  run (recall)", run_eval)

    _banner("完成：步骤耗时")
    for lbl, elapsed in timings.items():
        print(f"  {lbl:<28} {elapsed:8.2f} s")
    total = sum(timings.values())
    print(f"  {'TOTAL':<28} {total:8.2f} s")
    return rc


if __name__ == "__main__":
    sys.exit(main())

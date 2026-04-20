#!/usr/bin/env python3
"""
run_wiki.py — 一键运行 Wiki-all 基准测试

读取 JSON 配置（cfg/cagra.json、cfg/ivfpq.json、cfg/ivfflat.json、cfg/hnsw.json）
并顺序执行：
  1) create-table    创建数据库与表
  2) import          导入 base.fbin 向量数据
  3) create-index    创建向量索引（cagra / ivfpq / ivfflat / hnsw）
  4) run             跑召回率评估（使用 cuVS query.fbin + groundtruth.neighbors.ibin）

所有文件路径都从 JSON 的 dataset 块读取：
  {
    "dataset": {
      "base_fbin":        "/path/to/wiki_all_1M.fbin",
      "query_fbin":       "/path/to/queries.fbin",
      "groundtruth_ibin": "/path/to/groundtruth.neighbors.ibin",
      "id_offset": 1
    }
  }

典型用法：
  # 全流程
  python run_wiki.py --config cfg/cagra.json -n 1000 -k 10 --concurrency 4

  # 复用现有表/数据，只重建索引 + 评估
  python run_wiki.py --config cfg/cagra.json --skip-create-table --skip-import

  # 仅跑召回（表/索引已就绪）
  python run_wiki.py --config cfg/cagra.json --skip-create-table --skip-import --skip-create-index
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
from gen import load_csv_into_matrixone  # noqa: E402


def build_args(cli) -> SimpleNamespace:
    """从 CLI + JSON 构造一个兼容 run_vector_test 各 handler 的 args 命名空间。"""
    cfg = load_index_config(cli.config)
    if cfg is None:
        print(f"错误: --config 读取失败: {cli.config}")
        sys.exit(2)

    # 1) 先塞入 run_vector_test 的 argparse 默认连接参数
    ns = SimpleNamespace(**_ARG_DEFAULTS)

    # 2) JSON 里的连接字段覆盖默认
    apply_config_to_args(ns, cfg)

    # 3) 挂上 index 配置，供 run_wiki_create_index 和 run_eval(dataset 回退)使用
    ns._index_config = cfg

    # 4) 导入步骤需要的字段
    dataset = cfg.get("dataset", {}) or {}
    ns.fbin = dataset.get("base_fbin")
    ns.csv = cli.csv
    ns.batch_size = cli.batch_size
    ns.file_id_base = cli.file_id_base

    # 5) 评估步骤需要的字段（run_eval 使用 getattr/hasattr，宽松）
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
    # run_eval 会从 _index_config["dataset"] 回退取这三个，这里显式置 None
    ns.query_fbin = None
    ns.groundtruth_ibin = None
    ns.id_offset = None

    return ns


def _validate_paths(ns: SimpleNamespace, cli) -> int:
    """仅在不跳过对应步骤时校验路径存在。"""
    cfg = ns._index_config
    ds = cfg.get("dataset", {}) or {}

    if not cli.skip_import:
        if cli.csv:
            if not os.path.exists(cli.csv):
                print(f"错误: --csv 文件不存在: {cli.csv}")
                return 1
        else:
            if not ns.fbin:
                print("错误: JSON 的 dataset.base_fbin 未设置，且未加 --skip-import / --csv。")
                return 1
            if not os.path.exists(ns.fbin):
                print(f"错误: base_fbin 文件不存在: {ns.fbin}")
                return 1

    # 评估步骤总是会跑，校验 query / groundtruth 路径
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
    return 0


def _banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"[run_wiki] {title}")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wiki-all 一键基准测试：create-table -> import -> create-index -> recall",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="JSON 配置文件（见 cfg/*.json）")
    parser.add_argument("-n", "--num-queries", type=int, default=1000, help="召回评估查询条数（默认: 1000）")
    parser.add_argument("-k", "--k", type=int, default=10, help="Top-K（默认: 10）")
    parser.add_argument("--concurrency", type=int, default=4, help="评估并发数（默认: 4）")
    parser.add_argument(
        "--sql-mode",
        choices=["l2_only", "l2_filter", "l2_filter_threshold"],
        default="l2_only",
        help="SQL 模式（默认: l2_only；cuVS ground truth 仅对 l2_only 有效）",
    )
    parser.add_argument("--skip-create-table", action="store_true", help="跳过步骤 1：创建表")
    parser.add_argument("--skip-import", action="store_true", help="跳过步骤 2：导入数据")
    parser.add_argument("--skip-create-index", action="store_true", help="跳过步骤 3：创建索引")
    parser.add_argument("--batch-size", type=int, default=20000, help="导入批量大小（默认: 20000）")
    parser.add_argument("--file-id-base", type=int, default=20000000, help="file_id 起始值（默认: 20000000）")
    parser.add_argument(
        "--csv",
        default=None,
        help="使用 LOAD DATA LOCAL INFILE 从 CSV 导入（替代 .fbin INSERT 路径）。CSV 应为 gen.py 生成的 6 列格式。",
    )

    cli = parser.parse_args()
    ns = build_args(cli)
    if _validate_paths(ns, cli):
        return 1

    timings: dict[str, float] = {}

    def _run_step(label: str, fn) -> int:
        _banner(label)
        t0 = time.perf_counter()
        rc = fn(ns)
        timings[label] = time.perf_counter() - t0
        if rc:
            print(f"[run_wiki] 步骤失败: {label} (rc={rc})")
        return rc

    if not cli.skip_create_table:
        if _run_step("1/4  create-table", run_wiki_create_table):
            return 1
    else:
        print("[run_wiki] 跳过 1/4 create-table")

    if not cli.skip_import:
        if cli.csv:
            def _import_via_csv(ns_):
                load_csv_into_matrixone(
                    csv_path=cli.csv,
                    host=ns_.host,
                    port=ns_.port,
                    user=ns_.user,
                    password=ns_.password,
                    database=ns_.database,
                    table=ns_.table,
                )
                return 0
            if _run_step("2/4  import (csv LOAD DATA)", _import_via_csv):
                return 1
        else:
            if _run_step("2/4  import", run_wiki_import):
                return 1
    else:
        print("[run_wiki] 跳过 2/4 import")

    if not cli.skip_create_index:
        if _run_step("3/4  create-index", run_wiki_create_index):
            return 1
    else:
        print("[run_wiki] 跳过 3/4 create-index")

    rc = _run_step("4/4  run (recall)", run_eval)

    _banner("完成：步骤耗时")
    for label, elapsed in timings.items():
        print(f"  {label:<20} {elapsed:8.2f} s")
    total = sum(timings.values())
    print(f"  {'TOTAL':<20} {total:8.2f} s")

    return rc


if __name__ == "__main__":
    sys.exit(main())

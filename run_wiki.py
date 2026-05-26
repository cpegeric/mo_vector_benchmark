#!/usr/bin/env python3
"""
run_wiki.py — Wiki-all 基准测试子命令入口

用法：
  python run_wiki.py <command> --config cfg/xxx.json [options]

命令：
  all           顺序执行：清理旧库/建表 → 导入 → 建索引 → recall
                导入优先级：--s3-* / cfg.dataset.s3 > --csv > cfg.base_fbin (INSERT)
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

  # 全流程（S3）：dataset.s3 + cfg/s3_credentials.json（cp example 后填 AK/SK）
  python run_wiki.py all --config cfg/ivfflat_10M.json -n 5000 -k 100 --concurrency 32

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
)
from gen import convert_fbin_to_csv  # noqa: E402
from s3_credentials import DEFAULT_S3_CREDENTIALS_FILE  # noqa: E402
from wiki_pipeline import (  # noqa: E402
    attach_dataset_fields,
    run_all_pipeline,
    run_import_step,
    validate_import_paths,
    validate_recall_paths,
)


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

    ns.csv = cli.csv
    ns.input_csv_prefix = getattr(cli, "input_csv_prefix", None)
    ns.s3_endpoint = getattr(cli, "s3_endpoint", None)
    ns.s3_bucket = getattr(cli, "s3_bucket", None)
    ns.s3_filepath = getattr(cli, "s3_filepath", None)
    ns.s3_region = getattr(cli, "s3_region", None)
    ns.s3_compression = getattr(cli, "s3_compression", None)
    ns.s3_access_key_id = getattr(cli, "s3_access_key_id", None)
    ns.s3_secret_access_key = getattr(cli, "s3_secret_access_key", None)
    ns.s3_credentials_file = getattr(cli, "s3_credentials_file", None)

    ns.sql_mode = cli.sql_mode
    ns.num_queries = cli.num_queries
    ns.k = cli.k
    ns.concurrency = cli.concurrency
    ns.filter_val = cli.filter_val
    ns.duration = None
    ns.distribute_file_ids = False
    ns.max_distinct_file_ids = 50
    ns.filter_mode = cli.filter_mode
    ns.filter_file_id_base = cli.filter_file_id_base
    ns.filter_distinct_file_ids = cli.filter_distinct_file_ids
    ns.query_fbin = None
    ns.groundtruth_ibin = None
    ns.id_offset = None
    attach_dataset_fields(ns, cfg)

    return ns


def _banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"[run_wiki] {title}")
    print("=" * 70)


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
        help="SQL 模式（默认: l2_only；l2_filter/l2_filter_threshold 需配合 --filter-val）",
    )
    parser.add_argument(
        "--filter-val",
        type=int,
        default=None,
        help="file_id 过滤值；l2_filter / l2_filter_threshold 必填。"
        " SQL 中以 WHERE file_id=? 生效，本地 GT 也据此筛可用邻居（eligible recall@k）。",
    )
    parser.add_argument(
        "--filter-mode",
        choices=["pre", "post", "force", "include"],
        default=None,
        help="SQL 后缀：BY RANK WITH OPTION 'mode=pre|post|force|include'（可选）",
    )
    parser.add_argument(
        "--filter-file-id-base",
        type=int,
        default=20000000,
        help="本地过滤 GT 用的 file_id_base（与 gen.py --file-id-base 一致；默认 20000000）",
    )
    parser.add_argument(
        "--filter-distinct-file-ids",
        type=int,
        default=50,
        help="本地过滤 GT 用的 distinct_file_ids（与 gen.py --distinct-file-ids 一致；默认 50）",
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

    # S3 导入（优先级高于 --csv / base_fbin）
    parser.add_argument("--s3-endpoint", default=None, help="S3/OSS endpoint，如 oss-cn-shanghai.aliyuncs.com")
    parser.add_argument("--s3-bucket", default=None, help="S3 bucket 名")
    parser.add_argument(
        "--s3-filepath",
        default=None,
        help="对象路径，支持通配如 wiki/*.csv（相对 bucket）",
    )
    parser.add_argument("--s3-region", default=None, help="区域，如 oss-cn-shanghai")
    parser.add_argument(
        "--s3-compression",
        default=None,
        help="压缩格式：none / gzip / bz2 / lz4 / auto（默认 none）",
    )
    parser.add_argument(
        "--s3-credentials-file",
        default=DEFAULT_S3_CREDENTIALS_FILE,
        help=f"S3 密钥 JSON 路径（默认: {DEFAULT_S3_CREDENTIALS_FILE}，勿提交仓库）",
    )
    parser.add_argument("--s3-access-key-id", default=None, help="Access Key（覆盖凭证文件）")
    parser.add_argument(
        "--s3-secret-access-key",
        default=None,
        help="Secret Key（覆盖凭证文件）",
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

    needs_filter = cli.sql_mode in ("l2_filter", "l2_filter_threshold")
    if cli.command in ("all", "recall") and needs_filter and cli.filter_val is None:
        print(
            f"错误: --sql-mode {cli.sql_mode} 需要 --filter-val=<file_id>（整数）。"
        )
        return 2

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
        if validate_import_paths(ns):
            return 1
        label, fn = run_import_step(ns, log_prefix="[run_wiki]")
        return _run_step(label, fn)

    if cmd == "create_index":
        return _run_step("create-index", run_wiki_create_index)

    if cmd == "drop_index":
        return _run_step("drop-index", _drop_index)

    if cmd == "recall":
        validate_recall_paths(ns)
        return _run_step("run (recall)", run_eval)

    # cmd == "all"
    return run_all_pipeline(ns, log_prefix="[run_wiki]")


if __name__ == "__main__":
    sys.exit(main())

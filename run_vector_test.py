#!/usr/bin/env python3
"""
Wiki-all 向量数据集测试工具

用于 cuVS Bench Wiki-all 数据集（768维向量）的导入和测试。

支持的命令:
  wiki info              - 显示 Wiki 数据集信息
  wiki create-table      - 创建 historical_file_blocks_wiki 表
  wiki import --fbin <path>  - 导入 .fbin 向量数据
  wiki create-index      - 创建向量索引
  wiki test              - 运行搜索测试
  wiki setup             - 一键设置（创建表+导入+建索引+测试）
  ann                    - 生成 ANN 评测文件
  run                    - 运行召回率/QPS 评估

示例用法:
  # 显示数据集信息
  python run_vector_test.py wiki info

  # 创建表
  python run_vector_test.py wiki create-table --database jst_app_wiki

  # 导入 .fbin 数据
  python run_vector_test.py wiki import --fbin /path/to/wiki_all_1M.fbin

  # 创建向量索引
  python run_vector_test.py wiki create-index --ivf-lists 100

  # 运行测试
  python run_vector_test.py wiki test -n 1000 -k 10 --concurrency 4

  # 生成 ANN 文件
  python run_vector_test.py ann --sql-mode l2_only -n 1000 -k 10

  # 运行评估
  python run_vector_test.py run --sql-mode l2_filter --filter-val 20000000 -n 1000 -k 10 --concurrency 100

  # 一键完整流程（自动创建表、导入数据、创建索引）
  python run_vector_test.py wiki setup --fbin /path/to/wiki_all_1M.fbin --ivf-lists 100

数据集信息:
  名称: cuVS Bench Wiki-all
  来源: https://github.com/rapidsai/cuvs
  维度: 768
  格式: .fbin (float32 binary)
"""

import argparse
import json
import os
import subprocess
import sys
import struct
from typing import Iterator, Optional, Dict

# 脚本路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WIKI_IMPORT_SCRIPT = os.path.join(SCRIPT_DIR, "import_wiki_all_vectors_to_matrixone.py")
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_vector_search_from_table.py")
DEFAULT_CONFIG_FILE = os.path.join(SCRIPT_DIR, "sql_config_simple.json")


def check_scripts():
    """检查依赖脚本是否存在"""
    missing = []
    if not os.path.exists(EVAL_SCRIPT):
        missing.append(os.path.basename(EVAL_SCRIPT))
    if missing:
        print(f"错误: 找不到依赖脚本: {', '.join(missing)}")
        print(f"请确保这些脚本与 {os.path.basename(__file__)} 在同一目录")
        sys.exit(1)


def load_sql_config(config_path: str = None) -> dict:
    """加载 SQL 配置文件"""
    path = config_path or DEFAULT_CONFIG_FILE
    if not os.path.exists(path):
        print(f"警告: 配置文件不存在: {path}，将使用内置模式")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_wiki_info():
    """显示 Wiki 数据集信息"""
    print("=" * 70)
    print("Wiki 数据集信息")
    print("=" * 70)
    print("""
数据集: cuVS Bench Wiki-all (768-dim)
来源: https://github.com/rapidsai/cuvs/tree/main/python/cuvs_bench/cuvs_bench/run/data
文件格式: .fbin (float32 binary format)
维度: 768
描述: 维基百科文章的向量嵌入，用于向量相似性搜索基准测试

推荐操作:
1. wiki info       - 显示此信息
2. wiki create-table - 创建 historical_file_blocks_wiki 表
3. wiki import --fbin <path> - 导入 .fbin 数据
4. wiki create-index - 创建向量索引
5. wiki test - 运行搜索测试
6. wiki setup --fbin <path> --ivf-lists 100 - 一键设置（自动创建表、导入、建索引）
""")
    return 0


def run_wiki_create_table(args):
    """创建 Wiki 向量表（先删除已存在的数据库，再重新创建）"""
    import pymysql

    print("=" * 70)
    print("创建 Wiki 向量表")
    print("=" * 70)

    try:
        # 步骤 1: 删除已存在的数据库（不指定 database 连接）
        conn = pymysql.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
        )
        with conn.cursor() as cur:
            drop_db_sql = f"DROP DATABASE IF EXISTS `{args.database}`"
            cur.execute(drop_db_sql)
            print(f"  数据库 {args.database} 已删除（如果存在）")

            create_db_sql = f"CREATE DATABASE `{args.database}`"
            cur.execute(create_db_sql)
            print(f"  数据库 {args.database} 已创建")
        conn.commit()
        conn.close()

        # 步骤 2: 连接指定数据库并创建表
        conn = pymysql.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            database=args.database,
        )
        create_table_sql = f"""
        CREATE TABLE `{args.table}` (
            `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT '主键',
            `file_id` BIGINT NOT NULL,
            `content` TEXT DEFAULT NULL,
            `embedding` VECF32(768) DEFAULT NULL,
            `page_num` INT NOT NULL DEFAULT 0,
            `meta` JSON DEFAULT NULL,
            PRIMARY KEY (`id`),
            KEY `idx_file` (`file_id`),
            FULLTEXT `idx_content`(`content`) WITH PARSER ngram
        )
        """
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            print(f"  表 {args.table} 已创建")
        conn.commit()
        conn.close()
        print("完成!")
        return 0
    except Exception as e:
        print(f"错误: 创建表失败: {e}")
        return 1


def run_wiki_import(args):
    """导入 Wiki .fbin 数据"""
    print("=" * 70)
    print("导入 Wiki .fbin 数据")
    print("=" * 70)

    if not os.path.exists(WIKI_IMPORT_SCRIPT):
        print(f"错误: 导入脚本不存在: {WIKI_IMPORT_SCRIPT}")
        return 1

    if not args.fbin:
        print("错误: 请指定 --fbin 参数提供 .fbin 文件路径")
        return 1

    if not os.path.exists(args.fbin):
        print(f"错误: .fbin 文件不存在: {args.fbin}")
        return 1

    cmd = [
        "python3",
        WIKI_IMPORT_SCRIPT,
        "--host", str(args.host),
        "--port", str(args.port),
        "--user", str(args.user),
        "--password", str(args.password),
        "--database", str(args.database),
        "--table", str(args.table),
        "--fbin", str(args.fbin),
        "--batch-size", str(args.batch_size),
        "--file-id-base", str(args.file_id_base),
    ]

    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_wiki_create_index(args):
    """创建 Wiki 向量索引"""
    import pymysql

    print("=" * 70)
    print("创建 Wiki 向量索引")
    print("=" * 70)

    try:
        conn = pymysql.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            database=args.database,
        )

        # 删除旧索引（如果存在），忽略错误
        try:
            drop_idx_sql = f"DROP INDEX IF EXISTS idx_embedding ON `{args.table}`"
            with conn.cursor() as cur:
                cur.execute(drop_idx_sql)
                print("  旧索引已删除（如果存在）")
        except Exception:
            # 索引可能不存在，忽略错误继续
            pass

        # 创建新索引（MatrixOne 格式）
        ivf_lists = args.ivf_lists
        create_idx_sql = f'''CREATE INDEX idx_l2 USING ivfflat ON `{args.table}`(embedding) lists={ivf_lists} op_type "vector_l2_ops"'''
        with conn.cursor() as cur:
            cur.execute(create_idx_sql)
            print(f'  向量索引已创建 (IVFFLAT, lists={ivf_lists}, op_type="vector_l2_ops")')

        conn.commit()
        conn.close()
        print("完成!")
        return 0
    except Exception as e:
        print(f"错误: 创建索引失败: {e}")
        return 1


def run_wiki_test(args):
    """运行 Wiki 向量搜索测试（调用 eval_vector_search_from_table.py）"""
    print("=" * 70)
    print("运行 Wiki 向量搜索测试")
    print("=" * 70)

    sql_mode = getattr(args, 'sql_mode', 'l2_only')
    filter_val = getattr(args, 'filter_val', None)

    print(f"  SQL 模式: {sql_mode}")
    if sql_mode in ['l2_filter', 'l2_filter_threshold'] and filter_val:
        print(f"  Filter 值: {filter_val}")

    # 调用 eval_vector_search_from_table.py 进行测试
    cmd = [sys.executable, EVAL_SCRIPT]
    cmd.extend(["--mode", sql_mode])
    cmd.extend(["--k", str(args.k)])
    cmd.extend(["--num-queries", str(args.num_queries)])
    cmd.extend(["--concurrency", str(args.concurrency)])
    cmd.extend(["--database", args.database])
    cmd.extend(["--table", args.table])

    # S2/S3 过滤值
    if filter_val:
        cmd.extend(["--mode23-filter", str(filter_val)])

    # 跳过数据库验证
    cmd.append("--skip-db-verify")

    print(f"\n执行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_ann(args):
    """生成 ANN 评测文件（调用 eval_vector_search_from_table.py）"""
    cmd = [sys.executable, EVAL_SCRIPT]

    # 基本参数
    cmd.extend(["--mode", args.sql_mode])
    cmd.extend(["--k", str(args.k)])
    cmd.extend(["--num-queries", str(args.num_queries)])
    cmd.extend(["--concurrency", str(args.concurrency)])

    # 数据库配置
    if args.database:
        cmd.extend(["--database", args.database])

    # 表名
    if hasattr(args, 'table') and args.table:
        cmd.extend(["--table", args.table])

    # S2/S3 过滤值
    if args.filter_val:
        cmd.extend(["--mode23-filter", str(args.filter_val)])
    
    # ANN 文件生成选项
    cmd.append("--write-ann-files")
    cmd.append("--annfiles-only")
    
    # file_id 分布选项
    if args.distribute_file_ids:
        cmd.append("--ann-distribute-file-ids")
        if args.max_distinct_file_ids != 50:
            cmd.extend(["--ann-max-distinct-file-ids", str(args.max_distinct_file_ids)])

    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_eval(args):
    """运行召回率/QPS 评估（调用 eval_vector_search_from_table.py）"""
    cmd = [sys.executable, EVAL_SCRIPT]

    # 基本参数
    cmd.extend(["--mode", args.sql_mode])
    cmd.extend(["--k", str(args.k)])
    cmd.extend(["--num-queries", str(args.num_queries)])
    cmd.extend(["--concurrency", str(args.concurrency)])

    # 数据库配置
    if args.database:
        cmd.extend(["--database", args.database])

    # 表名
    if hasattr(args, 'table') and args.table:
        cmd.extend(["--table", args.table])

    # S2/S3 过滤值
    if args.filter_val:
        cmd.extend(["--mode23-filter", str(args.filter_val)])

    # 持续时间（压测模式）
    if hasattr(args, 'duration') and args.duration:
        cmd.extend(["--duration", str(args.duration)])

    # file_id 分布选项
    if hasattr(args, 'distribute_file_ids') and args.distribute_file_ids:
        cmd.append("--ann-distribute-file-ids")
        if hasattr(args, 'max_distinct_file_ids') and args.max_distinct_file_ids != 50:
            cmd.extend(["--ann-max-distinct-file-ids", str(args.max_distinct_file_ids)])

    # 跳过数据库验证
    if hasattr(args, 'skip_db_verify') and args.skip_db_verify:
        cmd.append("--skip-db-verify")

    # probe_limit 设置
    if hasattr(args, 'probe') and args.probe is not None:
        cmd.extend(["--probe", str(args.probe)])

    # filter_mode 设置
    if hasattr(args, 'filter_mode') and args.filter_mode:
        cmd.extend(["--filter-mode", args.filter_mode])

    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_wiki_setup(args):
    """一键设置: 创建表、导入数据、创建索引、测试"""
    print("=" * 70)
    print("Wiki 一键设置")
    print("=" * 70)

    # 如果指定了 --fbin，自动执行创建表、导入、创建索引
    auto_mode = args.fbin and os.path.exists(args.fbin)

    # 步骤 1: 创建表
    if auto_mode or args.create_table:
        ret = run_wiki_create_table(args)
        if ret != 0:
            return ret

    # 步骤 2: 导入数据
    if auto_mode:
        ret = run_wiki_import(args)
        if ret != 0:
            return ret

    # 步骤 3: 创建索引
    if auto_mode or args.create_index:
        ret = run_wiki_create_index(args)
        if ret != 0:
            return ret

    # 步骤 4: 自动测试（仅当显式指定 --auto-test 时）
    if args.auto_test:
        ret = run_wiki_test(args)
        if ret != 0:
            return ret

    print("=" * 70)
    print("Wiki 设置完成!")
    print("=" * 70)
    return 0


def run_wiki(args):
    """处理 wiki 命令"""
    if not hasattr(args, 'wiki_command') or args.wiki_command is None:
        run_wiki_info()
        return 0

    if args.wiki_command == "info":
        return run_wiki_info()
    elif args.wiki_command == "create-table":
        return run_wiki_create_table(args)
    elif args.wiki_command == "import":
        return run_wiki_import(args)
    elif args.wiki_command == "create-index":
        return run_wiki_create_index(args)
    elif args.wiki_command == "test":
        return run_wiki_test(args)
    elif args.wiki_command == "setup":
        return run_wiki_setup(args)
    else:
        print(f"未知 wiki 子命令: {args.wiki_command}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Wiki-all 向量数据集测试工具 - 支持 cuVS Bench Wiki-all 数据集（768维）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据集信息:
  名称: cuVS Bench Wiki-all
  来源: https://github.com/rapidsai/cuvs
  维度: 768
  格式: .fbin (float32 binary)

示例:
  # 显示数据集信息
  python run_vector_test.py wiki info

  # 创建表
  python run_vector_test.py wiki create-table --database jst_app_wiki

  # 导入 .fbin 数据
  python run_vector_test.py wiki import --fbin /path/to/wiki_all_1M.fbin

  # 创建向量索引
  python run_vector_test.py wiki create-index --ivf-lists 100

  # 运行测试
  python run_vector_test.py wiki test -n 1000 -k 10 --concurrency 4

  # 生成 ANN 文件
  python run_vector_test.py ann --sql-mode l2_only -n 1000 -k 10

  # 运行评估
  python run_vector_test.py run --sql-mode l2_filter --filter-val 20000000 -n 1000 -k 10 --concurrency 100

  # 一键完整流程（自动创建表、导入数据、创建索引）
  python run_vector_test.py wiki setup --fbin /path/to/wiki_all_1M.fbin --ivf-lists 100

  # 一键流程+测试
  python run_vector_test.py wiki setup --fbin /path/to/wiki_all_1M.fbin --ivf-lists 100 --auto-test -n 1000
        """,
    )

    # 全局参数
    parser.add_argument("--host", default="127.0.0.1", help="数据库主机（默认: 127.0.0.1）")
    parser.add_argument("--port", type=int, default=6001, help="端口（默认: 6001）")
    parser.add_argument("--user", default="dump", help="用户名（默认: dump）")
    parser.add_argument("--password", default="111", help="密码（默认: 111）")
    parser.add_argument("--database", default="jst_app_wiki", help="数据库名（默认: jst_app_wiki）")
    parser.add_argument("--table", default="historical_file_blocks_wiki", help="表名（默认: historical_file_blocks_wiki）")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ===== wiki 命令 =====
    wiki_parser = subparsers.add_parser(
        "wiki",
        help="Wiki 数据集导入与测试 (cuVS Bench Wiki-all, VECF32(768))",
    )

    # wiki 子命令
    wiki_subparsers = wiki_parser.add_subparsers(dest="wiki_command", help="Wiki 子命令")

    # wiki info
    wiki_subparsers.add_parser("info", help="显示 Wiki 数据集信息")

    # wiki create-table
    wiki_subparsers.add_parser("create-table", help="创建 Wiki 向量表 (embedding VECF32(768))")

    # wiki import
    wiki_import_parser = wiki_subparsers.add_parser("import", help="导入 Wiki .fbin 数据")
    wiki_import_parser.add_argument("--fbin", required=True, help=".fbin 向量文件路径")
    wiki_import_parser.add_argument("--batch-size", type=int, default=20000, help="批量导入大小（默认: 20000）")
    wiki_import_parser.add_argument("--file-id-base", type=int, default=20000000, help="file_id 起始值（默认: 20000000）")

    # wiki create-index
    wiki_idx_parser = wiki_subparsers.add_parser("create-index", help="创建 Wiki 向量索引")
    wiki_idx_parser.add_argument("--ivf-lists", type=int, default=1000, help="IVF lists 数量（默认: 1000）")

    # wiki test
    wiki_test_parser = wiki_subparsers.add_parser("test", help="运行 Wiki 向量搜索测试")
    wiki_test_parser.add_argument("-n", "--num-queries", type=int, default=1000, help="查询数量（默认: 1000）")
    wiki_test_parser.add_argument("-k", "--k", type=int, default=10, help="Top-K（默认: 10）")
    wiki_test_parser.add_argument("--concurrency", type=int, default=4, help="并发数（默认: 4）")
    wiki_test_parser.add_argument("--sql-mode", choices=["l2_only", "l2_filter", "l2_filter_threshold"], default="l2_only", help="SQL 模式（默认: l2_only）")
    wiki_test_parser.add_argument("--filter-val", type=int, help="file_id 过滤值（用于 l2_filter 和 l2_filter_threshold 模式）")

    # wiki setup (一键设置)
    wiki_setup_parser = wiki_subparsers.add_parser("setup", help="一键设置：创建表、导入数据、创建索引（只需 --fbin 即可自动执行前三步）")
    wiki_setup_parser.add_argument("--fbin", help=".fbin 向量文件路径（指定后自动执行创建表、导入、建索引）")
    wiki_setup_parser.add_argument("--create-table", action="store_true", help="显式创建表（--fbin 时自动执行）")
    wiki_setup_parser.add_argument("--create-index", action="store_true", help="显式创建向量索引（--fbin 时自动执行）")
    wiki_setup_parser.add_argument("--ivf-lists", type=int, default=1000, help="IVF lists 数量（默认: 1000）")
    wiki_setup_parser.add_argument("--auto-test", action="store_true", help="设置完成后自动运行测试")
    wiki_setup_parser.add_argument("-n", "--num-queries", type=int, default=1000, help="测试查询数量（默认: 1000）")
    wiki_setup_parser.add_argument("-k", "--k", type=int, default=10, help="Top-K（默认: 10）")
    wiki_setup_parser.add_argument("--concurrency", type=int, default=4, help="并发数（默认: 4）")
    wiki_setup_parser.add_argument("--batch-size", type=int, default=20000, help="批量导入大小（默认: 20000）")
    wiki_setup_parser.add_argument("--file-id-base", type=int, default=20000000, help="file_id 起始值（默认: 20000000）")

    # ===== ann 命令 =====
    ann_parser = subparsers.add_parser("ann", help="生成 ANN 评测文件")
    ann_parser.add_argument("--sql-mode", choices=["l2_only", "l2_filter", "l2_filter_threshold"], default="l2_only", help="SQL 模式（默认: l2_only）")
    ann_parser.add_argument("-n", "--num-queries", type=int, default=1000, help="查询数量（默认: 1000）")
    ann_parser.add_argument("-k", "--k", type=int, default=10, help="Top-K（默认: 10）")
    ann_parser.add_argument("--concurrency", type=int, default=1, help="并发数（默认: 1）")
    ann_parser.add_argument("--filter-val", type=int, help="file_id 过滤值（用于 l2_filter 模式）")
    ann_parser.add_argument("--distribute-file-ids", action="store_true", help="将查询分布到多个不同的 file_id")
    ann_parser.add_argument("--max-distinct-file-ids", type=int, default=50, help="最多使用多少个不同的 file_id")

    # ===== run 命令 =====
    run_parser = subparsers.add_parser("run", help="运行召回率/QPS 评估")
    run_parser.add_argument("--sql-mode", choices=["l2_only", "l2_filter", "l2_filter_threshold"], default="l2_only", help="SQL 模式（默认: l2_only）")
    run_parser.add_argument("-n", "--num-queries", type=int, default=1000, help="查询数量（默认: 1000）")
    run_parser.add_argument("-k", "--k", type=int, default=10, help="Top-K（默认: 10）")
    run_parser.add_argument("--concurrency", type=int, default=1, help="并发数（默认: 1）")
    run_parser.add_argument("--filter-val", type=int, help="file_id 过滤值（用于 l2_filter 模式）")
    run_parser.add_argument("--duration", type=float, help="持续时间（秒），用于压测模式")
    run_parser.add_argument("--distribute-file-ids", action="store_true", help="将查询分布到多个不同的 file_id")
    run_parser.add_argument("--max-distinct-file-ids", type=int, default=50, help="最多使用多少个不同的 file_id")
    run_parser.add_argument("--skip-db-verify", action="store_true", help="跳过数据库预检")
    run_parser.add_argument("--probe", type=int, help="设置 probe_limit 值（用于 IVF 索引查询）")
    run_parser.add_argument("--filter-mode", choices=["pre", "post", "force"], help="SQL 后缀模式：pre（预过滤）、post（后过滤）、force（强制精确搜索）")

    args = parser.parse_args()

    if args.command == "wiki":
        return run_wiki(args)
    elif args.command == "ann":
        return run_ann(args)
    elif args.command == "run":
        return run_eval(args)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())

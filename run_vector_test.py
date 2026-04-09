#!/usr/bin/env python3
"""
向量搜索测试入口脚本 - 支持可配置 SQL 模式

调用现有的两个脚本:
- generate_historical_file_blocks.py: 生成测试数据
- eval_vector_search_from_table.py: 向量搜索评估

支持从配置文件加载自定义 SQL 模式。

配置文件: sql_config_simple.json

配置文件格式:
  {
    "sql_modes": {
      "m1_l2_only": {
        "type": "l2_only",
        "sql": "SELECT ... ORDER BY l2_distance(...) LIMIT %s"
      },
      "m2_l2_filter": {
        "type": "l2_filter",
        "sql": "SELECT ... WHERE file_id = %s ORDER BY ... LIMIT %s"
      },
      "m3_l2_filter_threshold": {
        "type": "l2_filter_threshold",
        "sql": "SELECT ... WHERE file_id = %s AND l2_distance(...) <= {max_distance} ...",
        "extra": { "max_distance": 1.77 }
      }
    },
    "default": {
      "table": "historical_file_blocks",
      "emb_col": "embedding",
      "filter_col": "file_id"
    }
  }

type 说明:
  - l2_only:               全表向量搜索，不需要 file_id
  - l2_filter:             预过滤后向量搜索，需要 file_id
  - l2_filter_threshold:   预过滤+距离阈值，需要 file_id

filter_mode 说明:
  - pre:   Ground Truth SQL 添加 'BY RANK WITH OPTION 'mode=pre''
  - force: Ground Truth SQL 添加 'BY RANK WITH OPTION 'mode=force''（默认）
  - post:  不添加任何后缀

示例用法:
  # 生成 100万行数据
  python run_vector_test.py generate -n 1m --distinct-file-ids 100

  # 列出所有可用的 SQL 模式
  python run_vector_test.py list-modes

  # 使用 m1_l2_only 模式生成 ANN 文件
  python run_vector_test.py ann --sql-mode m1_l2_only -n 1000 -k 10

  # 使用 m2_l2_filter 模式，指定 filter_mode=force
  python run_vector_test.py ann --sql-mode m2_l2_filter --filter-mode force -n 1000 -k 10

  # 使用 m3_l2_filter_threshold 模式评估
  python run_vector_test.py eval --sql-mode m3_l2_filter_threshold -n 1000 -k 10 --concurrency 4

  # 完整测试流程（多模式对比）
  python run_vector_test.py run -n 100k --sql-modes m1_l2_only m2_l2_filter m3_l2_filter_threshold

  # 指定数据库连接信息
  python run_vector_test.py ann --sql-mode m2_l2_filter -n 1000 -k 10 \\
    --host 127.0.0.1 --port 6001 --user dump --password 111 \\
    --database jst_app --table historical_file_blocks

  # 使用自定义配置文件
  python run_vector_test.py ann --config my_config.json --sql-mode custom -n 1000

  # 初始化环境：创建数据库、创建表、导入数据、创建向量索引（IVF）
  python run_vector_test.py init --database jst_app --table historical_file_blocks \\
    --data-file data.csv --create-db --create-table --create-index --index-type ivfflat

  # 一键完整流程：初始化 + 测试（自动生成数据、创建表、导入、建IVF索引、测试）
  python run_vector_test.py init -n 100k --distinct-file-ids 50 \\
    --auto-generate --create-db --create-table --create-index --index-type ivfflat \\
    --sql-modes m1_l2_only m2_l2_filter --auto-run --concurrency 4
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Any


# 脚本路径（与当前脚本在同一目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATE_SCRIPT = os.path.join(SCRIPT_DIR, "generate_historical_file_blocks.py")
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_vector_search_from_table.py")
DEFAULT_CONFIG_FILE = os.path.join(SCRIPT_DIR, "sql_config_simple.json")


def check_scripts():
    """检查依赖脚本是否存在"""
    missing = []
    if not os.path.exists(GENERATE_SCRIPT):
        missing.append(os.path.basename(GENERATE_SCRIPT))
    if not os.path.exists(EVAL_SCRIPT):
        missing.append(os.path.basename(EVAL_SCRIPT))
    if missing:
        print(f"错误: 找不到依赖脚本: {', '.join(missing)}")
        print(f"请确保这些脚本与 {os.path.basename(__file__)} 在同一目录")
        sys.exit(1)


def load_sql_config(config_path: Optional[str] = None) -> Dict:
    """加载 SQL 配置文件，并统一格式（支持精简版和完整版）"""
    path = config_path or DEFAULT_CONFIG_FILE
    if not os.path.exists(path):
        print(f"错误: 配置文件不存在: {path}")
        print(f"请创建配置文件或指定正确的路径")
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 统一格式（处理精简版字段名）
    config = normalize_config(config)

    return config


def list_sql_modes(config: Dict) -> None:
    """列出所有可用的 SQL 模式"""
    print("=" * 70)
    print("可用的 SQL 模式")
    print("=" * 70)

    modes = config.get("sql_modes", {})
    default = config.get("default_config", config.get("default", {}))

    print(f"\n默认配置:")
    print(f"  表名: {default.get('table', 'historical_file_blocks')}")
    print(f"  向量列: {default.get('emb_col', 'embedding')}")
    print(f"  过滤列: {default.get('filter_col', 'file_id')}")

    print(f"\nSQL 模式列表:")
    for mode_id, mode_config in modes.items():
        name = mode_config.get("name", mode_id)  # 精简版使用 mode_id 作为名称
        desc = mode_config.get("description", "")  # 精简版可能为空
        mode_type = mode_config.get("type", "unknown")
        sql = mode_config.get("sql_template", mode_config.get("sql", ""))[:80]
        print(f"\n  {mode_id}:")
        if "name" in mode_config:
            print(f"    名称: {name}")
        print(f"    类型: {mode_type}")
        if "description" in mode_config:
            print(f"    描述: {desc}")
        print(f"    SQL: {sql}...")
        # 显示额外参数（兼容 extra 和 extra_args）
        extra = mode_config.get("extra_args") or mode_config.get("extra")
        if extra:
            print(f"    额外参数: {extra}")

    print(f"\n共 {len(modes)} 个模式")
    print(f"\n使用说明:")
    print(f"  1. SQL 配置文件中只定义基础 SQL（无后缀）")
    print(f"  2. 通过 --filter-mode 参数指定 Ground Truth SQL 后缀:")
    print(f"     pre   - 添加 'BY RANK WITH OPTION 'mode=pre'")
    print(f"     force - 添加 'BY RANK WITH OPTION 'mode=force'")
    print(f"     post  - 不添加任何后缀（默认）")
    print("=" * 70)


def get_sql_mode_config(config: Dict, mode_id: str) -> Optional[Dict]:
    """获取指定 SQL 模式的配置"""
    modes = config.get("sql_modes", {})
    return modes.get(mode_id)


def apply_filter_mode(sql: str, filter_mode: str) -> str:
    """
    根据 filter_mode 添加 SQL 后缀
    - pre: 添加 BY RANK WITH OPTION 'mode=pre'
    - force: 添加 BY RANK WITH OPTION 'mode=force'
    - post: 不添加任何后缀
    """
    if filter_mode == "pre":
        return sql.rstrip().rstrip(";") + " BY RANK WITH OPTION 'mode=pre';"
    elif filter_mode == "force":
        return sql.rstrip().rstrip(";") + " BY RANK WITH OPTION 'mode=force';"
    else:  # post 或其他值
        return sql


def normalize_config(config: Dict) -> Dict:
    """
    统一配置文件格式，支持精简版和完整版
    字段映射:
    - sql_template -> sql
    - extra_args -> extra
    - default_config -> default
    - params -> 自动推断（如果不存在）
    """
    # 处理顶层 default 配置
    if "default" in config and "default_config" not in config:
        config["default_config"] = config["default"]

    # 处理 sql_modes
    modes = config.get("sql_modes", {})
    for mode_id, mode in modes.items():
        # sql_template <-> sql
        if "sql" in mode and "sql_template" not in mode:
            mode["sql_template"] = mode["sql"]
        # extra <-> extra_args
        if "extra" in mode and "extra_args" not in mode:
            mode["extra_args"] = mode["extra"]
        # 如果没有 params，根据 SQL 中的 %s 数量推断
        if "params" not in mode:
            sql = mode.get("sql_template", mode.get("sql", ""))
            count = sql.count("%s")
            # 根据 type 推断参数名称（现在 type 名称和 eval 脚本一致）
            mode_type = mode.get("type", "unknown")
            if mode_type == "l2_only":
                mode["params"] = ["query_vec", "k"]
            elif mode_type in ["l2_filter", "custom"]:
                mode["params"] = ["query_vec", "filter_val", "k"]
            elif mode_type == "l2_filter_threshold":
                mode["params"] = ["query_vec", "filter_val", "query_vec", "k"]
            else:
                # 通用推断：根据 %s 数量
                if count == 2:
                    mode["params"] = ["query_vec", "k"]
                elif count == 3:
                    mode["params"] = ["query_vec", "filter_val", "k"]
                elif count >= 4:
                    mode["params"] = ["query_vec", "filter_val", "query_vec", "k"][:count]
                else:
                    mode["params"] = []

    return config


def build_sql_from_config(
    mode_config: Dict,
    default_config: Dict,
    table: Optional[str] = None,
    for_ground_truth: bool = False,
    filter_mode: str = "post",
    **kwargs
) -> tuple:
    """
    根据配置构建 SQL 和参数
    返回: (sql, params, mode_type)

    参数:
    - for_ground_truth: 是否构建 ground truth SQL（会添加 filter_mode 后缀）
    - filter_mode: pre/force/post，仅当 for_ground_truth=True 时生效
    """
    # 处理字段名别名（精简版 vs 完整版）
    sql_template = mode_config.get("sql_template") or mode_config.get("sql", "")
    mode_type = mode_config.get("type", "unknown")
    params_template = mode_config.get("params", [])
    extra_args = mode_config.get("extra_args") or mode_config.get("extra", {})

    # 合并默认配置和传入参数
    template_vars = {
        "table": table or default_config.get("table", "historical_file_blocks"),
        "emb_col": default_config.get("emb_col", "embedding"),
        "filter_col": default_config.get("filter_col", "file_id"),
        "pk_col": default_config.get("pk_col", "id"),
    }

    # 处理 extra_args 中的变量
    for key, value in extra_args.items():
        template_vars[key] = value

    # 填充 SQL 模板
    sql = sql_template.format(**template_vars)

    # 根据 filter_mode 添加后缀（仅 ground truth 查询需要）
    if for_ground_truth:
        sql = apply_filter_mode(sql, filter_mode)

    return sql, params_template, mode_type


def run_generate(args):
    """调用 generate_historical_file_blocks.py 生成数据"""
    cmd = [sys.executable, GENERATE_SCRIPT]

    # 行数
    if args.rows:
        cmd.extend(["-n", str(args.rows)])

    # 输出文件
    if args.output:
        cmd.extend(["-o", args.output])

    # 不同的 file_id 数量
    if hasattr(args, 'distinct_file_ids') and args.distinct_file_ids != 50:
        cmd.extend(["--distinct-file-ids", str(args.distinct_file_ids)])

    # 随机种子
    if hasattr(args, 'seed') and args.seed != 42:
        cmd.extend(["--seed", str(args.seed)])

    # 表头
    if hasattr(args, 'with_header') and args.with_header:
        cmd.extend(["--with-header"])

    # 5列格式
    if hasattr(args, 'five_column') and args.five_column:
        cmd.extend(["--five-column"])

    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def validate_type(mode_type: str) -> str:
    """验证 type 是否合法，返回原值（现在 type 名称和 eval 脚本一致，不需要映射）"""
    valid_types = ["l2_only", "l2_filter", "l2_filter_threshold", "custom"]
    if mode_type not in valid_types:
        print(f"警告: 未知的 type '{mode_type}'，将使用 'l2_filter'")
        return "l2_filter"
    return mode_type


def run_ann_with_config(args, config: Dict):
    """
    使用配置文件中的 SQL 模式生成 ANN 文件
    """
    mode_id = args.sql_mode
    mode_config = get_sql_mode_config(config, mode_id)

    if not mode_config:
        print(f"错误: 未知的 SQL 模式: {mode_id}")
        print(f"可用模式: {', '.join(config.get('sql_modes', {}).keys())}")
        return 1

    # 获取 filter_mode 参数
    filter_mode = getattr(args, 'filter_mode', 'post')

    print(f"使用 SQL 模式: {mode_id}")
    print(f"  名称: {mode_config.get('name', '')}")
    print(f"  类型: {mode_config.get('type', '')}")
    print(f"  描述: {mode_config.get('description', '')}")

    # 构建 SQL（用于显示和调试）
    default_config = config.get("default_config", {})
    table = getattr(args, 'table', None) or default_config.get('table')

    # 显示普通 SQL（不带后缀）
    sql_normal, params_template, mode_type = build_sql_from_config(
        mode_config, default_config, table=table, for_ground_truth=False
    )
    # 显示 ground truth SQL（带 filter_mode 后缀）
    sql_gt, _, _ = build_sql_from_config(
        mode_config, default_config, table=table, for_ground_truth=True, filter_mode=filter_mode
    )

    print(f"\nSQL 模板 (普通查询):")
    print(f"  {sql_normal[:100]}...")
    print(f"\nSQL 模板 (Ground Truth, filter_mode={filter_mode}):")
    print(f"  {sql_gt[:100]}...")
    print(f"  参数: {params_template}")

    # type 名称已经和 eval 脚本一致，直接使用
    base_mode = validate_type(mode_type)

    cmd = [sys.executable, EVAL_SCRIPT]
    cmd.extend(["--mode", base_mode])
    cmd.extend(["--k", str(args.k)])
    cmd.extend(["--num-queries", str(args.num_queries)])
    cmd.append("--annfiles-only")
    cmd.append("--write-ann-files")
    cmd.extend(["--concurrency", str(args.concurrency)])

    if args.database:
        cmd.extend(["--database", args.database])

    if args.filter_val:
        cmd.extend(["--mode23-filter", str(args.filter_val)])

    if args.distribute_file_ids:
        cmd.append("--ann-distribute-file-ids")
        if args.max_distinct_file_ids != 50:
            cmd.extend(["--ann-max-distinct-file-ids", str(args.max_distinct_file_ids)])

    print(f"\n执行: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    # 重命名输出文件以标识 SQL 模式
    if result.returncode == 0:
        rename_ann_files(mode_id, args.k)

    return result.returncode


def run_ann(args):
    """
    生成 ANN 文件
    支持配置文件模式或内置模式
    """
    config = None
    if hasattr(args, 'config') and args.config:
        config = load_sql_config(args.config)
    elif os.path.exists(DEFAULT_CONFIG_FILE):
        config = load_sql_config()

    # 如果指定了 sql_mode 且有配置文件，使用配置模式
    if hasattr(args, 'sql_mode') and args.sql_mode and config:
        return run_ann_with_config(args, config)

    # 否则使用传统的 --mode 参数
    mode_map = {
        "prefilter": "l2_filter",
        "postfilter": "l2_filter",
        "threshold": "l2_filter_threshold",
        "l2_only": "l2_only",
        "l2_filter": "l2_filter",
        "l2_filter_threshold": "l2_filter_threshold",
    }

    cmd = [sys.executable, EVAL_SCRIPT]

    if args.mode in ["prefilter", "postfilter"]:
        base_mode = "l2_filter"
    elif args.mode == "threshold":
        base_mode = "l2_filter_threshold"
    elif args.mode in mode_map:
        base_mode = mode_map[args.mode]
    else:
        base_mode = args.mode

    cmd.extend(["--mode", base_mode])
    cmd.extend(["--k", str(args.k)])
    cmd.extend(["--num-queries", str(args.num_queries)])
    cmd.append("--annfiles-only")
    cmd.append("--write-ann-files")
    cmd.extend(["--concurrency", str(args.concurrency)])

    if args.database:
        cmd.extend(["--database", args.database])

    if args.filter_val:
        cmd.extend(["--mode23-filter", str(args.filter_val)])

    if args.distribute_file_ids:
        cmd.append("--ann-distribute-file-ids")
        if args.max_distinct_file_ids != 50:
            cmd.extend(["--ann-max-distinct-file-ids", str(args.max_distinct_file_ids)])

    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        rename_ann_files(args.mode, args.k)

    return result.returncode


def rename_ann_files(mode: str, k: int):
    """重命名 ANN 输出文件以标识 SQL 模式"""
    mode_mapping = {
        "l2_only": "l2_only",
        "l2_filter": "l2_filter",
        "l2_filter_threshold": "l2_filter_threshold",
        "prefilter": "l2_filter",
        "postfilter": "l2_filter",
        "threshold": "l2_filter_threshold",
    }

    base_mode = mode_mapping.get(mode, mode)

    suffixes = [
        (f"query_{base_mode}_k{k}.fvecs", f"query_{mode}_k{k}.fvecs"),
        (f"groundtruth_{base_mode}_k{k}.ivecs", f"groundtruth_{mode}_k{k}.ivecs"),
        (f"id_mapping_{base_mode}_k{k}.txt", f"id_mapping_{mode}_k{k}.txt"),
    ]

    for old_name, new_name in suffixes:
        if os.path.exists(old_name) and not os.path.exists(new_name):
            os.rename(old_name, new_name)
            print(f"重命名: {old_name} -> {new_name}")


def run_eval_with_config(args, config: Dict):
    """使用配置文件中的 SQL 模式执行评估"""
    mode_id = args.sql_mode
    mode_config = get_sql_mode_config(config, mode_id)

    if not mode_config:
        print(f"错误: 未知的 SQL 模式: {mode_id}")
        return 1

    filter_mode = getattr(args, 'filter_mode', 'post')

    print(f"使用 SQL 模式: {mode_id}")
    print(f"  名称: {mode_config.get('name', '')}")
    print(f"  描述: {mode_config.get('description', '')}")
    print(f"  filter_mode: {filter_mode}")

    # 显示 SQL（用于调试）
    default_config = config.get("default_config", {})
    sql_gt, _, _ = build_sql_from_config(
        mode_config, default_config,
        table=getattr(args, 'table', None),
        for_ground_truth=True, filter_mode=filter_mode
    )
    print(f"  Ground Truth SQL: {sql_gt[:100]}...")

    mode_type = mode_config.get("type", "unknown")
    # type 名称已经和 eval 脚本一致，直接使用
    base_mode = validate_type(mode_type)

    cmd = [sys.executable, EVAL_SCRIPT]
    cmd.extend(["--mode", base_mode])
    cmd.extend(["--k", str(args.k)])
    cmd.extend(["--num-queries", str(args.num_queries)])
    cmd.extend(["--concurrency", str(args.concurrency)])

    if hasattr(args, 'duration') and args.duration:
        cmd.extend(["--duration", str(args.duration)])

    if args.database:
        cmd.extend(["--database", args.database])

    if hasattr(args, 'filter_val') and args.filter_val:
        cmd.extend(["--mode23-filter", str(args.filter_val)])

    if hasattr(args, 'distribute_file_ids') and args.distribute_file_ids:
        cmd.append("--ann-distribute-file-ids")
        if hasattr(args, 'max_distinct_file_ids') and args.max_distinct_file_ids != 50:
            cmd.extend(["--ann-max-distinct-file-ids", str(args.max_distinct_file_ids)])

    if hasattr(args, 'skip_db_verify') and args.skip_db_verify:
        cmd.append("--skip-db-verify")

    if hasattr(args, 'write_ann_files') and args.write_ann_files:
        cmd.append("--write-ann-files")

    print(f"\n执行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_eval(args):
    """运行召回率评估"""
    # 检查是否有配置文件和 sql_mode
    config = None
    if hasattr(args, 'config') and args.config:
        config = load_sql_config(args.config)
    elif os.path.exists(DEFAULT_CONFIG_FILE):
        config = load_sql_config()

    if hasattr(args, 'sql_mode') and args.sql_mode and config:
        return run_eval_with_config(args, config)

    # 传统模式
    mode_map = {
        "prefilter": "l2_filter",
        "postfilter": "l2_filter",
        "threshold": "l2_filter_threshold",
        "l2_only": "l2_only",
        "l2_filter": "l2_filter",
        "l2_filter_threshold": "l2_filter_threshold",
    }

    cmd = [sys.executable, EVAL_SCRIPT]

    if args.mode in ["prefilter", "postfilter"]:
        base_mode = "l2_filter"
    elif args.mode == "threshold":
        base_mode = "l2_filter_threshold"
    elif args.mode in mode_map:
        base_mode = mode_map[args.mode]
    else:
        base_mode = args.mode

    cmd.extend(["--mode", base_mode])
    cmd.extend(["--k", str(args.k)])
    cmd.extend(["--num-queries", str(args.num_queries)])
    cmd.extend(["--concurrency", str(args.concurrency)])

    if hasattr(args, 'duration') and args.duration:
        cmd.extend(["--duration", str(args.duration)])

    if args.database:
        cmd.extend(["--database", args.database])

    if hasattr(args, 'filter_val') and args.filter_val:
        cmd.extend(["--mode23-filter", str(args.filter_val)])

    if hasattr(args, 'distribute_file_ids') and args.distribute_file_ids:
        cmd.append("--ann-distribute-file-ids")
        if hasattr(args, 'max_distinct_file_ids') and args.max_distinct_file_ids != 50:
            cmd.extend(["--ann-max-distinct-file-ids", str(args.max_distinct_file_ids)])

    if hasattr(args, 'skip_db_verify') and args.skip_db_verify:
        cmd.append("--skip-db-verify")

    if hasattr(args, 'write_ann_files') and args.write_ann_files:
        cmd.append("--write-ann-files")

    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_full_test(args):
    """
    完整测试流程:
    1. 生成数据
    2. 对不同 SQL 模式进行测试
    """
    print("=" * 70)
    print("完整向量搜索测试流程")
    print("=" * 70)

    # 加载配置
    config = None
    if hasattr(args, 'config') and args.config:
        config = load_sql_config(args.config)
    elif os.path.exists(DEFAULT_CONFIG_FILE):
        config = load_sql_config()

    # 阶段 A: 生成数据（如果 init 命令已生成则跳过）
    if getattr(args, 'skip_data_gen', False):
        print("\n[阶段 A] 跳过数据生成（init 命令已生成数据）")
    else:
        print("\n[阶段 A] 生成测试数据...")
        data_file = f"test_data_{args.rows}.csv"

        gen_args = argparse.Namespace(
            rows=args.rows,
            output=data_file,
            distinct_file_ids=args.distinct_file_ids,
            seed=42,
            with_header=False,
            five_column=False,
        )

        ret = run_generate(gen_args)
        if ret != 0:
            print("错误: 数据生成失败")
            return ret

        print(f"\n数据文件已生成: {data_file}")
        print(f"请先将数据导入数据库:")
        print(f"  LOAD DATA INFILE '{os.path.abspath(data_file)}' INTO TABLE {args.database}.{args.table} ...")
        print(f"\n确认数据已导入后，继续执行评估...")

        if hasattr(args, 'auto_continue') and not args.auto_continue:
            input("按 Enter 继续...")

    # 步骤 2 & 3: 对不同模式进行测试
    results = []
    sql_modes = args.sql_modes if hasattr(args, 'sql_modes') and args.sql_modes else ["m2_l2_filter"]

    for mode_id in sql_modes:
        print(f"\n{'='*70}")
        print(f"测试模式: {mode_id}")
        print(f"{'='*70}")

        # 创建 ann 参数
        ann_args = argparse.Namespace(
            sql_mode=mode_id if config and mode_id in config.get("sql_modes", {}) else None,
            mode="l2_filter",  # 默认回退
            k=args.k,
            num_queries=min(args.num_queries, 500),
            concurrency=1,
            database=args.database,
            table=args.table,
            filter_val=None,
            distribute_file_ids=True,
            max_distinct_file_ids=args.distinct_file_ids,
            config=getattr(args, 'config', None),
            filter_mode=getattr(args, 'filter_mode', 'post'),
        )

        print(f"\n[阶段 B] 生成 ANN 文件 ({mode_id})...")
        if config and mode_id in config.get("sql_modes", {}):
            ret = run_ann_with_config(ann_args, config)
        else:
            ret = run_ann(ann_args)

        if ret != 0:
            print(f"警告: {mode_id} 模式 ANN 文件生成失败")
            continue

        # 执行评估
        print(f"\n[阶段 C] 执行评估 ({mode_id})...")
        eval_args = argparse.Namespace(
            sql_mode=mode_id if config and mode_id in config.get("sql_modes", {}) else None,
            mode="l2_filter",
            k=args.k,
            num_queries=args.num_queries,
            concurrency=args.concurrency,
            duration=None,
            database=args.database,
            table=args.table,
            filter_val=None,
            distribute_file_ids=True,
            max_distinct_file_ids=args.distinct_file_ids,
            skip_db_verify=False,
            write_ann_files=False,
            config=getattr(args, 'config', None),
            filter_mode=getattr(args, 'filter_mode', 'post'),
        )

        if config and mode_id in config.get("sql_modes", {}):
            ret = run_eval_with_config(eval_args, config)
        else:
            ret = run_eval(eval_args)

        results.append((mode_id, ret))

    # 汇总结果
    print(f"\n{'='*70}")
    print("测试完成")
    print(f"{'='*70}")
    for mode_id, ret in results:
        status = "成功" if ret == 0 else "失败"
        print(f"  {mode_id}: {status}")

    return 0


def run_init(args):
    """
    初始化测试环境：创建数据库、创建表、导入数据、创建向量索引
    """
    import pymysql

    print("=" * 70)
    print("初始化测试环境")
    print("=" * 70)

    # 数据库配置
    db_config = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "password": args.password,
        "database": args.database,
    }

    # 步骤 1: 生成/获取数据文件
    data_file = None
    if args.auto_generate:
        print("\n[步骤 1] 自动生成数据...")
        data_file = f"{args.table}_{args.rows}.csv"
        gen_args = argparse.Namespace(
            rows=args.rows,
            output=data_file,
            distinct_file_ids=args.distinct_file_ids,
            seed=42,
            with_header=False,
            five_column=False,
        )
        ret = run_generate(gen_args)
        if ret != 0:
            print("错误: 数据生成失败")
            return ret
    elif args.data_file:
        data_file = args.data_file
        print(f"\n[步骤 1] 使用已有数据文件: {data_file}")
    else:
        print("\n[步骤 1] 跳过数据准备（请使用 --auto-generate 或 --data-file 指定数据）")

    # 步骤 2: 创建数据库
    if args.create_db:
        print(f"\n[步骤 2] 创建数据库: {args.database}")
        try:
            conn = pymysql.connect(
                host=args.host,
                port=args.port,
                user=args.user,
                password=args.password,
            )
            with conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{args.database}`")
                print(f"  数据库 {args.database} 已创建或已存在")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"错误: 创建数据库失败: {e}")
            return 1
    else:
        print(f"\n[步骤 2] 跳过创建数据库（使用 --create-db 启用）")

    # 步骤 3: 创建表
    need_create_table = args.create_table or args.data_file or args.auto_generate
    if need_create_table:
        print(f"\n[步骤 3] 创建表: {args.table}")
        try:
            conn = pymysql.connect(
                host=args.host,
                port=args.port,
                user=args.user,
                password=args.password,
                database=args.database,
            )
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS `{args.table}` (
                `id` BIGINT NOT NULL AUTO_INCREMENT,
                `file_id` BIGINT NOT NULL,
                `content` TEXT,
                `embedding` VECF32(1024),
                `page_num` INT,
                `meta` JSON,
                PRIMARY KEY (`id`)
            )
            """
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                print(f"  表 {args.table} 已创建或已存在")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"错误: 创建表失败: {e}")
            return 1
    else:
        print(f"\n[步骤 3] 跳过创建表")

    # 步骤 4: 导入数据
    if data_file and os.path.exists(data_file):
        print(f"\n[步骤 4] 导入数据: {data_file}")
        try:
            conn = pymysql.connect(
                host=args.host,
                port=args.port,
                user=args.user,
                password=args.password,
                database=args.database,
            )
            with conn.cursor() as cur:
                abs_path = os.path.abspath(data_file)
                load_sql = f"""
                LOAD DATA INFILE '{abs_path}'
                INTO TABLE `{args.table}`
                FIELDS TERMINATED BY ','
                OPTIONALLY ENCLOSED BY '"'
                LINES TERMINATED BY '\n'
                (`id`, `file_id`, `content`, `embedding`, `page_num`, `meta`)
                """
                cur.execute(load_sql)
                print(f"  数据导入完成")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"警告: 数据导入失败（可能表已有数据或不支持 LOAD DATA）: {e}")
            print(f"  请手动导入数据: mysql -e \"LOAD DATA INFILE '{abs_path}' INTO TABLE {args.database}.{args.table} ...\"")
    else:
        print(f"\n[步骤 4] 跳过数据导入")

    # 步骤 5: 创建向量索引
    if args.create_index:
        print(f"\n[步骤 5] 创建向量索引")
        try:
            conn = pymysql.connect(
                host=args.host,
                port=args.port,
                user=args.user,
                password=args.password,
                database=args.database,
            )
            # 构建 Lindb/MatrixOne 格式的索引 SQL
            # 格式: CREATE INDEX idx_l2 USING ivfflat ON table(embedding) lists=XXX op_type "vector_l2_ops"
            index_name = "idx_l2"
            index_type = args.index_type  # ivfflat
            lists_val = args.ivf_lists if args.ivf_lists else 100  # 默认 100
            op_type = getattr(args, 'op_type', 'vector_l2_ops')

            create_index_sql = f'''CREATE INDEX {index_name} USING {index_type} ON `{args.table}`(embedding) lists={lists_val} op_type "{op_type}"'''

            with conn.cursor() as cur:
                cur.execute(create_index_sql)
                print(f'  向量索引已创建: {index_name} USING {index_type} lists={lists_val} op_type "{op_type}"')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"警告: 创建向量索引失败: {e}")
            lists_val = args.ivf_lists if args.ivf_lists else 100
            op_type = getattr(args, 'op_type', 'vector_l2_ops')
            print(f'  请手动创建索引: CREATE INDEX idx_l2 USING ivfflat ON {args.table}(embedding) lists={lists_val} op_type "{op_type}"')
    else:
        print(f"\n[步骤 5] 跳过创建向量索引（使用 --create-index 启用）")

    print(f"\n{'='*70}")
    print("初始化完成")
    print(f"{'='*70}")

    # 步骤 6: 自动运行测试（如果指定）
    if args.auto_run:
        print(f"\n[步骤 6] 自动运行测试...")
        # 标记数据已生成，跳过 run_full_test 的数据生成阶段
        args.skip_data_gen = True
        return run_full_test(args)

    return 0


def main():
    check_scripts()

    parser = argparse.ArgumentParser(
        description="向量搜索测试入口 - 支持可配置 SQL 模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置文件:
  默认配置文件: sql_config_simple.json

配置文件格式:
  {
    "sql_modes": {
      "mode_id": {
        "type": "l2_only|l2_filter|l2_filter_threshold",
        "sql": "SELECT ...",  // 基础 SQL，无后缀
        "extra": {...}        // 可选，如 max_distance
      }
    },
    "default": {
      "table": "...",
      "emb_col": "embedding",
      "filter_col": "file_id"
    }
  }

type 说明:
  l2_only               - 全表向量搜索
  l2_filter             - 预过滤后向量搜索（需要 file_id）
  l2_filter_threshold     - 预过滤+距离阈值（需要 file_id）

filter_mode 说明（仅影响 Ground Truth SQL）:
  pre   - 添加 BY RANK WITH OPTION 'mode=pre'
  force - 添加 BY RANK WITH OPTION 'mode=force'（默认）
  post  - 不添加任何后缀

示例:
  # 列出所有可用的 SQL 模式
  python run_vector_test.py list-modes

  # 使用 m1_l2_only 模式生成 ANN 文件
  python run_vector_test.py ann --sql-mode m1_l2_only -n 1000 -k 10

  # 使用 m2_l2_filter 模式，指定 filter_mode=force
  python run_vector_test.py ann --sql-mode m2_l2_filter --filter-mode force -n 1000 -k 10

  # 使用 m3_l2_filter_threshold 模式评估
  python run_vector_test.py eval --sql-mode m3_l2_filter_threshold -n 1000 -k 10 --concurrency 4

  # 指定数据库连接信息
  python run_vector_test.py ann --sql-mode m2_l2_filter -n 1000 -k 10 \\
    --host 127.0.0.1 --port 6001 --user dump --password 111 \\
    --database jst_app --table historical_file_blocks

  # 使用传统内置模式（不读取配置文件）
  python run_vector_test.py ann --mode l2_filter -n 1000 -k 10

  # 多模式对比测试（指定数据库）
  python run_vector_test.py run -n 100k --sql-modes m1_l2_only m2_l2_filter m3_l2_filter_threshold \\
    --database mydb --table mytable --user admin --password secret

  # 使用自定义配置文件
  python run_vector_test.py ann --config my_config.json --sql-mode my_custom_mode -n 1000

  # 初始化环境：创建数据库、创建表、导入数据、创建向量索引（IVF）
  python run_vector_test.py init --database jst_app --table historical_file_blocks \\
    --data-file data.csv --create-db --create-table --create-index --index-type ivf

  # 创建 IVF 索引（指定 lists 和 op_type）
  python run_vector_test.py init --database jst_app --table historical_file_blocks \\
    --create-index --index-type ivfflat --ivf-lists 100 --op-type vector_l2_ops

  # 一键完整流程：初始化 + 自动生成数据 + 测试（使用 IVF 索引）
  python run_vector_test.py init --database jst_app --table historical_file_blocks \\
    -n 100k --distinct-file-ids 50 --auto-generate --create-db --create-table \\
    --create-index --index-type ivfflat --ivf-lists 200 \\
    --auto-run --sql-modes m1_l2_only m2_l2_filter --concurrency 4
        """,
    )

    # 全局参数
    parser.add_argument("--config", help="配置文件路径（默认: sql_config_simple.json）")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ===== list-modes 命令 =====
    list_parser = subparsers.add_parser(
        "list-modes",
        help="列出所有可用的 SQL 模式",
    )

    # ===== generate 命令 =====
    gen_parser = subparsers.add_parser(
        "generate",
        help="生成测试数据",
    )
    gen_parser.add_argument(
        "-n", "--rows",
        default="1m",
        help="生成行数（支持 k/m/g/t 后缀）",
    )
    gen_parser.add_argument(
        "-o", "--output",
        default="historical_file_blocks_test.csv",
        help="输出文件路径",
    )
    gen_parser.add_argument(
        "--distinct-file-ids",
        type=int,
        default=50,
        help="不同的 file_id 数量（默认 50，起始值 20000000）",
    )
    gen_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    gen_parser.add_argument("--with-header", action="store_true", help="包含表头")
    gen_parser.add_argument("--five-column", action="store_true", help="使用 5 列格式")

    # ===== ann 命令 =====
    ann_parser = subparsers.add_parser(
        "ann",
        help="生成 ANN 评测文件",
    )
    # ann 数据库配置
    ann_db = ann_parser.add_argument_group("数据库配置")
    ann_db.add_argument("--host", default="127.0.0.1", help="数据库主机（默认: 127.0.0.1）")
    ann_db.add_argument("--port", type=int, default=6001, help="端口（默认: 6001）")
    ann_db.add_argument("--user", default="dump", help="用户名（默认: dump）")
    ann_db.add_argument("--password", default="111", help="密码（默认: 111）")
    ann_db.add_argument("--database", default="jst_app", help="数据库名（默认: jst_app）")
    ann_db.add_argument("--table", default="historical_file_blocks", help="表名（默认: historical_file_blocks）")

    ann_parser.add_argument(
        "--mode",
        choices=["prefilter", "postfilter", "threshold", "l2_only", "l2_filter", "l2_filter_threshold"],
        help="内置过滤模式（与 --sql-mode 互斥）",
    )
    ann_parser.add_argument(
        "--sql-mode",
        help="配置文件中的 SQL 模式 ID（如 m2_prefilter）",
    )
    ann_parser.add_argument(
        "-n", "--num-queries",
        type=int,
        default=1000,
        help="查询数量（默认: 1000）",
    )
    ann_parser.add_argument(
        "-k", "--k",
        type=int,
        default=10,
        help="Top-K（默认: 10）",
    )
    ann_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发数（默认: 1）",
    )
    ann_parser.add_argument(
        "--filter-mode",
        choices=["pre", "post", "force"],
        default="post",
        help="Ground Truth SQL 后缀模式（默认: post）",
    )
    ann_parser.add_argument("--filter-val", help="file_id 过滤值")
    ann_parser.add_argument(
        "--distribute-file-ids",
        action="store_true",
        help="将查询分布到多个不同的 file_id",
    )
    ann_parser.add_argument(
        "--max-distinct-file-ids",
        type=int,
        default=50,
        help="最多使用多少个不同的 file_id",
    )

    # ===== eval 命令 =====
    eval_parser = subparsers.add_parser(
        "eval",
        help="运行召回率/QPS 评估",
    )
    # eval 数据库配置
    eval_db = eval_parser.add_argument_group("数据库配置")
    eval_db.add_argument("--host", default="127.0.0.1", help="数据库主机（默认: 127.0.0.1）")
    eval_db.add_argument("--port", type=int, default=6001, help="端口（默认: 6001）")
    eval_db.add_argument("--user", default="dump", help="用户名（默认: dump）")
    eval_db.add_argument("--password", default="111", help="密码（默认: 111）")
    eval_db.add_argument("--database", default="jst_app", help="数据库名（默认: jst_app）")
    eval_db.add_argument("--table", default="historical_file_blocks", help="表名（默认: historical_file_blocks）")

    eval_parser.add_argument(
        "--mode",
        choices=["prefilter", "postfilter", "threshold", "l2_only", "l2_filter", "l2_filter_threshold"],
        help="内置过滤模式",
    )
    eval_parser.add_argument(
        "--sql-mode",
        help="配置文件中的 SQL 模式 ID",
    )
    eval_parser.add_argument(
        "-n", "--num-queries",
        type=int,
        default=1000,
        help="查询数量",
    )
    eval_parser.add_argument(
        "-k", "--k",
        type=int,
        default=10,
        help="Top-K",
    )
    eval_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发数",
    )
    eval_parser.add_argument(
        "--filter-mode",
        choices=["pre", "post", "force"],
        default="post",
        help="Ground Truth SQL 后缀模式（默认: post）",
    )
    eval_parser.add_argument(
        "--duration",
        type=float,
        help="持续时间（秒），用于压测模式",
    )
    eval_parser.add_argument("--filter-val", help="file_id 过滤值")
    eval_parser.add_argument(
        "--distribute-file-ids",
        action="store_true",
        help="将查询分布到多个不同的 file_id",
    )
    eval_parser.add_argument(
        "--max-distinct-file-ids",
        type=int,
        default=50,
        help="最多使用多少个不同的 file_id",
    )
    eval_parser.add_argument(
        "--skip-db-verify",
        action="store_true",
        help="跳过数据库预检",
    )
    eval_parser.add_argument(
        "--write-ann-files",
        action="store_true",
        help="同时导出 ANN 文件",
    )

    # ===== init 命令 =====
    init_parser = subparsers.add_parser(
        "init",
        help="初始化测试环境：创建数据库、创建表、导入数据、创建向量索引",
    )
    # init 数据库配置
    init_db = init_parser.add_argument_group("数据库配置")
    init_db.add_argument("--host", default="127.0.0.1", help="数据库主机（默认: 127.0.0.1）")
    init_db.add_argument("--port", type=int, default=6001, help="端口（默认: 6001）")
    init_db.add_argument("--user", default="dump", help="用户名（默认: dump）")
    init_db.add_argument("--password", default="111", help="密码（默认: 111）")
    init_db.add_argument("--database", default="jst_app", help="数据库名（默认: jst_app）")
    init_db.add_argument("--table", default="historical_file_blocks", help="表名（默认: historical_file_blocks）")

    init_parser.add_argument(
        "--create-db",
        action="store_true",
        help="创建数据库（如果不存在）",
    )
    init_parser.add_argument(
        "--create-table",
        action="store_true",
        help="创建表（如果指定了 --auto-load 或 --data-file 则自动创建）",
    )
    init_parser.add_argument(
        "--data-file",
        help="CSV 数据文件路径（用于 LOAD DATA）",
    )
    init_parser.add_argument(
        "--auto-generate",
        action="store_true",
        help="自动生成数据（与 generate 命令相同参数）",
    )
    init_parser.add_argument(
        "-n", "--rows",
        default="100k",
        help="生成数据行数（与 --auto-generate 一起使用）",
    )
    init_parser.add_argument(
        "--distinct-file-ids",
        type=int,
        default=50,
        help="不同的 file_id 数量（默认 50）",
    )
    init_parser.add_argument(
        "--create-index",
        action="store_true",
        help="创建向量索引",
    )
    init_parser.add_argument(
        "--index-type",
        choices=["ivfflat"],
        default="ivfflat",
        help="向量索引类型（默认: ivfflat）",
    )
    init_parser.add_argument(
        "--ivf-lists",
        type=int,
        default=None,
        help="IVF 索引的 lists 数量（聚类中心数）",
    )
    init_parser.add_argument(
        "--op-type",
        type=str,
        default="vector_l2_ops",
        help="向量操作类型（默认: vector_l2_ops）",
    )
    init_parser.add_argument(
        "--auto-run",
        action="store_true",
        help="初始化完成后自动运行测试",
    )
    init_parser.add_argument(
        "--sql-modes",
        nargs="+",
        default=["m1_l2_only", "m2_l2_filter"],
        help="自动运行时的 SQL 模式列表",
    )
    init_parser.add_argument(
        "--num-queries",
        type=int,
        default=1000,
        help="自动运行时的查询数",
    )
    init_parser.add_argument(
        "-k", "--k",
        type=int,
        default=10,
        help="自动运行时的 Top-K",
    )
    init_parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="自动运行时的并发数",
    )

    # ===== run 命令 =====
    run_parser = subparsers.add_parser(
        "run",
        help="完整测试流程",
    )
    # run 数据库配置
    run_db = run_parser.add_argument_group("数据库配置")
    run_db.add_argument("--host", default="127.0.0.1", help="数据库主机（默认: 127.0.0.1）")
    run_db.add_argument("--port", type=int, default=6001, help="端口（默认: 6001）")
    run_db.add_argument("--user", default="dump", help="用户名（默认: dump）")
    run_db.add_argument("--password", default="111", help="密码（默认: 111）")
    run_db.add_argument("--database", default="jst_app", help="数据库名（默认: jst_app）")
    run_db.add_argument("--table", default="historical_file_blocks", help="表名（默认: historical_file_blocks）")

    run_parser.add_argument(
        "-n", "--rows",
        default="100k",
        help="生成数据行数",
    )
    run_parser.add_argument(
        "--distinct-file-ids",
        type=int,
        default=50,
        help="不同的 file_id 数量",
    )
    run_parser.add_argument(
        "--sql-modes",
        nargs="+",
        default=["m2_prefilter"],
        help="要测试的 SQL 模式列表（从配置文件中选择）",
    )
    run_parser.add_argument(
        "--num-queries",
        type=int,
        default=1000,
        help="评估查询数",
    )
    run_parser.add_argument(
        "-k", "--k",
        type=int,
        default=10,
        help="Top-K",
    )
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="评估并发数",
    )
    run_parser.add_argument(
        "--filter-mode",
        choices=["pre", "post", "force"],
        default="post",
        help="Ground Truth SQL 后缀模式（默认: post）",
    )
    run_parser.add_argument(
        "--auto-continue",
        action="store_true",
        help="自动继续（不等待用户确认）",
    )

    args = parser.parse_args()

    if args.command == "list-modes":
        config = load_sql_config(getattr(args, 'config', None))
        list_sql_modes(config)
    elif args.command == "generate":
        return run_generate(args)
    elif args.command == "ann":
        return run_ann(args)
    elif args.command == "eval":
        return run_eval(args)
    elif args.command == "run":
        return run_full_test(args)
    elif args.command == "init":
        return run_init(args)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())

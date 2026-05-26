#!/usr/bin/env python3
"""Wiki 基准测试全流程流水线（清库建表 → 导入 → 建索引 → recall）。"""

from __future__ import annotations

import glob
import os
import time
from typing import Any, Tuple

from gen import load_csv_into_matrixone, load_s3_into_matrixone
from s3_credentials import DEFAULT_S3_CREDENTIALS_FILE, load_s3_credentials


def resolve_s3_config(args: Any, cfg: dict) -> dict | None:
    """合并 cfg.dataset.s3、CLI 与 cfg/s3_credentials.json。"""
    ds = cfg.get("dataset", {}) or {}
    s3_cfg = dict(ds.get("s3") or {})

    cli_map = {
        "endpoint": getattr(args, "s3_endpoint", None),
        "bucket": getattr(args, "s3_bucket", None),
        "filepath": getattr(args, "s3_filepath", None),
        "region": getattr(args, "s3_region", None),
        "compression": getattr(args, "s3_compression", None),
        "access_key_id": getattr(args, "s3_access_key_id", None),
        "secret_access_key": getattr(args, "s3_secret_access_key", None),
    }
    for k, v in cli_map.items():
        if v is not None and v != "":
            s3_cfg[k] = v

    cred_file = getattr(args, "s3_credentials_file", None) or DEFAULT_S3_CREDENTIALS_FILE
    file_creds = load_s3_credentials(cred_file)

    if not s3_cfg.get("access_key_id"):
        s3_cfg["access_key_id"] = (
            file_creds.get("access_key_id") or os.environ.get("MO_S3_ACCESS_KEY_ID")
        )
    if not s3_cfg.get("secret_access_key"):
        s3_cfg["secret_access_key"] = (
            file_creds.get("secret_access_key") or os.environ.get("MO_S3_SECRET_ACCESS_KEY")
        )

    if not any(s3_cfg.get(k) for k in ("endpoint", "bucket", "filepath")):
        return None
    return s3_cfg


def resolve_input_csvs(args: Any) -> list[str] | None:
    if getattr(args, "csv", None):
        return [args.csv]
    prefix = getattr(args, "input_csv_prefix", None)
    if prefix:
        return sorted(glob.glob(f"{prefix}*.csv"))
    return None


def attach_dataset_fields(args: Any, cfg: dict) -> None:
    """从 cfg.dataset 填充导入/召回相关 args 字段。"""
    dataset = cfg.get("dataset", {}) or {}
    raw_fbin = dataset.get("base_fbin")
    if raw_fbin is None:
        args.fbin = getattr(args, "fbin", None)
    elif isinstance(raw_fbin, str):
        args.fbin = [raw_fbin]
    else:
        args.fbin = list(raw_fbin)

    args.s3 = resolve_s3_config(args, cfg)

    if not hasattr(args, "batch_size") or args.batch_size is None:
        args.batch_size = 20000
    if not hasattr(args, "file_id_base") or args.file_id_base is None:
        args.file_id_base = 20000000

    env = cfg.get("env", {}) or {}
    if getattr(args, "probe", None) is None:
        args.probe = env.get("probe_limit")
    if not getattr(args, "skip_db_verify", False):
        args.skip_db_verify = True


def validate_import_paths(args: Any) -> int:
    if getattr(args, "s3", None):
        required = (
            "endpoint",
            "access_key_id",
            "secret_access_key",
            "bucket",
            "filepath",
            "region",
        )
        missing = [k for k in required if not args.s3.get(k)]
        if missing:
            print(
                "错误: S3 导入缺少参数: "
                + ", ".join(missing)
                + "。请在 cfg.dataset.s3、cfg/s3_credentials.json（见 s3_credentials.example.json）、"
                "CLI --s3-* 或环境变量中配置密钥。"
            )
            return 1
        return 0

    csvs = resolve_input_csvs(args)
    if csvs is not None:
        if not csvs:
            print(
                f"错误: --input-csv-prefix 未匹配到 "
                f"{args.input_csv_prefix}*.csv"
            )
            return 1
        for p in csvs:
            if not os.path.exists(p):
                print(f"错误: CSV 文件不存在: {p}")
                return 1
        return 0

    fbin = getattr(args, "fbin", None)
    if not fbin:
        print(
            "错误: 未配置导入源。请在 cfg.dataset.s3 中设置 S3，"
            "或 dataset.base_fbin / --csv / --input-csv-prefix。"
        )
        return 1
    for p in fbin:
        if not os.path.exists(p):
            print(f"错误: base_fbin 文件不存在: {p}")
            return 1
    return 0


def validate_recall_paths(args: Any) -> None:
    cfg = getattr(args, "_index_config", None) or {}
    ds = cfg.get("dataset", {}) or {}
    qf = ds.get("query_fbin")
    gi = ds.get("groundtruth_ibin")
    qf_ok = bool(qf) and os.path.exists(qf)
    gi_ok = bool(gi) and os.path.exists(gi)
    if not qf_ok:
        print(
            f"警告: dataset.query_fbin 未设置或不存在: {qf!r}。"
            " 召回将改走 DB 抽样 + 在线 ground truth。"
        )
    if not gi_ok:
        print(
            f"警告: dataset.groundtruth_ibin 未设置或不存在: {gi!r}。"
            " 召回将改走 DB 抽样 + 在线 ground truth。"
        )


def run_import_step(args: Any, log_prefix: str = "[pipeline]") -> Tuple[str, Any]:
    """执行数据导入（S3 > CSV > fbin INSERT）。返回 (步骤名, 可调用对象)。"""
    if getattr(args, "s3", None):

        def _do_s3() -> int:
            print(
                f"{log_prefix} LOAD DATA S3: bucket={args.s3['bucket']}, "
                f"filepath={args.s3['filepath']}",
                flush=True,
            )
            load_s3_into_matrixone(
                s3=args.s3,
                host=args.host,
                port=args.port,
                user=args.user,
                password=args.password,
                database=args.database,
                table=args.table,
            )
            return 0

        return "import (S3 LOAD DATA)", _do_s3

    csvs = resolve_input_csvs(args)
    if csvs is not None:

        def _do_csv() -> int:
            for p in csvs:
                print(f"{log_prefix} LOAD DATA: {p}", flush=True)
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

        suffix = f" x{len(csvs)}" if len(csvs) > 1 else ""
        return f"import (csv LOAD DATA{suffix})", _do_csv

    from run_vector_test import run_wiki_import

    return "import (fbin INSERT)", lambda: run_wiki_import(args)


def _banner(title: str, log_prefix: str) -> None:
    print()
    print("=" * 70)
    print(f"{log_prefix} {title}")
    print("=" * 70)


def run_all_pipeline(args: Any, log_prefix: str = "[all]") -> int:
    """清库建表 → 导入 → 建索引 → recall。"""
    cfg = getattr(args, "_index_config", None)
    if not cfg:
        print("错误: 全流程需要 --config cfg/xxx.json（含 index / env / dataset）")
        return 2

    attach_dataset_fields(args, cfg)
    if validate_import_paths(args):
        return 1
    validate_recall_paths(args)

    from run_vector_test import (
        run_eval,
        run_wiki_create_index,
        run_wiki_create_table,
    )

    timings: dict[str, float] = {}

    def _run_step(label: str, fn) -> int:
        _banner(label, log_prefix)
        t0 = time.perf_counter()
        rc = fn()
        timings[label] = time.perf_counter() - t0
        if rc:
            print(f"{log_prefix} 步骤失败: {label} (rc={rc})")
        return rc

    if _run_step("1/5  drop-db + create-table", lambda: run_wiki_create_table(args)):
        return 1

    import_label, import_fn = run_import_step(args, log_prefix=log_prefix)
    if _run_step(f"2/5  {import_label}", import_fn):
        return 1

    if _run_step("3/5  create-index", lambda: run_wiki_create_index(args)):
        return 1

    rc = _run_step("5/5  run (recall)", lambda: run_eval(args))

    _banner("完成：步骤耗时", log_prefix)
    for lbl, elapsed in timings.items():
        print(f"  {lbl:<32} {elapsed:8.2f} s")
    print(f"  {'TOTAL':<32} {sum(timings.values()):8.2f} s")
    return rc

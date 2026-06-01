#!/usr/bin/env python3
"""从 S3/OSS 下载召回评测文件（ann fvecs/ivecs 与 cuVS fbin/ibin）。"""

from __future__ import annotations

import os
from typing import Any, Optional

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

ANN_FILE_ATTRS = (
    "query_fvecs",
    "groundtruth_ivecs",
    "id_mapping",
    "query_filters",
)

FBIN_FILE_ATTRS = (
    "query_fbin",
    "groundtruth_ibin",
)

SQL_MODE_ANN_BLOCKS = ("l2_only", "l2_filter", "l2_filter_threshold")


def resolve_gt_source_from_args(args: Any) -> str:
    """CLI --gt-source 优先，其次 cfg.dataset.gt_source，默认 auto。"""
    cli = getattr(args, "gt_source", None)
    if cli is not None and str(cli).strip():
        return str(cli).strip().lower()
    ds = (getattr(args, "_index_config", None) or {}).get("dataset", {}) or {}
    return str(ds.get("gt_source", "auto")).strip().lower() or "auto"


def resolve_ann_file_specs(args: Any, ann_s3: dict, ds: Optional[dict] = None) -> dict:
    """按 args.sql_mode 从 ann_s3 的 per-mode 块或顶层扁平字段解析 S3 对象名。

    配置示例::

        "ann_s3": {
          "prefix": "vector/wiki_ann/ivfflat_10m",
          "local_dir": "/tmp/wiki_ann_ivfflat_10m",
          "l2_only": { "query_fvecs": "query_l2_only_k100.fvecs", ... },
          "l2_filter": { ... },
          "l2_filter_threshold": { ... }
        }

    顶层 query_fvecs 等（无 mode 块时）视为 l2_only 兼容旧配置。
    """
    ds = ds or {}
    mode = getattr(args, "sql_mode", None) or "l2_only"
    specs: dict = {}

    block = ann_s3.get(mode)
    if isinstance(block, dict):
        for attr in ANN_FILE_ATTRS:
            v = block.get(attr)
            if v:
                specs[attr] = v

    if not specs and mode == "l2_only":
        for attr in ANN_FILE_ATTRS:
            v = ann_s3.get(attr) or ds.get(attr)
            if v:
                specs[attr] = v

    return specs


def resolve_fbin_file_specs(fbin_s3: dict, ds: Optional[dict] = None) -> dict:
    """解析 cuVS 路径：dataset 顶层本地路径优先，否则用 fbin_s3 中的 S3 对象名。

    配置示例::

        "query_fbin": "/path/to/queries.fbin",
        "groundtruth_ibin": "/path/to/groundtruth.10M.neighbors.ibin",
        "fbin_s3": {
          "prefix": "vector/wiki_all_10m",
          "local_dir": "/tmp/wiki_fbin_10m",
          "query_fbin": "queries.fbin",
          "groundtruth_ibin": "groundtruth.10M.neighbors.ibin"
        }
    """
    ds = ds or {}
    specs: dict = {}
    for attr in FBIN_FILE_ATTRS:
        local = ds.get(attr)
        if local:
            specs[attr] = local
        else:
            v = fbin_s3.get(attr)
            if v:
                specs[attr] = v
    return specs


def _fbin_materialize_specs(fbin_s3: dict, ds: dict, refresh: bool) -> dict[str, str]:
    """每条目：本地文件存在则用本地路径，否则用 fbin_s3 对象名走 S3 下载。"""
    specs: dict[str, str] = {}
    for attr in FBIN_FILE_ATTRS:
        local = ds.get(attr)
        if local and os.path.isfile(local) and not refresh:
            specs[attr] = local
            continue
        s3_name = fbin_s3.get(attr)
        if s3_name:
            specs[attr] = s3_name
        elif local:
            specs[attr] = local
    return specs


def _fbin_dataset_ready(ds: dict) -> bool:
    fbin_s3 = ds.get("fbin_s3") or {}
    for attr in FBIN_FILE_ATTRS:
        if not (ds.get(attr) or fbin_s3.get(attr)):
            return False
    return True


def resolve_s3_connection(args: Any, cfg: dict) -> dict:
    """合并 dataset.s3、ann_s3 / fbin_s3 连接字段、CLI 与凭证文件（不要求 filepath）。"""
    from s3_credentials import DEFAULT_S3_CREDENTIALS_FILE, load_s3_credentials

    ds = cfg.get("dataset", {}) or {}
    s3_cfg = dict(ds.get("s3") or {})
    for block_key in ("ann_s3", "fbin_s3"):
        block = ds.get(block_key) or {}
        for key in ("endpoint", "bucket", "region", "compression"):
            if block.get(key):
                s3_cfg[key] = block[key]

    cli_map = {
        "endpoint": getattr(args, "s3_endpoint", None),
        "bucket": getattr(args, "s3_bucket", None),
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
    return s3_cfg


def _s3_object_key(prefix: str, name: str) -> str:
    prefix = (prefix or "").strip("/")
    name = name.lstrip("/")
    return f"{prefix}/{name}" if prefix else name


def _endpoint_url(endpoint: str) -> str:
    ep = (endpoint or "").strip()
    if not ep:
        return ep
    if ep.startswith("http://") or ep.startswith("https://"):
        return ep
    return f"https://{ep}"


def _download_s3_object(s3_cfg: dict, key: str, local_path: str, log_tag: str) -> None:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise RuntimeError(
            "从 S3 下载评测文件需要 boto3: pip install boto3"
        ) from e

    bucket = s3_cfg["bucket"]
    parent = os.path.dirname(local_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    client = boto3.client(
        "s3",
        endpoint_url=_endpoint_url(s3_cfg["endpoint"]),
        aws_access_key_id=s3_cfg["access_key_id"],
        aws_secret_access_key=s3_cfg["secret_access_key"],
        region_name=s3_cfg.get("region") or None,
        config=Config(signature_version="s3v4"),
    )
    print(f"[{log_tag}] 下载 s3://{bucket}/{key} -> {local_path}", flush=True)
    client.download_file(bucket, key, local_path)


def _resolve_local_cache_dir(
    cfg: dict, s3_block: dict, cache_kind: str, default_sub: str
) -> str:
    if s3_block.get("local_dir"):
        return os.path.abspath(s3_block["local_dir"])
    db = cfg.get("database", "wiki")
    table = cfg.get("table", "table")
    sub = s3_block.get("prefix", default_sub).strip("/").replace("/", "_") or default_sub
    return os.path.join(_PKG_DIR, ".cache", cache_kind, f"{db}_{table}_{sub}")


def _materialize_s3_file_specs(
    args: Any,
    *,
    block_label: str,
    log_tag: str,
    s3_block: dict,
    file_specs: dict[str, str],
    local_dir: str,
    conn: dict,
    refresh: bool,
) -> Optional[str]:
    prefix = s3_block.get("prefix", "")

    for attr, spec in file_specs.items():
        if not spec:
            continue

        if os.path.isfile(spec) and not spec.startswith("s3://"):
            setattr(args, attr, os.path.abspath(spec))
            continue

        if spec.startswith("s3://"):
            without = spec[5:]
            if "/" not in without:
                return f"无效 S3 URI: {spec}"
            uri_bucket, key = without.split("/", 1)
            if uri_bucket != conn["bucket"]:
                return f"S3 URI bucket 与配置不一致: {spec}"
        else:
            key = _s3_object_key(prefix, spec)

        basename = os.path.basename(key)
        local_path = os.path.join(local_dir, basename)

        if os.path.isfile(local_path) and not refresh:
            print(f"[{log_tag}] 使用缓存: {local_path}", flush=True)
        else:
            try:
                _download_s3_object(conn, key, local_path, log_tag)
            except Exception as e:
                return f"下载 {block_label} 文件失败 ({attr}, key={key}): {e}"

        setattr(args, attr, local_path)

    return None


def materialize_ann_files_from_s3(args: Any) -> Optional[str]:
    """若 cfg.dataset.ann_s3 已配置，将 ann 文件下载到本地并写入 args 路径属性。

    返回错误信息；成功或跳过返回 None。
    """
    cfg = getattr(args, "_index_config", None) or {}
    ds = cfg.get("dataset", {}) or {}
    ann_s3 = ds.get("ann_s3")
    if not ann_s3:
        return None

    gt = resolve_gt_source_from_args(args)
    if gt == "fbin":
        return None
    if gt == "auto" and _fbin_dataset_ready(ds):
        return None

    conn = resolve_s3_connection(args, cfg)
    missing = [
        k
        for k in ("endpoint", "access_key_id", "secret_access_key", "bucket", "region")
        if not conn.get(k)
    ]
    if missing:
        return (
            "ann_s3 下载缺少 S3 连接参数: "
            + ", ".join(missing)
            + "（在 dataset.s3 / dataset.ann_s3 或 cfg/s3_credentials.json 中配置）"
        )

    local_dir = _resolve_local_cache_dir(cfg, ann_s3, "ann", "ann")
    refresh = bool(getattr(args, "ann_s3_refresh", False))
    mode = getattr(args, "sql_mode", None) or "l2_only"
    file_specs = resolve_ann_file_specs(args, ann_s3, ds)
    if not file_specs:
        return (
            f"ann_s3 未配置 sql_mode={mode!r} 的 ann 文件"
            f"（请在 ann_s3.{mode} 或顶层 l2_only 字段中设置 query_fvecs 等）"
        )

    return _materialize_s3_file_specs(
        args,
        block_label="ann",
        log_tag="ann_s3",
        s3_block=ann_s3,
        file_specs=file_specs,
        local_dir=local_dir,
        conn=conn,
        refresh=refresh,
    )


def materialize_fbin_files_from_s3(args: Any) -> Optional[str]:
    """若 cfg.dataset.fbin_s3 已配置，将 cuVS fbin/ibin 下载到本地并写入 args。

    返回错误信息；成功或跳过返回 None。
    """
    cfg = getattr(args, "_index_config", None) or {}
    ds = cfg.get("dataset", {}) or {}
    fbin_s3 = ds.get("fbin_s3")
    if not fbin_s3:
        return None

    gt = resolve_gt_source_from_args(args)
    if gt == "ann":
        return None

    conn = resolve_s3_connection(args, cfg)
    missing = [
        k
        for k in ("endpoint", "access_key_id", "secret_access_key", "bucket", "region")
        if not conn.get(k)
    ]
    if missing:
        return (
            "fbin_s3 下载缺少 S3 连接参数: "
            + ", ".join(missing)
            + "（在 dataset.s3 / dataset.fbin_s3 或 cfg/s3_credentials.json 中配置）"
        )

    local_dir = _resolve_local_cache_dir(cfg, fbin_s3, "fbin", "fbin")
    refresh = bool(
        getattr(args, "fbin_s3_refresh", False)
        or getattr(args, "ann_s3_refresh", False)
    )
    file_specs = _fbin_materialize_specs(fbin_s3, ds, refresh)
    if not file_specs.get("query_fbin") or not file_specs.get("groundtruth_ibin"):
        return (
            "fbin 需同时配置 query_fbin 与 groundtruth_ibin"
            "（dataset 本地路径和/或 fbin_s3 中的 S3 对象名）"
        )

    return _materialize_s3_file_specs(
        args,
        block_label="fbin",
        log_tag="fbin_s3",
        s3_block=fbin_s3,
        file_specs=file_specs,
        local_dir=local_dir,
        conn=conn,
        refresh=refresh,
    )


def materialize_recall_files_from_s3(args: Any) -> Optional[str]:
    """按 gt_source 从 S3 物化 ann 或 fbin 评测文件到本地。"""
    err = materialize_fbin_files_from_s3(args)
    if err:
        return err
    return materialize_ann_files_from_s3(args)


def apply_recall_dataset_paths(args: Any, paths: dict) -> None:
    """将合并后的 GT 路径写入 args（CLI 已指定的不覆盖）。"""
    for arg_name in (
        "query_fvecs",
        "groundtruth_ivecs",
        "id_mapping",
        "query_filters",
        "query_fbin",
        "groundtruth_ibin",
    ):
        if getattr(args, arg_name, None) not in (None, ""):
            continue
        v = paths.get(arg_name)
        if v:
            setattr(args, arg_name, v)


def prepare_recall_dataset_from_config(args: Any) -> Optional[str]:
    """从 --config 加载 cfg，按 gt_source 从 S3 物化 ann/fbin 并填充 args 路径。

    无 --config 或 cfg 中无 ann_s3/fbin_s3 时跳过；成功返回 None，失败返回错误信息。
    """
    cfg = getattr(args, "_index_config", None)
    config_path = getattr(args, "config", None)
    if cfg is None:
        if not config_path:
            return None
        if not os.path.isfile(config_path):
            return f"配置文件不存在: {config_path}"
        import json

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        args._index_config = cfg

    ds = cfg.get("dataset", {}) or {}
    if not ds.get("ann_s3") and not ds.get("fbin_s3"):
        return None

    args.sql_mode = getattr(args, "mode", None) or getattr(args, "sql_mode", None) or "l2_only"

    err = materialize_recall_files_from_s3(args)
    if err:
        return err

    from run_vector_test import resolve_recall_dataset_paths

    try:
        paths = resolve_recall_dataset_paths(args)
    except ValueError as e:
        return str(e)

    apply_recall_dataset_paths(args, paths)

    gt = resolve_gt_source_from_args(args)
    effective = paths.get("_gt_source_effective", "db")
    if effective == "fbin":
        print(f"[eval] GT 来源: cuVS fbin/ibin (--gt-source={gt})")
    elif effective == "ann":
        print(f"[eval] GT 来源: ann fvecs/ivecs/id_mapping (--gt-source={gt})")
    return None

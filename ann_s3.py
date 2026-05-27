#!/usr/bin/env python3
"""从 S3/OSS 下载 ann-benchmarks 召回文件（query.fvecs / groundtruth.ivecs / id_mapping）。"""

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

SQL_MODE_ANN_BLOCKS = ("l2_only", "l2_filter", "l2_filter_threshold")


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


def resolve_s3_connection(args: Any, cfg: dict) -> dict:
    """合并 dataset.s3、ann_s3 连接字段、CLI 与凭证文件（不要求 filepath）。"""
    from s3_credentials import DEFAULT_S3_CREDENTIALS_FILE, load_s3_credentials

    ds = cfg.get("dataset", {}) or {}
    s3_cfg = dict(ds.get("s3") or {})
    ann_s3 = ds.get("ann_s3") or {}
    for key in ("endpoint", "bucket", "region", "compression"):
        if ann_s3.get(key):
            s3_cfg[key] = ann_s3[key]

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


def _download_s3_object(s3_cfg: dict, key: str, local_path: str) -> None:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise RuntimeError(
            "从 S3 下载 ann 文件需要 boto3: pip install boto3"
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
    print(f"[ann_s3] 下载 s3://{bucket}/{key} -> {local_path}", flush=True)
    client.download_file(bucket, key, local_path)


def _resolve_local_cache_dir(args: Any, cfg: dict, ann_s3: dict) -> str:
    if ann_s3.get("local_dir"):
        return os.path.abspath(ann_s3["local_dir"])
    db = cfg.get("database", "wiki")
    table = cfg.get("table", "table")
    sub = ann_s3.get("prefix", "ann").strip("/").replace("/", "_") or "ann"
    return os.path.join(_PKG_DIR, ".cache", "ann", f"{db}_{table}_{sub}")


def materialize_ann_files_from_s3(args: Any) -> Optional[str]:
    """若 cfg.dataset.ann_s3 已配置，将 ann 文件下载到本地并写入 args 路径属性。

    返回错误信息；成功返回 None。
    """
    cfg = getattr(args, "_index_config", None) or {}
    ds = cfg.get("dataset", {}) or {}
    ann_s3 = ds.get("ann_s3")
    if not ann_s3:
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

    prefix = ann_s3.get("prefix", "")
    local_dir = _resolve_local_cache_dir(args, cfg, ann_s3)
    refresh = bool(getattr(args, "ann_s3_refresh", False))
    mode = getattr(args, "sql_mode", None) or "l2_only"
    file_specs = resolve_ann_file_specs(args, ann_s3, ds)
    if not file_specs:
        return (
            f"ann_s3 未配置 sql_mode={mode!r} 的 ann 文件"
            f"（请在 ann_s3.{mode} 或顶层 l2_only 字段中设置 query_fvecs 等）"
        )

    for attr, spec in file_specs.items():
        if not spec:
            continue

        # 已是本地存在的绝对/相对路径则直接使用
        if os.path.isfile(spec) and not spec.startswith("s3://"):
            setattr(args, attr, os.path.abspath(spec))
            continue

        if spec.startswith("s3://"):
            # s3://bucket/key — 仅支持本 cfg bucket
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
            print(f"[ann_s3] 使用缓存: {local_path}", flush=True)
        else:
            try:
                _download_s3_object(conn, key, local_path)
            except Exception as e:
                return f"下载 ann 文件失败 ({attr}, key={key}): {e}"

        setattr(args, attr, local_path)

    return None

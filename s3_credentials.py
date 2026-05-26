#!/usr/bin/env python3
"""S3/OSS 访问密钥的统一文件加载（cfg/s3_credentials.json）。"""

from __future__ import annotations

import json
import os
from typing import Optional

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_S3_CREDENTIALS_FILE = os.path.join(_PKG_DIR, "cfg", "s3_credentials.json")


def load_s3_credentials(credentials_file: Optional[str] = None) -> dict:
    """从 JSON 文件读取 access_key_id / secret_access_key。

    文件不存在或字段为空时返回空 dict，不抛错（由调用方汇总其它来源）。
    """
    path = credentials_file or DEFAULT_S3_CREDENTIALS_FILE
    if not path or not os.path.isfile(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"S3 凭证文件必须是 JSON 对象: {path}")

    access_key_id = (
        data.get("access_key_id")
        or data.get("MO_S3_ACCESS_KEY_ID")
        or ""
    ).strip()
    secret_access_key = (
        data.get("secret_access_key")
        or data.get("MO_S3_SECRET_ACCESS_KEY")
        or ""
    ).strip()

    out = {}
    if access_key_id:
        out["access_key_id"] = access_key_id
    if secret_access_key:
        out["secret_access_key"] = secret_access_key
    return out

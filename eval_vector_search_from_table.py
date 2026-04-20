import argparse
import hashlib
import json
import os
import pickle
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, local as _thread_local_cls
from typing import Any, List, Optional, Tuple

import pymysql

# ===== 配置区 =====

DB_CONFIG = dict(
    host="127.0.0.1",
    port=6001,
    user="dump",
    password="111",
    database="jst_app",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.Cursor,
)

# 与 vector_query_concurrent_benchmark.py 一致：S3 的 l2 上界、默认表与列名（可被 sql_config_simple.json 覆盖）
S3_L2_DISTANCE_MAX: float = 1.77
# S2/S3 预检：分区内 / S3 阈值球内行数至少为该值（可被 sql_config_simple.json 覆盖）
MIN_VERIFY_PARTITION_ROWS: int = 2000
TABLE_NAME = "historical_file_blocks"
EMB_COL = "embedding"
FILTER_COL = "file_id"
# 主键列名（S2/S3 在 file_id 分区内可先用 RAND 抽 id 再取 embedding，减轻对向量列排序）
PK_COL = "id"
# True：WHERE file_id=%s 子查询随机 PK，再 JOIN 取 embedding；False：直接 SELECT embedding … ORDER BY RAND()
S23_SAMPLE_VIA_PK_SUBQUERY = True

# 连库抽样 / 跑评测前：校验分区内行数 **严格大于 k**（即 COUNT > k，等价至少 k+1 行）。
# False 时改为 COUNT >= max(k, 1)。
REQUIRE_PARTITION_ROW_COUNT_GT_K: bool = True
# S3：随机探针向量下满足 l2<=S3_L2_DISTANCE_MAX 的行数也须满足同上规则；最大探针次数
S3_THRESHOLD_COVERAGE_MAX_PROBES: int = 32


def load_sql_config_simple(config_path: Optional[str] = None) -> None:
    """从 sql_config_simple.json 读取 max_distance（S3 阈值）、min_verify_partition_rows（预检最小行数）。"""
    global S3_L2_DISTANCE_MAX, MIN_VERIFY_PARTITION_ROWS
    path = config_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sql_config_simple.json"
    )
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    default = cfg.get("default") or {}
    m = default.get("min_verify_partition_rows")
    if m is not None:
        MIN_VERIFY_PARTITION_ROWS = int(m)
    modes = cfg.get("sql_modes") or {}
    m3 = modes.get("m3_l2_filter_threshold") or {}
    extra = m3.get("extra") or {}
    md = extra.get("max_distance")
    if md is not None:
        S3_L2_DISTANCE_MAX = float(md)


load_sql_config_simple()


def _fq_table(simple_name: str) -> str:
    db = DB_CONFIG.get("database")
    if db:
        return f"`{db}`.`{simple_name}`"
    return f"`{simple_name}`"


def _init_sql_mode_templates() -> Tuple[str, str, str]:
    """
    与 vector_query_concurrent_benchmark.run_task_sql1/sql2/sql3 相同语义；
    LIMIT 使用 %s 占位以便与现有 pymysql 参数传递一致。
    """
    s1 = _fq_table(TABLE_NAME)
    s23 = _fq_table(TABLE_NAME)
    ec = f"`{EMB_COL}`"
    fc = f"`{FILTER_COL}`"
    lim = S3_L2_DISTANCE_MAX
    m1 = (
        f"SELECT `file_id`,`id` FROM {s1} "
        f"ORDER BY l2_distance({ec}, %s) ASC LIMIT %s"
    )
    m2 = (
        f"SELECT `file_id`, `id`, l2_distance({ec}, %s) AS dist "
        f"FROM {s23} WHERE {fc} = %s "
        f"ORDER BY dist ASC LIMIT %s"
    )
    m3 = (
        f"SELECT `file_id`, `id`, l2_distance({ec}, %s) AS dist "
        f"FROM {s23} WHERE {fc} = %s "
        f"AND l2_distance({ec}, %s) <= {lim} ORDER BY dist ASC LIMIT %s"
    )
    return m1, m2, m3


SQL_MODE_L2_ONLY, SQL_MODE_L2_FILTER, SQL_MODE_L2_FILTER_THRESHOLD = (
    _init_sql_mode_templates()
)


def refresh_sql_mode_templates() -> None:
    """DB_CONFIG['database'] 变更后重算带库前缀的 SQL（模块加载时默认已算一次）。"""
    global SQL_MODE_L2_ONLY, SQL_MODE_L2_FILTER, SQL_MODE_L2_FILTER_THRESHOLD
    SQL_MODE_L2_ONLY, SQL_MODE_L2_FILTER, SQL_MODE_L2_FILTER_THRESHOLD = (
        _init_sql_mode_templates()
    )


def fixed_query_paths(
    mode_int: int, filter_val: Optional[Any] = None
) -> Tuple[str, str]:
    """
    按 mode 缓存查询向量；S2/S3 再按 filter（file_id）区分，避免与 GT 分区不一致。
    """
    base = f"fixed_query_vectors_m{mode_int}"
    if mode_int in (2, 3) and filter_val is not None:
        h = hashlib.md5(str(filter_val).encode("utf-8")).hexdigest()[:10]
        base = f"{base}_f{h}"
    return f"{base}.pkl", f"{base}.txt"


# 调试开关：仅在 debug_single_query 等调试场景下设置为 True，用于打印 SQL 和参数
DEBUG_PRINT_SQL = False

K = 10                # Top-K
NUM_QUERIES = 10000   # 默认抽样/评测条数；与 --duration 同时用时作为循环向量池大小

# ===== 工具函数 =====

def get_conn():
    return pymysql.connect(**DB_CONFIG)


# 线程级连接缓存：用于 precomputed-GT 召回路径，避免每条查询都新建连接
_tls_conn = _thread_local_cls()


def get_thread_conn():
    c = getattr(_tls_conn, "conn", None)
    if c is None:
        c = get_conn()
        _tls_conn.conn = c
    return c


def row_to_eval_id(row: Tuple) -> str:
    """与 SELECT file_id, id[, dist] 对齐：召回用 file_id\\tid."""
    if not row:
        return ""
    if len(row) >= 2:
        return f"{row[0]}\t{row[1]}"
    return str(row[0])


def fetch_sample_filter_value(conn) -> Any:
    """mode 2/3：抽一条与 concurrent 一致的过滤列值（默认 file_id）。"""
    s23 = _fq_table(TABLE_NAME)
    ec = f"`{EMB_COL}`"
    fc = f"`{FILTER_COL}`"
    sql = f"SELECT {fc} FROM {s23} WHERE {ec} IS NOT NULL LIMIT 1"
    with conn.cursor() as cur:
        cur.execute(sql)
        r = cur.fetchone()
    if not r or r[0] is None:
        raise ValueError(
            f"无法从 {TABLE_NAME} 抽样 `{FILTER_COL}`，请检查表与 `{EMB_COL}` 非空数据"
        )
    return r[0]


def mode_str_to_int(mode_str: str) -> int:
    """
    将模式字符串转换为数字：
    - l2_only -> 1
    - l2_filter -> 2
    - l2_filter_threshold -> 3
    """
    mode_map = {
        "l2_only": 1,
        "l2_filter": 2,
        "l2_filter_threshold": 3,
    }
    if mode_str not in mode_map:
        raise ValueError(f"invalid mode: {mode_str}, must be one of {list(mode_map.keys())}")
    return mode_map[mode_str]


def mode_int_to_str(mode_int: int) -> str:
    """
    将模式数字转换为字符串：
    - 1 -> l2_only
    - 2 -> l2_filter
    - 3 -> l2_filter_threshold
    """
    mode_map = {
        1: "l2_only",
        2: "l2_filter",
        3: "l2_filter_threshold",
    }
    if mode_int not in mode_map:
        raise ValueError(f"invalid mode: {mode_int}, must be 1, 2, or 3")
    return mode_map[mode_int]


def sample_query_vectors(
    conn,
    num_queries: int,
    mode_int: int = 1,
    filter_val: Optional[Any] = None,
) -> List:
    """
    S1：从 historical_file_blocks_cos 随机抽 embedding。
    S2/S3：**在与 GT 相同的** `{FILTER_COL}`（如 file_id）分区内抽 embedding，
    避免查询向量来自其它 file_id 导致 l2 阈值下 GT 全空。
    """
    ec = f"`{EMB_COL}`"
    if mode_int == 1:
        t = _fq_table(TABLE_NAME)
        sql = (
            f"SELECT {ec} FROM {t} WHERE {ec} IS NOT NULL "
            f"ORDER BY RAND() LIMIT %s"
        )
        params: Tuple[Any, ...] = (num_queries,)
        tag = TABLE_NAME
    else:
        t = _fq_table(TABLE_NAME)
        fc = f"`{FILTER_COL}`"
        pk = f"`{PK_COL}`"
        if filter_val is not None:
            if S23_SAMPLE_VIA_PK_SUBQUERY:
                sql = (
                    f"SELECT t0.{ec} FROM {t} AS t0 "
                    f"INNER JOIN (SELECT {pk} AS rid FROM {t} "
                    f"WHERE {fc} = %s ORDER BY RAND() LIMIT %s) AS r "
                    f"ON t0.{pk} = r.rid"
                )
            else:
                sql = (
                    f"SELECT {ec} FROM {t} WHERE {fc} = %s "
                    f"ORDER BY RAND() LIMIT %s"
                )
            params = (filter_val, num_queries)
            tag = f"{TABLE_NAME} WHERE {FILTER_COL}={filter_val!r} (sample via {PK_COL})"
        else:
            sql = (
                f"SELECT {ec} FROM {t} WHERE {ec} IS NOT NULL "
                f"ORDER BY RAND() LIMIT %s"
            )
            params = (num_queries,)
            tag = f"{TABLE_NAME} (no filter)"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    vecs = [r[0] for r in rows if r[0] is not None]
    print(f"sampled {len(vecs)} query vectors from {tag}")
    return vecs


def load_or_create_fixed_query_vectors(
    conn, num_queries: int, mode_int: int = 1, filter_val: Optional[Any] = None
) -> List:
    """
    将随机抽样的向量固定下来（路径依赖 mode，见 fixed_query_paths）：
    - 若存在对应 .pkl，则加载（DB 原始类型，如 bytes）；
    - 若存在对应 .txt，则按行读取为字符串；
    - 否则从表随机抽样并保存。
    """
    pkl_path, txt_path = fixed_query_paths(mode_int, filter_val)
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            vecs = pickle.load(f)
        print(f"loaded {len(vecs)} fixed query vectors from {pkl_path}")
        if len(vecs) >= num_queries:
            return vecs[:num_queries]
        extra = sample_query_vectors(
            conn, num_queries - len(vecs), mode_int, filter_val=filter_val
        )
        vecs.extend(extra)
        with open(pkl_path, "wb") as f:
            pickle.dump(vecs, f)
        return vecs[:num_queries]

    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            vecs = [line.strip() for line in f if line.strip()]
        print(f"loaded {len(vecs)} fixed query vectors from {txt_path}")
        if len(vecs) >= num_queries:
            return vecs[:num_queries]
        extra = sample_query_vectors(
            conn, num_queries - len(vecs), mode_int, filter_val=filter_val
        )
        vecs.extend(extra)
        with open(txt_path, "w", encoding="utf-8") as f:
            for v in vecs:
                f.write((v if isinstance(v, str) else str(v)) + "\n")
        return vecs[:num_queries]

    vecs = sample_query_vectors(conn, num_queries, mode_int, filter_val=filter_val)
    if vecs and isinstance(vecs[0], (bytes, bytearray)):
        with open(pkl_path, "wb") as f:
            pickle.dump(vecs, f)
        print(f"saved {len(vecs)} fixed query vectors to {pkl_path}")
    else:
        with open(txt_path, "w", encoding="utf-8") as f:
            for v in vecs:
                f.write((v if isinstance(v, str) else str(v)) + "\n")
        print(f"saved {len(vecs)} fixed query vectors to {txt_path}")
    return vecs


def multi_filter_cache_key(distinct_ids: List[Any]) -> str:
    return hashlib.md5(
        ",".join(sorted(str(x) for x in distinct_ids)).encode("utf-8")
    ).hexdigest()[:12]


def fetch_distinct_filter_values(conn, max_ids: int) -> List[Any]:
    """
    取 TABLE_NAME 上 FILTER_COL 的去重值，按值排序；max_ids<=0 表示不限制条数。
    """
    t = _fq_table(TABLE_NAME)
    fc = f"`{FILTER_COL}`"
    lim = f" LIMIT {int(max_ids)} " if max_ids > 0 else ""
    sql = (
        f"SELECT DISTINCT {fc} FROM {t} WHERE {fc} IS NOT NULL "
        f"ORDER BY {fc}{lim}"
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [r[0] for r in rows if r[0] is not None]


def sample_queries_across_filter_values(
    conn,
    num_queries: int,
    mode_int: int,
    distinct_filters: List[Any],
) -> Tuple[List[Any], List[Any]]:
    """
    将 num_queries 尽量均分到各个 file_id（分区），每区内在该 FILTER_COL 下随机抽 embedding。
    返回 (vectors, parallel filter 列表)。
    """
    n_ids = len(distinct_filters)
    if n_ids == 0 or num_queries <= 0:
        return [], []
    base, rem = divmod(num_queries, n_ids)
    vecs: List[Any] = []
    fvals: List[Any] = []
    for i, fid in enumerate(distinct_filters):
        q = base + (1 if i < rem else 0)
        if q <= 0:
            continue
        part = sample_query_vectors(conn, q, mode_int, filter_val=fid)
        got = len(part)
        if got < q:
            print(
                f"WARNING: `{FILTER_COL}`={fid!r} 期望抽 {q} 条向量，实际 {got} 条"
            )
        vecs.extend(part)
        fvals.extend([fid] * got)
    return vecs, fvals


def load_or_create_fixed_query_vectors_multi(
    conn,
    num_queries: int,
    mode_int: int,
    distinct_ids: List[Any],
) -> Tuple[List[Any], List[Any]]:
    """
    多 file_id 场景：缓存为 pkl，内含 vecs + filters 等长列表。
    """
    key = multi_filter_cache_key(distinct_ids)
    pkl_path = f"fixed_query_vectors_m{mode_int}_multi_{key}.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        vecs: List[Any] = obj["vecs"]
        filters: List[Any] = obj["filters"]
        if len(vecs) >= num_queries and len(filters) >= num_queries:
            print(
                f"loaded {num_queries} multi-{FILTER_COL} query pairs from {pkl_path} "
                f"(distinct {FILTER_COL} count={len(distinct_ids)})"
            )
            return vecs[:num_queries], filters[:num_queries]
    vecs, filters = sample_queries_across_filter_values(
        conn, num_queries, mode_int, distinct_ids
    )
    with open(pkl_path, "wb") as f:
        pickle.dump({"vecs": vecs, "filters": filters}, f)
    print(
        f"saved {len(vecs)} multi-{FILTER_COL} query pairs to {pkl_path} "
        f"({len(distinct_ids)} distinct {FILTER_COL})"
    )
    return vecs, filters


def verify_matrixone_preconditions_multi(
    conn,
    mode_int: int,
    k: int,
    distinct_ids: List[Any],
    num_queries: int,
) -> bool:
    """对每个分片分别做与单分区相同的 Top-K / S3 球覆盖预检。"""
    if not distinct_ids:
        print(f"ERROR: 无任何 DISTINCT `{FILTER_COL}`，无法做多分区 ann。")
        return False
    per_part_queries = max(1, (num_queries + len(distinct_ids) - 1) // len(distinct_ids))
    for fid in distinct_ids:
        if not verify_matrixone_preconditions(
            conn, mode_int, k, fid, per_part_queries
        ):
            print(f"ERROR: 预检失败于 `{FILTER_COL}`={fid!r}。")
            return False
    print(
        f"[verify] 多分区：`{FILTER_COL}` 共 {len(distinct_ids)} 个取值均已通过预检。"
    )
    return True


def parse_vec_literal(vec_literal: str) -> List[float]:
    """
    将类似 "[0.1,0.2,...]" 的向量字符串解析成 float list。
    如果你的 vecf32 字符串格式不同，可以在这里做相应调整。
    """
    s = vec_literal.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return []
    parts = s.split(",")
    return [float(p) for p in parts]


def vec_to_floats(v) -> List[float]:
    """
    将向量转为 float list：
    - 若为 bytes（如 DB 的 vecf32 二进制），优先按 int32(dim) + dim*float32 解析；
      若头部 dim 与总长度不匹配，则退化为“纯 float32 数组”（dim = len(v) / 4）。
    - 若为 str 则用 parse_vec_literal。
    """
    if isinstance(v, (bytes, bytearray)):
        if len(v) < 4:
            return []
        # 尝试按 [int32 dim][dim*float32] 解析
        dim = struct.unpack("<i", v[:4])[0]
        header_len = 4 + dim * 4
        if 0 < dim and header_len == len(v):
            return list(struct.unpack("<%df" % dim, v[4:header_len]))
        # 否则退化为“纯 float32 数组”：部分存储格式没有显式 dim 头
        dim_raw = len(v) // 4
        if dim_raw <= 0:
            return []
        return list(struct.unpack("<%df" % dim_raw, v[: dim_raw * 4]))
    if isinstance(v, str):
        return parse_vec_literal(v)
    return []


def floats_to_vec_bytes(vec: List[float]) -> bytes:
    """
    将 float list 编码为 vecf32 二进制格式：int32(dim) + dim*float32（小端）。
    """
    dim = len(vec)
    return struct.pack("<i", dim) + struct.pack("<%df" % dim, *vec)


def normalize_vec_param(v):
    """
    统一向量参数格式，供 SQL 占位符使用：
    - bytes/bytearray: 直接返回（DB 支持二进制时可使用）
    - list/tuple: 转为字符串 "[x,y,z,...]"，避免二进制经驱动/字符集后报 malformed vector
    - 其他类型（str 等）：原样返回
    """
    if isinstance(v, (bytes, bytearray)):
        return v
    if isinstance(v, (list, tuple)):
        # 数据库常接受字符串形式；二进制在 PyMySQL/字符集下易被破坏导致 malformed vector
        return "[" + ",".join(str(x) for x in v) + "]"
    return v


def _partition_count_satisfies_topk(n: int, k: int) -> bool:
    """分区内（或阈值球内）行数是否满足 Top-K：默认要求 n > k。"""
    if REQUIRE_PARTITION_ROW_COUNT_GT_K:
        return n > k
    return n >= max(k, 1)


def _s2_s3_verify_row_count(n: int, k: int) -> bool:
    """
    S2/S3 预检：分区内或 S3 阈值球内行数至少 MIN_VERIFY_PARTITION_ROWS（来自 sql_config_simple.json），
    且仍须满足 Top-K 抽样规则。
    """
    if n < MIN_VERIFY_PARTITION_ROWS:
        return False
    return _partition_count_satisfies_topk(n, k)


def count_s1_embedding_rows(conn) -> int:
    t = _fq_table(TABLE_NAME)
    ec = f"`{EMB_COL}`"
    sql = f"SELECT COUNT(*) FROM {t} WHERE {ec} IS NOT NULL"
    with conn.cursor() as cur:
        cur.execute(sql)
        r = cur.fetchone()
    return int(r[0]) if r else 0


def count_s23_partition_rows(conn, filter_val: Any) -> int:
    t = _fq_table(TABLE_NAME)
    fc = f"`{FILTER_COL}`"
    sql = f"SELECT COUNT(*) FROM {t} WHERE {fc} = %s"
    with conn.cursor() as cur:
        cur.execute(sql, (filter_val,))
        r = cur.fetchone()
    return int(r[0]) if r else 0


def fetch_one_embedding_from_partition(conn, filter_val: Any) -> Optional[Any]:
    t = _fq_table(TABLE_NAME)
    ec, fc = f"`{EMB_COL}`", f"`{FILTER_COL}`"
    sql = f"SELECT {ec} FROM {t} WHERE {fc} = %s ORDER BY RAND() LIMIT 1"
    with conn.cursor() as cur:
        cur.execute(sql, (filter_val,))
        r = cur.fetchone()
    if not r:
        return None
    return r[0]


def count_s23_l2_within_threshold(
    conn, filter_val: Any, query_emb: Any
) -> int:
    t = _fq_table(TABLE_NAME)
    ec, fc = f"`{EMB_COL}`", f"`{FILTER_COL}`"
    lim = S3_L2_DISTANCE_MAX
    q = normalize_vec_param(query_emb)
    sql = (
        f"SELECT COUNT(*) FROM {t} WHERE {fc} = %s "
        f"AND l2_distance({ec}, %s) <= {lim}"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (filter_val, q))
        r = cur.fetchone()
    return int(r[0]) if r else 0


def verify_matrixone_preconditions(
    conn,
    mode_int: int,
    k: int,
    filter_val: Optional[Any],
    num_queries: int,
) -> bool:
    """
    连接当前 DB_CONFIG 的 MatrixOne，校验数据量满足预检规则。
    S2/S3：分区内行数至少 MIN_VERIFY_PARTITION_ROWS，且满足 Top-K；S3 另校验阈值球覆盖。
    失败打印原因并返回 False。
    """
    rule = "COUNT > k" if REQUIRE_PARTITION_ROW_COUNT_GT_K else "COUNT >= k"
    rule_s23 = (
        f"COUNT >= {MIN_VERIFY_PARTITION_ROWS} 且 {rule}（Top-{k}）"
    )
    if mode_int == 1:
        n = count_s1_embedding_rows(conn)
        if not _partition_count_satisfies_topk(n, k):
            print(
                f"ERROR: S1 表 {TABLE_NAME} 中带非空 `{EMB_COL}` 的行数为 {n}，"
                f"不满足 Top-{k} 所需（{rule}）。"
            )
            return False
        print(f"[verify] S1 `{EMB_COL}` 非空行数={n}，满足 {rule}（k={k}）。")
        return True

    if mode_int not in (2, 3):
        return True
    if filter_val is None:
        print("ERROR: S2/S3 校验需要 filter_val（如 file_id）。")
        return False

    n_part = count_s23_partition_rows(conn, filter_val)
    if not _s2_s3_verify_row_count(n_part, k):
        print(
            f"ERROR: {TABLE_NAME} WHERE `{FILTER_COL}`={filter_val!r} 行数={n_part}，"
            f"不满足 {rule_s23}。请换分区或增大簇后再试。"
        )
        return False
    print(
        f"[verify] {TABLE_NAME} 分区 `{FILTER_COL}`={filter_val!r} 行数={n_part}，满足 {rule_s23}。"
    )
    if n_part < num_queries:
        print(
            f"[verify] 警告: 计划抽样 num_queries={num_queries}，但分区内仅 {n_part} 行，"
            f"实际抽样条数以较少者为准。"
        )

    if mode_int == 3:
        ok_probe = False
        for attempt in range(1, S3_THRESHOLD_COVERAGE_MAX_PROBES + 1):
            emb = fetch_one_embedding_from_partition(conn, filter_val)
            if emb is None:
                break
            n_q = count_s23_l2_within_threshold(conn, filter_val, emb)
            if _s2_s3_verify_row_count(n_q, k):
                print(
                    f"[verify] S3 l2<={S3_L2_DISTANCE_MAX} 覆盖：探针 {attempt}/"
                    f"{S3_THRESHOLD_COVERAGE_MAX_PROBES} 得到合格行数={n_q}，满足 {rule_s23}。"
                )
                ok_probe = True
                break
        if not ok_probe:
            print(
                f"ERROR: S3 在最多 {S3_THRESHOLD_COVERAGE_MAX_PROBES} 次随机探针下，"
                f"l2_distance<={S3_L2_DISTANCE_MAX} 的行数均未满足 {rule_s23}。"
                f" 请放宽 sql_config_simple.json 中 max_distance、换 file_id 或检查向量。"
            )
            return False

    return True


def write_fvecs(path: str, vectors: List[List[float]]) -> None:
    """
    以 ann-benchmarks 使用的 fvecs 格式写入：
    每个向量 = int32(dim) + dim 个 float32，按小端序存储。
    """
    with open(path, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("<i", dim))
            f.write(struct.pack("<%df" % dim, *vec))


def write_ivecs(path: str, neighbors: List[List[int]]) -> None:
    """
    以 ann-benchmarks 使用的 ivecs 格式写入：
    每个查询 = int32(K) + K 个 int32（邻居的索引）。
    """
    with open(path, "wb") as f:
        for neigh in neighbors:
            k = len(neigh)
            f.write(struct.pack("<i", k))
            if k:
                f.write(struct.pack("<%di" % k, *neigh))


def read_fvecs(path: str) -> List[List[float]]:
    """
    读取 ann-benchmarks 使用的 fvecs 格式，返回 float list 列表。
    """
    vectors: List[List[float]] = []
    with open(path, "rb") as f:
        while True:
            header = f.read(4)
            if not header:
                break
            if len(header) < 4:
                break
            dim = struct.unpack("<i", header)[0]
            body = f.read(dim * 4)
            if len(body) < dim * 4:
                break
            vec = list(struct.unpack("<%df" % dim, body))
            vectors.append(vec)
    return vectors


def read_ivecs(path: str) -> List[List[int]]:
    """
    读取 ann-benchmarks 使用的 ivecs 格式，返回 int 索引列表的列表。
    """
    neighbors: List[List[int]] = []
    with open(path, "rb") as f:
        while True:
            header = f.read(4)
            if not header:
                break
            if len(header) < 4:
                break
            k = struct.unpack("<i", header)[0]
            if k <= 0:
                neighbors.append([])
                continue
            body = f.read(k * 4)
            if len(body) < k * 4:
                break
            idxs = list(struct.unpack("<%di" % k, body))
            neighbors.append(idxs)
    return neighbors


# ===== cuVS / RAFT fbin / ibin 读取（用于 wiki_all 外部 ground truth）=====
#
# 格式（小端）：
#   .fbin                 : uint32 n, uint32 dim,       n*dim   float32 （行主序）
#   .neighbors.ibin       : uint32 n, uint32 k,         n*k     int32   （0-based 下标）
# 与 ann-benchmarks 的 fvecs/ivecs 不同：前者每条向量重复写 dim，后者只有一个头。


def _read_xbin_header(path: str) -> Tuple[int, int]:
    with open(path, "rb") as f:
        hdr = f.read(8)
    if len(hdr) != 8:
        raise ValueError(f"文件头过短（<8 字节）: {path}")
    n, d = struct.unpack("<II", hdr)
    return int(n), int(d)


def load_file_based_queries(
    fbin_path: str, num_queries: int
) -> List[List[float]]:
    """
    读取 cuVS query.fbin 的前 num_queries 条向量，返回 float 列表的列表（shape: N×d）。
    调用方把 list[float] 交给 normalize_vec_param() 转成 "[v1,v2,...]" 字面量后入 SQL。
    """
    n_total, d = _read_xbin_header(fbin_path)
    take = min(num_queries, n_total) if num_queries and num_queries > 0 else n_total
    vectors: List[List[float]] = []
    with open(fbin_path, "rb") as f:
        f.seek(8)  # 跳过头
        remaining = take
        # 分块读取以避免一次性大内存分配
        batch = 1024
        while remaining > 0:
            b = min(batch, remaining)
            nbytes = b * d * 4
            raw = f.read(nbytes)
            if len(raw) != nbytes:
                raise ValueError(
                    f"{fbin_path} 截断：期望 {nbytes} 字节，读到 {len(raw)}"
                )
            floats = struct.unpack(f"<{b * d}f", raw)
            for i in range(b):
                vectors.append(list(floats[i * d : (i + 1) * d]))
            remaining -= b
    print(f"loaded {len(vectors)} queries (dim={d}) from {fbin_path}")
    return vectors


def load_file_based_ground_truth(
    ibin_path: str, num_queries: int, k: int, id_offset: int
) -> List[List[str]]:
    """
    读取 cuVS groundtruth.neighbors.ibin 前 num_queries 行、前 k 列的邻居下标，
    把每个 0-based 下标 `i` 映射成 DB id 字符串 `str(i + id_offset)`。
    """
    n_total, k_file = _read_xbin_header(ibin_path)
    if k_file < k:
        raise ValueError(
            f"{ibin_path} 的 k={k_file} 小于请求的 k={k}"
        )
    take = min(num_queries, n_total) if num_queries and num_queries > 0 else n_total
    gt: List[List[str]] = []
    with open(ibin_path, "rb") as f:
        f.seek(8)  # 跳过头
        # 每行 k_file 个 int32；只取前 k 列，但必须读完整行以保持对齐
        row_bytes = k_file * 4
        for _ in range(take):
            raw = f.read(row_bytes)
            if len(raw) != row_bytes:
                raise ValueError(
                    f"{ibin_path} 截断：期望 {row_bytes} 字节，读到 {len(raw)}"
                )
            ints = struct.unpack(f"<{k_file}i", raw)
            gt.append([str(ints[j] + id_offset) for j in range(k)])
    print(
        f"loaded {len(gt)} ground truth rows (k={k}, file_k={k_file}, "
        f"id_offset={id_offset}) from {ibin_path}"
    )
    return gt


def load_id_mapping(path: str) -> List[str]:
    """
    加载 id_mapping 文件（格式：idx\\trow_id），row_id 通常为 file_id\\tid。
    使用 utf-8-sig 以去掉 UTF-8 BOM，避免首行解析失败导致整表为空。
    """
    mapping: dict[int, str] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 行格式：idx\trow_id；row_id 自身可能是 file_id\tid，故优先按首个 \t 分裂
            parts = line.split("\t", 1)
            if len(parts) != 2:
                # 兼容「idx 与 row_id 之间为空格」的旧文件或手写编辑
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
            idx_str, rid = parts[0], parts[1]
            idx_str = idx_str.strip().lstrip("\ufeff")
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            mapping[idx] = rid
    if not mapping:
        return []
    max_idx = max(mapping.keys())
    result = ["" for _ in range(max_idx + 1)]
    for idx, rid in mapping.items():
        if 0 <= idx <= max_idx:
            result[idx] = rid
    return result


def export_ann_files(
    query_vec_literals: List,
    all_gt_ids: List[List[str]],
    query_fvecs_path: str,
    groundtruth_ivecs_path: str,
    id_mapping_path: str,
) -> None:
    """
    按 ann-benchmarks 风格导出 query.fvecs 和 groundtruth.ivecs：
    - query_vec_literals 可为 str（如 "[0.1,0.2,...]"）或 bytes（vecf32 二进制）
    - id_mapping_path 记录整数索引与 row_id（如 file_id\\tid）的对应关系。
    """
    if not query_vec_literals or not all_gt_ids:
        print("no data to export ann-benchmark files.")
        return

    # 1) 解析查询向量（支持 str 或 bytes）
    query_vectors: List[List[float]] = []
    for s in query_vec_literals:
        vec = vec_to_floats(s)
        if not vec:
            continue
        query_vectors.append(vec)

    if not query_vectors:
        print("no valid query vectors parsed; skip ann file export.")
        return

    # 2) 为 ground truth 中出现的 row_id 分配连续整数索引
    id_to_idx: dict[str, int] = {}
    idx_to_id: List[str] = []
    neighbors_indices: List[List[int]] = []

    for gt_ids in all_gt_ids:
        neigh_idx: List[int] = []
        for rid in gt_ids:
            if rid not in id_to_idx:
                id_to_idx[rid] = len(idx_to_id)
                idx_to_id.append(rid)
            neigh_idx.append(id_to_idx[rid])
        neighbors_indices.append(neigh_idx)

    if not idx_to_id:
        nq = len(all_gt_ids)
        n_nonempty = sum(1 for g in all_gt_ids if g)
        print(
            "ERROR: 无法导出 ann：没有任何有效 ground truth id（id_mapping 会为空）。"
            f" 查询数={nq}，至少返回 1 行的 GT 条数={n_nonempty}。"
            "\n  常见原因（尤其 l2_filter_threshold / S3）："
            f"\n  - 当前 file_id 下没有满足 l2_distance <= {S3_L2_DISTANCE_MAX} 的行；"
            "\n  - --mode23-filter 的 file_id 不对或表数据/向量异常；"
            "\n  - 精确 GT SQL（含 BY RANK WITH OPTION）执行结果为 0 行。"
            "\n  请修正数据或过滤条件后重新执行 --write-ann-files，并删除已生成的不完整文件。"
            "\n  若曾用旧版脚本导出：请删掉 fixed_query_vectors_m3*.pkl/.txt 后重跑，"
            "使 S3 查询向量与当前 file_id 分区一致。"
        )
        return

    # 3) 写入 fvecs / ivecs / 映射文件
    write_fvecs(query_fvecs_path, query_vectors)
    write_ivecs(groundtruth_ivecs_path, neighbors_indices)
    with open(id_mapping_path, "w", encoding="utf-8") as f:
        for idx, rid in enumerate(idx_to_id):
            f.write(f"{idx}\t{rid}\n")

    print(
        f"exported {len(query_vectors)} queries to {query_fvecs_path}, "
        f"ground truth to {groundtruth_ivecs_path}, id mapping to {id_mapping_path}"
    )


def get_ground_truth_ids(
    conn,
    vec_literal: str,
    k: int,
    mode: int,
    filter_val: Optional[Any] = None,
) -> List[str]:
    """
    ground truth 查询：按不同模式对应的 SQL 生成 Top-K。
    要求：使用各自模式的 SQL 模板，并在 SQL 末尾追加
    `BY RANK WITH OPTION 'mode=force'` 以强制精确模式。

    注意：这里不再通过 build_where_and_params 动态拼 WHERE，
    而是直接复用上方定义的 SQL_MODE_* 常量（每种模式一条 SQL）。
    """
    # 选择不同模式对应的 SQL 模板
    if mode == 1:
        base_sql = SQL_MODE_L2_ONLY
    elif mode == 2:
        base_sql = SQL_MODE_L2_FILTER
    elif mode == 3:
        base_sql = SQL_MODE_L2_FILTER_THRESHOLD
    else:
        raise ValueError(f"invalid mode: {mode}, must be 1, 2, or 3")

    if mode in (2, 3) and filter_val is None:
        raise ValueError(
            "mode l2_filter / l2_filter_threshold 需要提供 filter_val（如 file_id）"
        )

    # 去掉注释行（避免注释中的 %s 被当成占位符）
    sql_lines = []
    for line in base_sql.splitlines():
        if line.strip().startswith("--"):
            continue
        sql_lines.append(line)
    sql_stripped = "\n".join(sql_lines).strip()

    # 去掉末尾分号后，追加 BY RANK WITH OPTION 'mode=force'
    if sql_stripped.endswith(";"):
        sql_stripped = sql_stripped[:-1]
    sql = sql_stripped + " BY RANK WITH OPTION 'mode=force';"

    # 统一处理向量参数
    vec_param = normalize_vec_param(vec_literal)

    # 参数顺序与 concurrent run_task_sql1/2/3 一致；LIMIT 用 %s
    if mode == 1:
        params = [vec_param, k]
    elif mode == 2:
        params = [vec_param, filter_val, k]
    else:
        params = [vec_param, filter_val, vec_param, k]

    if DEBUG_PRINT_SQL:
        print(f"[sql][ground_truth][mode={mode_int_to_str(mode)}]\n{sql}\nparams={params}")

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [row_to_eval_id(r) for r in rows]


def get_index_result_ids(
    conn,
    vec_literal: str,
    k: int,
    mode: int,
    filter_val: Optional[Any] = None,
    filter_mode: Optional[str] = None,
) -> List[str]:
    """
    实际检索 SQL。
    为了评估召回率，这里与 ground truth 使用**同一种模式的 SQL 模板**，
    可通过 filter_mode 追加 BY RANK WITH OPTION 'mode=xxx'。
    """
    # 选择不同模式对应的 SQL 模板（与 get_ground_truth_ids 相同）
    if mode == 1:
        base_sql = SQL_MODE_L2_ONLY
    elif mode == 2:
        base_sql = SQL_MODE_L2_FILTER
    elif mode == 3:
        base_sql = SQL_MODE_L2_FILTER_THRESHOLD
    else:
        raise ValueError(f"invalid mode: {mode}, must be 1, 2, or 3")

    if mode in (2, 3) and filter_val is None:
        raise ValueError(
            "mode l2_filter / l2_filter_threshold 需要提供 filter_val（如 file_id）"
        )

    # 去掉注释行（避免注释中的 %s 被当成占位符）
    sql_lines = []
    for line in base_sql.splitlines():
        if line.strip().startswith("--"):
            continue
        sql_lines.append(line)
    sql_stripped = "\n".join(sql_lines).strip()

    # 根据 filter_mode 追加 BY RANK WITH OPTION
    if filter_mode:
        sql = sql_stripped + f" BY RANK WITH OPTION 'mode={filter_mode}'"
    else:
        sql = sql_stripped

    # 统一处理向量参数
    vec_param = normalize_vec_param(vec_literal)

    if mode == 1:
        params = [vec_param, k]
    elif mode == 2:
        params = [vec_param, filter_val, k]
    else:
        params = [vec_param, filter_val, vec_param, k]

    if DEBUG_PRINT_SQL:
        print(f"[sql][index][mode={mode_int_to_str(mode)}]\n{sql}\nparams={params}")

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [row_to_eval_id(r) for r in rows]


def evaluate_single_query_with_precomputed_gt(
    vec_literal: str,
    gt_ids: List[str],
    k: int,
    mode_int: int,
    filter_val: Optional[Any] = None,
    filter_mode: Optional[str] = None,
) -> Tuple[float, List[str], List[str], float]:
    """
    使用预先计算好的 ground truth（从文件加载）来评估单个查询：
    - 只执行一次“实际检索 SQL”，同时记录该次请求耗时
    - 与提供的 gt_ids 做召回率计算
    返回 (recall, gt_ids, vec_literal, latency_sec)。
    """
    conn = get_thread_conn()
    t0 = time.perf_counter()
    res = get_index_result_ids(
        conn,
        vec_literal,
        k,
        mode_int,
        filter_val=filter_val,
        filter_mode=filter_mode,
    )
    latency = time.perf_counter() - t0
    # wiki_all 文件 GT（.ibin）只含 id；此处剥离 SQL 结果里的 "file_id\\t" 前缀
    res_ids_only = [r.split("\t")[-1] for r in res]
    r = recall_at_k(gt_ids, res_ids_only, k)
    return r, gt_ids, vec_literal, latency


def recall_at_k(gt_ids: List[str], res_ids: List[str], k: int) -> float:
    gt_topk = set(gt_ids[:k])
    res_topk = set(res_ids[:k])
    if k == 0:
        return 0.0
    return len(gt_topk & res_topk) / float(k)


# ===== 主评估逻辑 =====

def fetch_ground_truth_only(
    vec_literal: str,
    k: int,
    mode_int: int,
    filter_val: Optional[Any] = None,
) -> Tuple[List[str], str]:
    """
    仅拉取 ground truth（一条 SQL），不跑实际检索。用于 --annfiles-only 时生成 ann 文件，
    可减半数据库请求，加快生成速度。
    返回 (gt_ids, vec_literal)。
    """
    conn = get_conn()
    try:
        gt = get_ground_truth_ids(
            conn,
            vec_literal,
            k,
            mode_int,
            filter_val=filter_val,
        )
        return gt, vec_literal
    finally:
        conn.close()


def evaluate_single_query(
    vec_literal: str,
    k: int,
    mode_int: int,
    filter_val: Optional[Any] = None,
    filter_mode: Optional[str] = None,
) -> Tuple[float, List[str], List[str], float]:
    """
    执行单个查询的 ground truth 和实际检索，返回召回率、ground truth ids、查询向量及实际检索耗时。
    每个线程使用独立的数据库连接。
    """
    conn = get_conn()
    try:
        gt = get_ground_truth_ids(
            conn,
            vec_literal,
            k,
            mode_int,
            filter_val=filter_val,
        )
        t0 = time.perf_counter()
        res = get_index_result_ids(
            conn,
            vec_literal,
            k,
            mode_int,
            filter_val=filter_val,
            filter_mode=filter_mode,
        )
        latency = time.perf_counter() - t0
        r = recall_at_k(gt, res, k)
        return r, gt, vec_literal, latency
    finally:
        conn.close()


def evaluate_single_query_for_qps(
    vec_literal: str,
    k: int,
    mode_int: int,
    filter_val: Optional[Any] = None,
    filter_mode: Optional[str] = None,
) -> float:
    """
    执行单个查询用于 QPS 测试，返回延迟（秒）。
    """
    conn = get_conn()
    try:
        t0 = time.perf_counter()
        _ = get_index_result_ids(
            conn,
            vec_literal,
            k,
            mode_int,
            filter_val=filter_val,
            filter_mode=filter_mode,
        )
        t1 = time.perf_counter()
        return t1 - t0
    finally:
        conn.close()


def evaluate_by_duration(
    query_vecs: List[str],
    k: int,
    mode_int: int,
    write_ann_files: bool,
    concurrency: int,
    duration: float,
    filter_val: Optional[Any] = None,
    filter_vals: Optional[List[Any]] = None,
    filter_mode: Optional[str] = None,
):
    """
    在指定时间内持续运行查询，统计 QPS 和召回率。
    filter_vals 与 query_vecs 等长时，按对循环（多 file_id）；否则 S2/S3 用单一 filter_val。
    """
    import random
    import itertools

    if filter_vals is not None and len(filter_vals) == len(query_vecs):
        pair_cycle = itertools.cycle(list(zip(query_vecs, filter_vals)))
    else:
        pair_cycle = None
    vec_cycle = itertools.cycle(query_vecs)
    
    # 用于统计的共享变量
    query_count = 0
    query_count_lock = Lock()
    latencies = []
    latencies_lock = Lock()
    recalls = []
    recalls_lock = Lock()
    ann_query_literals = []
    all_gt_ids = []
    ann_lock = Lock()
    start_time = time.perf_counter()
    end_time = start_time + duration
    
    def worker():
        """工作线程：持续运行查询直到时间到"""
        nonlocal query_count
        conn = get_conn()
        local_count = 0
        local_latencies = []
        local_recalls = []
        local_ann_queries = []
        local_gt_ids = []
        
        try:
            while time.perf_counter() < end_time:
                if pair_cycle is not None:
                    vec_literal, fv = next(pair_cycle)
                else:
                    vec_literal = next(vec_cycle)
                    fv = filter_val

                # 执行查询并计算召回率
                try:
                    r, gt, vec, _ = evaluate_single_query(
                        vec_literal, k, mode_int, filter_val=fv, filter_mode=filter_mode
                    )
                    local_recalls.append(r)
                    if write_ann_files:
                        local_ann_queries.append(vec)
                        local_gt_ids.append(gt)

                    # 执行 QPS 测试
                    latency = evaluate_single_query_for_qps(
                        vec_literal, k, mode_int, filter_val=fv, filter_mode=filter_mode
                    )
                    local_latencies.append(latency)
                    local_count += 1
                except Exception as e:
                    print(f"query failed: {e}")
                    continue
        finally:
            conn.close()
            # 合并结果
            with query_count_lock:
                nonlocal query_count
                query_count += local_count
            with latencies_lock:
                latencies.extend(local_latencies)
            with recalls_lock:
                recalls.extend(local_recalls)
            if write_ann_files:
                with ann_lock:
                    ann_query_literals.extend(local_ann_queries)
                    all_gt_ids.extend(local_gt_ids)
    
    print(f"running duration test for {duration}s with {concurrency} workers...")
    print(f"start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 启动工作线程
    if concurrency == 1:
        worker()
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker) for _ in range(concurrency)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"worker failed: {e}")
    
    actual_duration = time.perf_counter() - start_time
    qps = query_count / actual_duration if actual_duration > 0 else 0.0
    
    # 计算统计信息
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000 if latencies else 0.0
    min_latency_ms = min(latencies) * 1000 if latencies else 0.0
    max_latency_ms = max(latencies) * 1000 if latencies else 0.0
    p50_latency_ms = sorted(latencies)[len(latencies) // 2] * 1000 if latencies else 0.0
    p95_latency_ms = sorted(latencies)[int(len(latencies) * 0.95)] * 1000 if len(latencies) > 0 else 0.0
    p99_latency_ms = sorted(latencies)[int(len(latencies) * 0.99)] * 1000 if len(latencies) > 0 else 0.0
    
    print(f"end time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("==== Summary ====")
    print(f"duration       = {duration:.2f} s (requested)")
    print(f"actual time    = {actual_duration:.2f} s")
    print(f"queries        = {query_count}")
    print(f"concurrency    = {concurrency}")
    print(f"avg recall@{k} = {avg_recall:.4f}")
    print(f"QPS            = {qps:.2f}")
    print(f"latency stats:")
    print(f"  avg         = {avg_latency_ms:.2f} ms")
    print(f"  min         = {min_latency_ms:.2f} ms")
    print(f"  max         = {max_latency_ms:.2f} ms")
    print(f"  p50         = {p50_latency_ms:.2f} ms")
    print(f"  p95         = {p95_latency_ms:.2f} ms")
    print(f"  p99         = {p99_latency_ms:.2f} ms")
    
    # 导出 ann 文件（如果启用）
    if write_ann_files and ann_query_literals:
        mode_suffix = mode_int_to_str(mode_int)
        query_fvecs_path = f"query_{mode_suffix}_k{k}_duration{duration:.0f}s.fvecs"
        groundtruth_ivecs_path = f"groundtruth_{mode_suffix}_k{k}_duration{duration:.0f}s.ivecs"
        id_mapping_path = f"id_mapping_{mode_suffix}_k{k}_duration{duration:.0f}s.txt"
        
        export_ann_files(
            ann_query_literals,
            all_gt_ids,
            query_fvecs_path,
            groundtruth_ivecs_path,
            id_mapping_path,
        )


def evaluate(
    num_queries: int = NUM_QUERIES,
    k: int = K,
    mode: str = "l2_only",
    query_fvecs_path: Optional[str] = None,
    groundtruth_ivecs_path: Optional[str] = None,
    id_mapping_path: Optional[str] = None,
    query_filters_path: Optional[str] = None,
    write_ann_files: bool = False,
    concurrency: int = 1,
    duration: Optional[float] = None,
    annfiles_only: bool = False,
    database: Optional[str] = None,
    mode23_filter_value: Optional[str] = None,
    skip_db_verify: bool = False,
    ann_distribute_file_ids: bool = False,
    ann_max_distinct_file_ids: int = 50,
    probe: Optional[int] = None,
    filter_mode: Optional[str] = None,
    query_fbin_path: Optional[str] = None,
    groundtruth_ibin_path: Optional[str] = None,
    id_offset: int = 1,
):
    if database:
        DB_CONFIG["database"] = database
    # 每次评测前重新读取 sql_config_simple.json（阈值、预检最小行数等）
    load_sql_config_simple()
    refresh_sql_mode_templates()
    print(
        f"[config] sql_config_simple.json: max_distance={S3_L2_DISTANCE_MAX}, "
        f"min_verify_partition_rows={MIN_VERIFY_PARTITION_ROWS}"
    )

    # 连接数据库设置全局 probe_limit
    if probe is not None:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SET GLOBAL probe_limit = {probe}")
                print(f"[config] SET GLOBAL probe_limit = {probe}")
        finally:
            conn.close()

        # 新开 session 验证 probe_limit 值
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SHOW VARIABLES LIKE 'probe_limit'")
                result = cur.fetchone()
                if result:
                    print(f"[config] Current probe_limit: {result[1]}")
        finally:
            conn.close()

    # 打印 filter_mode 配置
    if filter_mode:
        print(f"[config] Filter mode: {filter_mode}")

    # 打印示例 SQL
    mode_int = mode_str_to_int(mode)
    if mode_int == 1:
        example_sql = SQL_MODE_L2_ONLY
    elif mode_int == 2:
        example_sql = SQL_MODE_L2_FILTER
    else:
        example_sql = SQL_MODE_L2_FILTER_THRESHOLD

    # 构造带 filter_mode 的示例 SQL
    example_sql_clean = " ".join([ln for ln in example_sql.splitlines() if not ln.strip().startswith("--")]).strip()
    if filter_mode:
        example_sql_clean += f" BY RANK WITH OPTION 'mode={filter_mode}'"
    print(f"[config] Example SQL: {example_sql_clean[:200]}...")

    mode23_filter: Optional[Any] = None
    if mode_int in (2, 3) and not ann_distribute_file_ids:
        if mode23_filter_value is not None and str(mode23_filter_value).strip() != "":
            mode23_filter = mode23_filter_value
        else:
            c = get_conn()
            try:
                mode23_filter = fetch_sample_filter_value(c)
            finally:
                c.close()
    elif mode_int in (2, 3) and ann_distribute_file_ids:
        if mode23_filter_value is not None and str(mode23_filter_value).strip() != "":
            print(
                f"注意: 已启用 --ann-distribute-file-ids，将忽略 --mode23-filter={mode23_filter_value!r}，"
                f"改为使用表中至多 {ann_max_distinct_file_ids or '全部'} 个 DISTINCT `{FILTER_COL}`。"
            )

    mode_names = {
        "l2_only": "S1：cosine_similarity 全表 Top-K（historical_file_blocks_cos）",
        "l2_filter": "S2：file_id 过滤 + L2 Top-K（historical_file_blocks）",
        "l2_filter_threshold": (
            f"S3：file_id + L2<={S3_L2_DISTANCE_MAX} + Top-K（historical_file_blocks）"
        ),
    }
    if duration:
        print(
            f"running evaluate with mode={mode} ({mode_names.get(mode, mode)}), k={k}, "
            f"duration={duration}s (vector pool size={num_queries}), concurrency={concurrency}"
        )
    else:
        print(f"running evaluate with mode={mode} ({mode_names.get(mode, mode)}), k={k}, num_queries={num_queries}, concurrency={concurrency}")
    # 1. 决定查询向量与 ground truth 的来源
    precomputed_gt_ids: Optional[List[List[str]]] = None
    per_query_filters_opt: Optional[List[Any]] = None

    if ann_distribute_file_ids and mode_int == 1:
        print("提示: --ann-distribute-file-ids 仅适用于 l2_filter / l2_filter_threshold，S1 已忽略该选项。")

    # cuVS .fbin/.ibin 直读路径（wiki_all ground truth，免 id_mapping，只适用于 l2_only）
    if query_fbin_path and groundtruth_ibin_path:
        if mode_int != 1:
            print(
                f"警告: --query-fbin/--groundtruth-ibin 的 cuVS ground truth 是无过滤 L2，"
                f"不适用于 mode={mode}（仅 l2_only 有意义）。改走 DB 抽样 + 实时 GT 路径。"
            )
            # 清空让控制流走默认分支
            query_fbin_path = None
            groundtruth_ibin_path = None

    if query_fbin_path and groundtruth_ibin_path:
        print(
            f"loading queries from {query_fbin_path} and ground truth from "
            f"{groundtruth_ibin_path} (id_offset={id_offset})..."
        )
        load_n = num_queries if num_queries and num_queries > 0 else 0
        query_vectors = load_file_based_queries(query_fbin_path, load_n)
        precomputed_gt_ids = load_file_based_ground_truth(
            groundtruth_ibin_path, load_n, k, id_offset
        )
        total = min(len(query_vectors), len(precomputed_gt_ids))
        if not duration and num_queries and num_queries > 0:
            total = min(total, num_queries)
        query_vectors = query_vectors[:total]
        precomputed_gt_ids = precomputed_gt_ids[:total]
        if not query_vectors or not precomputed_gt_ids:
            print("ERROR: 加载的 query/ground truth 为空，退出。")
            return
        query_vecs = query_vectors
        print(
            f"loaded {len(query_vecs)} queries and corresponding ground truth "
            f"from cuVS fbin/ibin."
        )
    elif query_fvecs_path and groundtruth_ivecs_path and id_mapping_path:
        if ann_distribute_file_ids:
            print(
                "提示: 已指定 --query-fvecs 等文件，--ann-distribute-file-ids 不参与加载（vectors/filters 来自文件）。"
            )
        # 使用预先生成的 ann 文件：只跑实际检索 + 从文件比召回
        print(f"loading queries from {query_fvecs_path} and ground truth from {groundtruth_ivecs_path} ...")
        query_vectors = read_fvecs(query_fvecs_path)
        neighbors_indices = read_ivecs(groundtruth_ivecs_path)
        id_mapping = load_id_mapping(id_mapping_path)

        if not query_vectors or not neighbors_indices or not id_mapping:
            print(
                "failed to load query/groundtruth/id_mapping files; please check paths and contents."
            )
            print(
                f"  query_vectors={len(query_vectors)}, "
                f"neighbors_indices={len(neighbors_indices)}, "
                f"id_mapping_len={len(id_mapping)}"
            )
            if query_vectors and neighbors_indices and not id_mapping:
                print(
                    "  hint: 确认 id_mapping 路径正确；每行「idx\\trow_id」或「idx row_id」；"
                    "已用 utf-8-sig 读入以兼容 BOM。若从 Excel 导出请用制表符而非逗号。"
                )
            return

        # 对齐长度
        total = min(len(query_vectors), len(neighbors_indices))
        # 按 num_queries 进行裁剪（duration 模式下只按 duration 控制循环时间）
        if not duration:
            total = min(total, num_queries)
        query_vectors = query_vectors[:total]
        neighbors_indices = neighbors_indices[:total]

        # 将 index ground truth 转成 row_id 列表（与 id_mapping 一致）
        precomputed_gt_ids = []
        for idx_list in neighbors_indices:
            row_ids: List[str] = []
            for idx in idx_list:
                if 0 <= idx < len(id_mapping):
                    row_ids.append(id_mapping[idx])
            precomputed_gt_ids.append(row_ids)

        # 评估阶段直接使用 float list 作为 vec_literal
        query_vecs = query_vectors
        # S2/S3：可选每条 query 对应一个 filter（多 file_id 导出的 .filters.txt）
        filters_guess = (
            query_filters_path
            or (
                query_fvecs_path.replace(".fvecs", ".filters.txt")
                if query_fvecs_path.endswith(".fvecs")
                else None
            )
        )
        if mode_int in (2, 3) and filters_guess and os.path.isfile(filters_guess):
            with open(filters_guess, "r", encoding="utf-8") as ff:
                per_query_filters_opt = [ln.strip() for ln in ff if ln.strip()]
            if len(per_query_filters_opt) != len(query_vecs):
                print(
                    f"ERROR: {filters_guess} 行数 {len(per_query_filters_opt)} "
                    f"与 query 数 {len(query_vecs)} 不一致。"
                )
                return
            print(
                f"loaded per-query `{FILTER_COL}` from {filters_guess} "
                f"({len(per_query_filters_opt)} rows)."
            )
        print(f"loaded {len(query_vecs)} queries and corresponding ground truth from files.")
    else:
        # 从数据库抽样并固定查询向量，同时在线计算 ground truth
        conn = get_conn()
        try:
            # --duration 时同样加载 num_queries 条进入池子，在时间窗口内循环使用（不再截断为 1000）
            vecs_to_load = num_queries
            if ann_distribute_file_ids and mode_int in (2, 3):
                distinct = fetch_distinct_filter_values(conn, ann_max_distinct_file_ids)
                print(
                    f"multi-{FILTER_COL}: DISTINCT 取值 {len(distinct)} 个 "
                    f"(ann_max_distinct_file_ids={ann_max_distinct_file_ids or '无限制'})"
                )
                if len(distinct) == 0:
                    print(f"ERROR: 表中无可用 DISTINCT `{FILTER_COL}`。")
                    return
                if not skip_db_verify:
                    if not verify_matrixone_preconditions_multi(
                        conn, mode_int, k, distinct, vecs_to_load
                    ):
                        return
                query_vecs, per_query_filters_opt = load_or_create_fixed_query_vectors_multi(
                    conn, vecs_to_load, mode_int, distinct
                )
                if len(query_vecs) != len(per_query_filters_opt):
                    m = min(len(query_vecs), len(per_query_filters_opt))
                    query_vecs = query_vecs[:m]
                    per_query_filters_opt = per_query_filters_opt[:m]
            else:
                if not skip_db_verify:
                    if not verify_matrixone_preconditions(
                        conn,
                        mode_int,
                        k,
                        mode23_filter if mode_int in (2, 3) else None,
                        vecs_to_load,
                    ):
                        return
                query_vecs = load_or_create_fixed_query_vectors(
                    conn,
                    vecs_to_load,
                    mode_int,
                    filter_val=mode23_filter if mode_int in (2, 3) else None,
                )
            if not query_vecs:
                print("no query vectors sampled, please check data.")
                return
        finally:
            conn.close()

    if mode_int in (2, 3):
        if per_query_filters_opt is not None:
            q_filters = per_query_filters_opt
        else:
            q_filters = [mode23_filter] * len(query_vecs)
    else:
        q_filters = [None] * len(query_vecs)

    if len(q_filters) != len(query_vecs):
        print(
            f"ERROR: query 条数 {len(query_vecs)} 与 filter 条数 {len(q_filters)} 不一致。"
        )
        return

    # 如果指定了 duration，按时间运行
    if duration:
        return evaluate_by_duration(
            query_vecs,
            k,
            mode_int,
            write_ann_files,
            concurrency,
            duration,
            filter_val=mode23_filter,
            filter_vals=q_filters,
        )

    # 2. 召回率 + QPS 合并为单轮：每条 query 只打一次实际检索，同时统计召回和耗时
    recalls = []
    latencies: List[float] = []
    ann_query_literals: List[str] = []
    all_gt_ids: List[List[str]] = []
    processed_count = 0
    processed_lock = Lock()
    start_time = time.perf_counter()
    last_log_time = start_time
    # 仅生成 ann 文件时不计算召回，日志里不输出 recall
    progress_show_recall = not (annfiles_only and write_ann_files)

    def update_progress(last_recall: float):
        """
        每隔约 2 秒打印一次当前进度、近似 QPS；若在算召回则再打印 last recall@k。
        """
        nonlocal processed_count, last_log_time
        now = time.perf_counter()
        with processed_lock:
            processed_count += 1
            elapsed = now - last_log_time
            # 到达总数时强制打一条日志；否则按 2 秒节流
            if processed_count == len(query_vecs) or elapsed >= 2.0:
                total_elapsed = now - start_time
                current_qps = processed_count / total_elapsed if total_elapsed > 0 else 0.0
                if progress_show_recall:
                    print(
                        f"[recall] processed {processed_count}/{len(query_vecs)} queries, "
                        f"approx QPS={current_qps:.2f}, last recall@{k}={last_recall:.4f}"
                    )
                else:
                    print(
                        f"[ann] processed {processed_count}/{len(query_vecs)} queries, "
                        f"approx QPS={current_qps:.2f}"
                    )
                last_log_time = now

    if precomputed_gt_ids is not None:
        # 使用文件中的 ground truth，只执行实际检索 SQL，同时记录耗时
        if concurrency == 1:
            for i, (vec_literal, gt_ids) in enumerate(
                zip(query_vecs, precomputed_gt_ids)
            ):
                r, gt, vec, latency = evaluate_single_query_with_precomputed_gt(
                    vec_literal,
                    gt_ids,
                    k,
                    mode_int,
                    filter_val=q_filters[i],
                    filter_mode=filter_mode,
                )
                recalls.append(r)
                latencies.append(latency)
                update_progress(r)
        else:
            print(f"running recall + QPS (single pass) with {concurrency} workers (precomputed ground truth)...")
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_i = {
                    executor.submit(
                        evaluate_single_query_with_precomputed_gt,
                        query_vecs[i],
                        precomputed_gt_ids[i],
                        k,
                        mode_int,
                        q_filters[i],
                    ): i
                    for i in range(len(query_vecs))
                }

                results = []
                for future in as_completed(future_to_i):
                    try:
                        r, gt, vec, latency = future.result()
                        idx = future_to_i[future]
                        results.append((idx, r, gt, vec, latency))
                        update_progress(r)
                    except Exception as e:
                        print(f"query failed: {e}")

                results.sort(key=lambda x: x[0])
                for idx, r, gt, vec, latency in results:
                    recalls.append(r)
                    latencies.append(latency)
    else:
        # 在线计算 ground truth
        # annfiles_only 时只跑 ground truth SQL，不跑实际检索，约可减半耗时
        if annfiles_only and write_ann_files:
            if concurrency == 1:
                for i, vec_literal in enumerate(query_vecs):
                    gt, vec = fetch_ground_truth_only(
                        vec_literal, k, mode_int, filter_val=q_filters[i]
                    )
                    all_gt_ids.append(gt)
                    ann_query_literals.append(vec)
                    update_progress(1.0)
            else:
                print(f"generating ann files (ground truth only, no index query) with {concurrency} workers...")
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    future_to_i = {
                        executor.submit(
                            fetch_ground_truth_only,
                            query_vecs[i],
                            k,
                            mode_int,
                            q_filters[i],
                        ): i
                        for i in range(len(query_vecs))
                    }
                    results = []
                    for future in as_completed(future_to_i):
                        try:
                            gt, vec = future.result()
                            idx = future_to_i[future]
                            results.append((idx, gt, vec))
                            update_progress(1.0)
                        except Exception as e:
                            print(f"query failed: {e}")
                    results.sort(key=lambda x: x[0])
                    for idx, gt, vec in results:
                        all_gt_ids.append(gt)
                        ann_query_literals.append(vec)
        else:
            # 需要召回率：跑 ground truth + 实际检索，同时记录实际检索耗时
            if concurrency == 1:
                for i, vec_literal in enumerate(query_vecs):
                    r, gt, vec, latency = evaluate_single_query(
                        vec_literal, k, mode_int, filter_val=q_filters[i]
                    )
                    recalls.append(r)
                    latencies.append(latency)
                    if write_ann_files:
                        ann_query_literals.append(vec)
                        all_gt_ids.append(gt)
                    update_progress(r)
            else:
                print(f"running recall + QPS (single pass) with {concurrency} workers...")
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    future_to_i = {
                        executor.submit(
                            evaluate_single_query,
                            query_vecs[i],
                            k,
                            mode_int,
                            q_filters[i],
                            filter_mode,
                        ): i
                        for i in range(len(query_vecs))
                    }
                    
                    results = []
                    for future in as_completed(future_to_i):
                        try:
                            r, gt, vec, latency = future.result()
                            idx = future_to_i[future]
                            results.append((idx, r, gt, vec, latency))
                            update_progress(r)
                        except Exception as e:
                            print(f"query failed: {e}")
                    
                    results.sort(key=lambda x: x[0])
                    for idx, r, gt, vec, latency in results:
                        recalls.append(r)
                        latencies.append(latency)
                        if write_ann_files:
                            ann_query_literals.append(vec)
                            all_gt_ids.append(gt)

    phase_end_time = time.perf_counter()
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

    # 3. 如需要，先导出 ann-benchmark 风格的 query.fvecs 和 groundtruth.ivecs 文件
    #    当使用预先计算的 ground truth 时，不再重复导出
    if write_ann_files and precomputed_gt_ids is None:
        # 根据 mode 和其他参数生成文件名
        mode_suffix = mode
        query_fvecs_path = f"query_{mode_suffix}_k{k}.fvecs"
        groundtruth_ivecs_path = f"groundtruth_{mode_suffix}_k{k}.ivecs"
        id_mapping_path = f"id_mapping_{mode_suffix}_k{k}.txt"

        export_ann_files(
            ann_query_literals,
            all_gt_ids,
            query_fvecs_path,
            groundtruth_ivecs_path,
            id_mapping_path,
        )

        if mode_int in (2, 3):
            filters_out = f"query_{mode_suffix}_k{k}.filters.txt"
            with open(filters_out, "w", encoding="utf-8") as ff:
                for fv in q_filters:
                    ff.write(f"{fv}\n")
            print(
                f"exported per-query `{FILTER_COL}` ({len(q_filters)} lines) to {filters_out}"
            )

        if annfiles_only:
            print("ann files generated; skip QPS as annfiles_only=True.")
            return

    # 4. 召回与 QPS 已在一轮中完成，用同一轮耗时和 latencies 统计
    if latencies:
        total = phase_end_time - start_time
        qps = len(latencies) / total if total > 0 else 0.0
    else:
        total = 0.0
        qps = 0.0
    qpm = qps * 60.0
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000 if latencies else 0.0
    min_latency_ms = min(latencies) * 1000 if latencies else 0.0
    max_latency_ms = max(latencies) * 1000 if latencies else 0.0
    p50_latency_ms = sorted(latencies)[len(latencies) // 2] * 1000 if latencies else 0.0
    p95_latency_ms = sorted(latencies)[int(len(latencies) * 0.95)] * 1000 if latencies else 0.0
    p99_latency_ms = sorted(latencies)[int(len(latencies) * 0.99)] * 1000 if latencies else 0.0

    print("==== Summary ====")
    print(f"queries        = {len(query_vecs)}")
    print(f"concurrency    = {concurrency}")
    print(f"avg recall@{k} = {avg_recall:.4f}")
    print(f"QPS            = {qps:.2f}")
    print(f"QPM            = {qpm:.2f}")
    print(f"total time    = {total:.2f} s")
    print(f"latency stats:")
    print(f"  avg         = {avg_latency_ms:.2f} ms")
    print(f"  min         = {min_latency_ms:.2f} ms")
    print(f"  max         = {max_latency_ms:.2f} ms")
    print(f"  p50         = {p50_latency_ms:.2f} ms")
    print(f"  p95         = {p95_latency_ms:.2f} ms")
    print(f"  p99         = {p99_latency_ms:.2f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate vector search recall & QPS（SQL 与 vector_query_concurrent_benchmark S1/S2/S3 对齐）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="l2_only",
        choices=["l2_only", "l2_filter", "l2_filter_threshold"],
        help="l2_only=S1 cosine Top-K；l2_filter=S2 file_id+L2；l2_filter_threshold=S3 同上且 l2 距离阈值见 sql_config_simple.json",
    )
    parser.add_argument(
        "--table",
        type=str,
        default=None,
        help="指定表名（覆盖默认的 historical_file_blocks）",
    )
    parser.add_argument("--k", type=int, default=K, help="Top-K size")
    parser.add_argument(
        "--num-queries",
        type=int,
        default=NUM_QUERIES,
        help="从库抽样/缓存的 query 条数（默认 10000）；"
        "指定 --duration 时该值仍有效：先抽满池子再在时间窗内循环",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="压测时长（秒）：在指定时间内循环执行查询+QPS；需先有 num_queries 大小的向量池。"
        "示例: --num-queries 10000 --duration 120",
    )
    parser.add_argument(
        "--write-ann-files",
        action="store_true",
        help="if set, export query.fvecs and groundtruth.ivecs (ann-benchmarks style)",
    )
    parser.add_argument(
        "--query-fvecs",
        type=str,
        default=None,
        help="path to precomputed query.fvecs (if set, use it instead of sampling from DB)",
    )
    parser.add_argument(
        "--groundtruth-ivecs",
        type=str,
        default=None,
        help="path to precomputed groundtruth.ivecs (paired with --query-fvecs)",
    )
    parser.add_argument(
        "--id-mapping",
        type=str,
        default=None,
        help="id_mapping.txt：ivecs 下标 -> row_id（如 file_id\\tid）",
    )
    parser.add_argument(
        "--annfiles-only",
        action="store_true",
        help="only export ann files (query.fvecs/groundtruth.ivecs) and skip QPS",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="number of concurrent workers for testing (default: 1, serial execution)",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="覆盖 DB_CONFIG 中的库名（并与 SQL 中 `db`.`table` 一致）；默认 jst_app",
    )
    parser.add_argument(
        "--mode23-filter",
        type=str,
        default=None,
        dest="mode23_filter",
        metavar="VALUE",
        help="S2/S3 时固定过滤列值（默认与 FILTER_COL，常为 file_id）；不设则从表中抽样一条",
    )
    parser.add_argument(
        "--skip-db-verify",
        action="store_true",
        help="跳过连库预检（分区内行数须 >k、S3 l2 球覆盖等）",
    )
    parser.add_argument(
        "--ann-distribute-file-ids",
        action="store_true",
        help="生成 ann/在线评测时：S2/S3 将 num_queries 均分到表中多个 DISTINCT file_id，"
        "每条 query 在对应分区内抽样向量（需配合 --write-ann-files 或全量评测）",
    )
    parser.add_argument(
        "--ann-max-distinct-file-ids",
        type=int,
        default=50,
        help="配合 --ann-distribute-file-ids：最多使用多少个 DISTINCT file_id；0=不限制（全表 distinct）",
    )
    parser.add_argument(
        "--query-filters",
        type=str,
        default=None,
        help="与 --query-fvecs 配套：每行一个 file_id，与 query 顺序一一对应；"
        "不设则若存在同名 .filters.txt 会自动加载",
    )
    parser.add_argument(
        "--probe",
        type=int,
        default=None,
        help="设置 probe_limit 值（用于 IVF 索引查询）",
    )
    parser.add_argument(
        "--filter-mode",
        type=str,
        default=None,
        choices=["pre", "post", "force"],
        help="SQL 后缀模式：pre（预过滤）、post（后过滤）、force（强制精确搜索）",
    )
    parser.add_argument(
        "--query-fbin",
        type=str,
        default=None,
        help="cuVS query.fbin 路径（<II>头 + n*dim float32）。与 --groundtruth-ibin 配对；仅 l2_only 生效",
    )
    parser.add_argument(
        "--groundtruth-ibin",
        type=str,
        default=None,
        help="cuVS groundtruth.neighbors.ibin 路径（<II>头 + n*k int32，0-based）",
    )
    parser.add_argument(
        "--id-offset",
        type=int,
        default=1,
        help="fbin 0-based 下标 i 映射到 DB id = i + id_offset；AUTO_INCREMENT 从 1 起时取默认 1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 如果指定了表名，更新全局表名变量
    if args.table:
        TABLE_NAME = args.table
        # 刷新 SQL 模板
        refresh_sql_mode_templates()
    evaluate(
        num_queries=args.num_queries,
        k=args.k,
        mode=args.mode,
        query_fvecs_path=args.query_fvecs,
        groundtruth_ivecs_path=args.groundtruth_ivecs,
        id_mapping_path=args.id_mapping,
        query_filters_path=args.query_filters,
        write_ann_files=args.write_ann_files,
        concurrency=args.concurrency,
        duration=args.duration,
        annfiles_only=args.annfiles_only,
        database=args.database,
        mode23_filter_value=args.mode23_filter,
        skip_db_verify=args.skip_db_verify,
        ann_distribute_file_ids=args.ann_distribute_file_ids,
        ann_max_distinct_file_ids=args.ann_max_distinct_file_ids,
        probe=args.probe,
        filter_mode=args.filter_mode,
        query_fbin_path=args.query_fbin,
        groundtruth_ibin_path=args.groundtruth_ibin,
        id_offset=args.id_offset,
    )


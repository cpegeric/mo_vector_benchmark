#!/usr/bin/env python3
"""recall_narrow.py — recall/QPS for VECINT8 / VECUINT8 *base column* tables.

The base table stores embeddings scaled to integer range (gen_narrow_base.py:
int8 = round(v*127); uint8 = round(v*127+128)). The f32 query vectors must be
scaled the SAME way and formatted as bare integers, otherwise MO can't cast the
query literal to the int8/uint8 column. Uniform positive scale + offset preserve
L2 nearest-neighbor order, so the original groundtruth (ibin) still applies.

Prints the same metric lines run_matrix.py parses (avg recall@k, QPS, latency).
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, struct, sys, threading, time
import numpy as np
import pymysql


def read_fbin(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        return np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d)


def read_ibin(path):
    with open(path, "rb") as f:
        n, k = struct.unpack("<II", f.read(8))
        return np.frombuffer(f.read(n * k * 4), dtype=np.int32).reshape(n, k)


def scale(q, mode):
    if mode == "int8":
        return np.clip(np.round(q * 127.0), -127, 127).astype(int)
    return np.clip(np.round(q * 127.0 + 128.0), 0, 255).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", required=True)
    ap.add_argument("--table", default="hfb")
    ap.add_argument("--mode", required=True, choices=["int8", "uint8"])
    ap.add_argument("--query-fbin", required=True)
    ap.add_argument("--groundtruth-ibin", required=True)
    ap.add_argument("-n", "--num-queries", type=int, default=200)
    ap.add_argument("-k", "--k", type=int, default=10)
    ap.add_argument("--probe", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--id-offset", type=int, default=1)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=6001)
    ap.add_argument("--user", default="dump"); ap.add_argument("--password", default="111")
    a = ap.parse_args()

    q = read_fbin(a.query_fbin)[: a.num_queries]
    gt = read_ibin(a.groundtruth_ibin)[: a.num_queries]
    qi = scale(q, a.mode)
    gt_ids = [set((gt[i][: a.k] + a.id_offset).tolist()) for i in range(len(q))]
    lits = ["[" + ",".join(str(int(x)) for x in qi[i]) + "]" for i in range(len(q))]
    sql = f"SELECT `id` FROM `{a.database}`.`{a.table}` ORDER BY l2_distance(`embedding`, %s) ASC LIMIT %s"

    # One persistent connection per worker thread (reused across all the queries
    # that thread runs), with probe_limit set once at connection time. This
    # matches eval_vector_search_from_table.py's get_thread_conn model. Opening a
    # fresh connection per query (the old behavior) charged TCP+auth+session-init
    # + a SET round-trip to every query; that churn is excluded from the per-query
    # timer (t0 below) but included in the wall-clock QPS denominator, which
    # collapsed QPS while p50 stayed low. Reusing connections makes QPS for the
    # int8/uint8 base tables apples-to-apples with the f32/f16/bf16 harness.
    _tls = threading.local()

    def get_conn():
        c = getattr(_tls, "conn", None)
        if c is None:
            c = pymysql.connect(host=a.host, port=a.port, user=a.user, password=a.password,
                                database=a.database, read_timeout=600)
            with c.cursor() as cur:
                cur.execute(f"SET probe_limit={a.probe}")
            _tls.conn = c
        return c

    def work(i):
        conn = get_conn()
        with conn.cursor() as cur:
            t0 = time.perf_counter()
            cur.execute(sql, (lits[i], a.k))
            got = {r[0] for r in cur.fetchall()}
            dt = (time.perf_counter() - t0) * 1000.0
        inter = len(got & gt_ids[i])
        return dt, inter / a.k

    lats, recs = [], []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for dt, r in ex.map(work, range(len(q))):
            lats.append(dt); recs.append(r)
    wall = time.perf_counter() - t0
    lats.sort()
    pct = lambda p: lats[min(len(lats) - 1, int(len(lats) * p))]
    print("==== Summary ====")
    print(f"queries        = {len(q)}")
    print(f"avg recall@{a.k} = {np.mean(recs):.4f}")
    print(f"QPS            = {len(q)/wall:.2f}")
    print(f"total time    = {wall:.2f} s")
    print("latency stats:")
    print(f"  avg         = {np.mean(lats):.2f} ms")
    print(f"  p50         = {pct(0.50):.2f} ms")
    print(f"  p95         = {pct(0.95):.2f} ms")
    print(f"  p99         = {pct(0.99):.2f} ms")


if __name__ == "__main__":
    main()

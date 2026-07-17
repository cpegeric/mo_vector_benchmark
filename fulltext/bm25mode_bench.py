#!/usr/bin/env python3
"""
Bag-of-words retrieval: POSITION-FREE fulltext2 vs bm25, on CJK (gojieba).

The point of the consolidation is that a POSITION_FREE fulltext2 index serves the exact
same bag-of-words ranked retrieval as bm25 (both run Block-Max WAND over docID/tf
postings, no positions). This measures whether that holds — same result set + same speed.

Since the IN BM25 MODE grammar isn't wired yet, we drive fulltext2's bag-of-words
searchWAND path the equivalent way: feed it the query already jieba-tokenized and
space-separated, in BOOLEAN mode (bare tokens → disjunction → searchWAND). bm25 gets the
raw query (it tokenizes internally with the same gojieba dict).

  fulltext2-pf   MATCH(body) AGAINST('tok1 tok2 …' IN BOOLEAN MODE)   (position-free index)
  bm25           bm25(body) AGAINST('原始query')

Reports latency (p50/avg) and top-k result OVERLAP (Jaccard) — near-1.0 ⇒ the
position-free fulltext2 index is a drop-in for bm25.

  python fulltext/bm25mode_bench.py --config ftcfg/ft2.json --data data/wiki_zh_20k
"""
import argparse
import json
import re
import statistics
import time

import jieba
import pymysql

jieba.setLogLevel(60)
CJK = re.compile(r"[㐀-鿿豈-﫿]")


def connect(cfg):
    return pymysql.connect(host=cfg["host"], port=cfg["port"], user=cfg["user"],
                           password=cfg["password"], autocommit=True, local_infile=True)


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "''")


def cjk_tokens(sentence, want):
    """Up to `want` distinct multi-char CJK jieba tokens from a sentence."""
    out, seen = [], set()
    for tok in jieba.cut(sentence):
        if len(tok) >= 1 and CJK.match(tok) and tok not in seen:
            seen.add(tok)
            out.append(tok)
            if len(out) >= want:
                break
    return out


def build(cfg, csv, engine):
    ddl = {
        "ft2pf": "CREATE FULLTEXT2 INDEX ft ON t(body) WITH PARSER gojieba POSITION_FREE = TRUE",
        "bm25": "CREATE INDEX ft USING bm25 ON t(body) WITH PARSER gojieba",
    }[engine]
    probe = {
        "ft2pf": "SELECT COUNT(*) FROM (SELECT id FROM t WHERE MATCH(body) AGAINST('的' IN BOOLEAN MODE) LIMIT 1) x",
        "bm25": "SELECT COUNT(*) FROM (SELECT id FROM t WHERE bm25(body) AGAINST('的') LIMIT 1) x",
    }[engine]
    db = f"bm25mode_{engine}"
    conn = connect(cfg)
    cur = conn.cursor()
    cur.execute("SET experimental_fulltext2_index=1")
    cur.execute("SET experimental_bm25_index=1")
    cur.execute(f"DROP DATABASE IF EXISTS `{db}`")
    cur.execute(f"CREATE DATABASE `{db}`")
    cur.execute(f"USE `{db}`")
    cur.execute("CREATE TABLE t (id BIGINT PRIMARY KEY, body TEXT)")
    print(f"[{engine}] loading {csv} ...", flush=True)
    cur.execute(f"LOAD DATA LOCAL INFILE '{csv}' INTO TABLE t "
                f"FIELDS TERMINATED BY ',' ENCLOSED BY '\"' LINES TERMINATED BY '\\n' (id, body)")
    print(f"[{engine}] {ddl} ...", flush=True)
    t0 = time.time()
    cur.execute(ddl)
    while time.time() - t0 < 240:
        cur.execute(probe)
        if cur.fetchone()[0] > 0:
            break
        time.sleep(0.5)
    print(f"[{engine}] searchable in {time.time()-t0:.1f}s", flush=True)
    return conn, cur, db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/wiki_zh_20k")
    ap.add_argument("--toks", type=int, default=4, help="query tokens per query")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    csv = args.data + ".csv"
    oracle = json.load(open(args.data + ".oracle.json"))
    if args.limit:
        oracle = oracle[:args.limit]

    # queries: jieba tokens from each oracle sentence; the raw joined form for bm25,
    # the space-separated token form for fulltext2-pf.
    queries = []
    for e in oracle:
        toks = cjk_tokens(e["q"], args.toks)
        if len(toks) >= 2:
            queries.append(toks)
    print(f"bag-of-words retrieval: {len(queries)} queries, {args.toks} toks, k={args.k}\n", flush=True)

    conns, curs = {}, {}
    for engine in ("ft2pf", "bm25"):
        conn, cur, _ = build(cfg, csv, engine)
        conns[engine], curs[engine] = conn, cur

    lat = {"ft2pf": [], "bm25": []}
    resd = {"ft2pf": [], "bm25": []}
    for toks in queries:
        ftq = esc(" ".join(toks))          # space-separated → disjunction → searchWAND
        bmq = esc("".join(toks))           # raw joined → bm25 tokenizes internally
        for engine, cur in curs.items():
            if engine == "ft2pf":
                sql = f"SELECT id FROM t WHERE MATCH(body) AGAINST('{ftq}' IN BOOLEAN MODE) LIMIT {args.k}"
            else:
                sql = f"SELECT id FROM t WHERE bm25(body) AGAINST('{bmq}') LIMIT {args.k}"
            t0 = time.time()
            cur.execute(sql)
            ids = [r[0] for r in cur.fetchall()]
            lat[engine].append((time.time() - t0) * 1000.0)
            resd[engine].append(set(ids))

    overlaps = [len(a & b) / max(1, len(a | b)) for a, b in zip(resd["ft2pf"], resd["bm25"])]

    print(f"\n======== BAG-OF-WORDS RETRIEVAL ({args.data}, gojieba) ========")
    for e in ("ft2pf", "bm25"):
        L = sorted(lat[e])
        name = "fulltext2 POSITION_FREE" if e == "ft2pf" else "bm25"
        print(f"  {name:24s} p50={L[len(L)//2]:6.2f}ms  avg={statistics.mean(lat[e]):6.2f}ms  "
              f"p95={L[int(len(L)*0.95)]:6.2f}ms")
    print(f"  top-{args.k} result overlap (Jaccard): mean={statistics.mean(overlaps):.3f}  "
          f"median={statistics.median(sorted(overlaps)):.3f}")
    print("  (near-1.0 ⇒ position-free fulltext2 is a drop-in for bm25 on ranked retrieval)")
    print("=" * 58)

    for engine, cur in curs.items():
        cur.execute(f"DROP DATABASE `bm25mode_{engine}`")
        conns[engine].close()


if __name__ == "__main__":
    main()

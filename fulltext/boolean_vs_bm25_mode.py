#!/usr/bin/env python3
"""
IN BOOLEAN MODE vs IN BM25 MODE on the SAME positional fulltext2 index (gojieba).

A CJK query on a positional fulltext2 index answers two different questions depending on
the mode, so this measures the latency (and result-set size) of each on identical queries:

  IN BOOLEAN MODE   MATCH(body) AGAINST('我家有三个人' IN BOOLEAN MODE)
      A bare CJK operand is an EXACT POSITIONAL PHRASE — the tokens must be adjacent, in
      order. Decodes + intersects positions (matchPhrase). Precise, smaller result set.

  IN BM25 MODE      MATCH(body) AGAINST('我家有三个人' IN BM25 MODE)
      Bag-of-words: each token an OR term, ranked (searchWAND). Never touches positions.
      A superset, and cheaper per doc (no positional intersection).

Same index, same queries — so the delta is purely phrase-positional vs bag-of-words work.

  python fulltext/boolean_vs_bm25_mode.py --config ftcfg/ft2.json --data data/wiki_zh_2f
"""
import argparse
import json
import re
import statistics
import time

import pymysql

CJK_RUN = re.compile(r"[㐀-鿿豈-﫿]{2,}")


def connect(cfg):
    return pymysql.connect(host=cfg["host"], port=cfg["port"], user=cfg["user"],
                           password=cfg["password"], autocommit=True, local_infile=True)


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "''")


def cjk_window(sentence, w):
    runs = CJK_RUN.findall(sentence)
    if not runs:
        return None
    run = max(runs, key=len)
    if len(run) < w:
        return None
    start = (len(run) - w) // 2
    return run[start:start + w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/wiki_zh_2f")
    ap.add_argument("--width", type=int, default=6, help="CJK phrase window length (chars)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    csv = args.data + ".csv"
    oracle = json.load(open(args.data + ".oracle.json"))
    if args.limit:
        oracle = oracle[:args.limit]

    phrases = [w for w in (cjk_window(e["q"], args.width) for e in oracle) if w]

    conn = connect(cfg)
    cur = conn.cursor()
    cur.execute("SET experimental_fulltext2_index=1")
    cur.execute("DROP DATABASE IF EXISTS bvm")
    cur.execute("CREATE DATABASE bvm")
    cur.execute("USE bvm")
    cur.execute("CREATE TABLE t (id BIGINT PRIMARY KEY, body TEXT)")
    print(f"loading {csv} ...", flush=True)
    cur.execute(f"LOAD DATA LOCAL INFILE '{csv}' INTO TABLE t "
                f"FIELDS TERMINATED BY ',' ENCLOSED BY '\"' LINES TERMINATED BY '\\n' (id, body)")
    print("CREATE FULLTEXT2 INDEX (positional, gojieba) ...", flush=True)
    t0 = time.time()
    cur.execute("CREATE FULLTEXT2 INDEX ft ON t(body) WITH PARSER gojieba")
    print(f"searchable in {time.time()-t0:.1f}s; {len(phrases)} queries, k={args.k}\n", flush=True)

    modes = {"BOOLEAN (phrase)": "IN BOOLEAN MODE", "BM25 (bag-of-words)": "IN BM25 MODE"}
    lat = {m: [] for m in modes}
    cand = {m: [] for m in modes}
    for ph in phrases:
        p = esc(ph)
        for name, kw in modes.items():
            t1 = time.time()
            cur.execute(f"SELECT id FROM t WHERE MATCH(body) AGAINST('{p}' {kw}) LIMIT {args.k}")
            cur.fetchall()
            lat[name].append((time.time() - t1) * 1000.0)
            cur.execute(f"SELECT COUNT(*) FROM (SELECT id FROM t WHERE MATCH(body) AGAINST('{p}' {kw}) LIMIT 1000000) x")
            cand[name].append(cur.fetchone()[0])

    print(f"======== BOOLEAN vs BM25 MODE ({args.data}, {args.width}-char CJK, gojieba) ========")
    for name in modes:
        L = sorted(lat[name])
        print(f"  {name:22s} p50={L[len(L)//2]:6.2f}ms  avg={statistics.mean(lat[name]):6.2f}ms  "
              f"p95={L[int(len(L)*0.95)]:6.2f}ms  cand(med)={int(statistics.median(cand[name])):>8d}")
    print("  BOOLEAN = exact positional phrase (precise, positional cost); "
          "BM25 = bag-of-words (superset, no positions).")
    print("=" * 58)

    cur.execute("DROP DATABASE bvm")
    conn.close()


if __name__ == "__main__":
    main()

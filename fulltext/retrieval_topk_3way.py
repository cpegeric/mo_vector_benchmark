#!/usr/bin/env python3
"""
Ranked top-k RETRIEVAL benchmark across all three fulltext-family indexes:

  fulltext   MATCH(body) AGAINST('t1 t2 ...' IN BOOLEAN MODE) LIMIT k   (WAND-free classic)
  fulltext2  MATCH(body) AGAINST('t1 t2 ...' IN BOOLEAN MODE) LIMIT k   (Block-Max WAND)
  bm25       bm25(body) AGAINST('t1 t2 ...')                   LIMIT k   (position-free BM25 WAND)

Each engine runs its NATIVE ranked-retrieval query over the SAME corpus + the SAME
multi-term (OR) workload, at a few k. bm25 is a position-free bag-of-words ranked index
(no boolean/phrase) so it uses the `bm25()` verb; fulltext/fulltext2 use boolean-OR
MATCH with a LIMIT the planner pushes into the search TVF. Latency is the comparison —
the three don't return identical sets (bm25 is pure bag-of-words; boolean-OR is a
match-set then top-k), which is expected for different retrieval models.

  python fulltext/retrieval_topk_3way.py --config ftcfg/ft2.json --data data/wiki_en_20k
"""
import argparse
import json
import statistics
import sys
import time

import pymysql

# engine -> (index DDL template ({parser}), ranked query template, searchable-probe template)
ENGINES = {
    "fulltext":  ("CREATE FULLTEXT INDEX ft ON t(body) WITH PARSER {parser}",
                  "SELECT id FROM t WHERE MATCH(body) AGAINST('{q}' IN BOOLEAN MODE) LIMIT {k}",
                  "SELECT COUNT(*) FROM t WHERE MATCH(body) AGAINST('{p}' IN BOOLEAN MODE)"),
    "fulltext2": ("CREATE FULLTEXT2 INDEX ft ON t(body) WITH PARSER {parser}",
                  "SELECT id FROM t WHERE MATCH(body) AGAINST('{q}' IN BOOLEAN MODE) LIMIT {k}",
                  "SELECT COUNT(*) FROM t WHERE MATCH(body) AGAINST('{p}' IN BOOLEAN MODE)"),
    "bm25":      ("CREATE INDEX ft USING bm25 ON t(body) WITH PARSER {parser}",
                  "SELECT id FROM t WHERE bm25(body) AGAINST('{q}') LIMIT {k}",
                  "SELECT COUNT(*) FROM (SELECT id FROM t WHERE bm25(body) AGAINST('{p}') LIMIT 1) x"),
}


def connect(cfg):
    return pymysql.connect(host=cfg["host"], port=cfg["port"], user=cfg["user"],
                           password=cfg["password"], autocommit=True, local_infile=True)


def q1(cur, sql):
    cur.execute(sql)
    r = cur.fetchone()
    return r[0] if r else 0


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "''")


def run_engine(cfg, csv, queries, engine, ks, parser):
    ddl_tmpl, qtmpl, ptmpl = ENGINES[engine]
    ddl = ddl_tmpl.format(parser=parser)
    db = f"topk3_{engine}"
    conn = connect(cfg)
    cur = conn.cursor()
    cur.execute("SET experimental_fulltext2_index=1")
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
    probe = esc(queries[0]["terms"].split()[0])
    # time-until-searchable (fair to sync fulltext/fulltext2 and async-CDC bm25)
    while time.time() - t0 < 180:
        if q1(cur, ptmpl.format(p=probe)) > 0:
            break
        time.sleep(0.5)
    build_s = time.time() - t0
    print(f"[{engine}] searchable in {build_s:.1f}s", flush=True)

    out = {"build_s": build_s}
    for k in ks:
        lats = []
        for qd in queries:
            sql = qtmpl.format(q=esc(qd["terms"]), k=k)
            t1 = time.time()
            cur.execute(sql)
            cur.fetchall()
            lats.append((time.time() - t1) * 1000.0)
        out[k] = lats
        print(f"[{engine}] k={k}: {len(queries)} queries done", flush=True)

    cur.execute(f"DROP DATABASE `{db}`")
    conn.close()
    return out


def pctl(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(len(a) * p / 100.0))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/wiki_en_20k")
    ap.add_argument("--engines", default="fulltext,fulltext2,bm25")
    ap.add_argument("--parser", default="gojieba", help="index tokenizer: gojieba (word) or ngram (CJK 3-gram)")
    ap.add_argument("--ks", default="10,100,1000")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    csv = args.data + ".csv"
    allq = json.load(open(args.data + ".queries.json"))
    queries = [x for x in allq if x.get("nterms", 1) > 1]  # multi-term → OR disjunction
    ks = [int(x) for x in args.ks.split(",")]
    engines = [e.strip() for e in args.engines.split(",")]

    print(f"ranked top-k: {len(queries)} multi-term queries, k={ks}, engines={engines}, parser={args.parser}\n", flush=True)
    res = {e: run_engine(cfg, csv, queries, e, ks, args.parser) for e in engines}

    print(f"\n============= RANKED TOP-K RETRIEVAL ({args.data}, parser={args.parser}) =============")
    print(f"  build (time-until-searchable):")
    for e in engines:
        print(f"    {e:10s} {res[e]['build_s']:6.1f}s")
    for k in ks:
        print(f"  --- k={k} ---")
        for e in engines:
            lat = res[e][k]
            print(f"    {e:10s} avg={statistics.mean(lat):7.2f}ms  p50={pctl(lat,50):7.2f}  "
                  f"p95={pctl(lat,95):7.2f}  p99={pctl(lat,99):7.2f}  total={sum(lat)/1000:5.2f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()

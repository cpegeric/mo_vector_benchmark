#!/usr/bin/env python3
"""
CJK phrase-precision test: fulltext2 (positional) vs bm25 (position-free), both on the
gojieba parser. This is where the two engines genuinely DIVERGE — and for Chinese it
shows up on an ORDINARY query, not a quoted phrase.

A Chinese query string gojieba-tokenizes into several tokens, e.g.

    我家有三个人   ->   我家 | 有 | 三个 | 人

fulltext2 treats that multi-token CJK operand as an EXACT POSITIONAL PHRASE (the tokens
must be adjacent, in order); bm25 has no positions, so it compares the tokens
BAG-OF-WORDS (any doc containing them, any order, any distance). A 1-token (<=3 char)
query is identical on both; the divergence begins at >=4 chars, where the string spans
>= 2 tokens.

  fulltext2   MATCH(body) AGAINST('<cjk>' IN BOOLEAN MODE)   exact phrase (automatic for CJK)
  bm25        bm25(body) AGAINST('<cjk>')                    bag-of-words

For each oracle sentence (Chinese sentence + the doc id it came from) we take a
contiguous W-character window and ask each engine for it. Metrics per engine:
  P@1 / MRR   does the true source doc rank at the top
  cand        how many docs the engine matches (bm25's superset is far larger)

  python fulltext/phrase_precision.py --config ftcfg/ft2.json --data data/wiki_zh_2f
"""
import argparse
import json
import re
import statistics
import time

import pymysql

# maximal runs of CJK characters (a Chinese "phrase" is a contiguous char run, no spaces)
CJK_RUN = re.compile(r"[㐀-鿿豈-﫿]{2,}")


def connect(cfg):
    return pymysql.connect(host=cfg["host"], port=cfg["port"], user=cfg["user"],
                           password=cfg["password"], autocommit=True, local_infile=True)


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "''")


def cjk_window(sentence, w):
    """A contiguous w-character CJK window taken from the sentence's longest CJK run."""
    runs = CJK_RUN.findall(sentence)
    if not runs:
        return None
    run = max(runs, key=len)
    if len(run) < w:
        return None
    start = (len(run) - w) // 2  # middle window = more distinctive than the edges
    return run[start:start + w]


def build_index(cfg, csv, engine):
    ddl = {
        "fulltext2": "CREATE FULLTEXT2 INDEX ft ON t(body) WITH PARSER gojieba",
        "bm25": "CREATE INDEX ft USING bm25 ON t(body) WITH PARSER gojieba",
    }[engine]
    probe = {
        "fulltext2": "SELECT COUNT(*) FROM t WHERE MATCH(body) AGAINST('{p}' IN BOOLEAN MODE)",
        "bm25": "SELECT COUNT(*) FROM (SELECT id FROM t WHERE bm25(body) AGAINST('{p}') LIMIT 1) x",
    }[engine]
    db = f"pp_{engine}"
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
    while time.time() - t0 < 300:
        cur.execute(probe.format(p="的"))
        if cur.fetchone()[0] > 0:
            break
        time.sleep(0.5)
    print(f"[{engine}] searchable in {time.time()-t0:.1f}s", flush=True)
    return conn, cur, db


def rank_and_cand(cur, engine, phrase, k):
    if engine == "fulltext2":
        rank_sql = f"SELECT id FROM t WHERE MATCH(body) AGAINST('{esc(phrase)}' IN BOOLEAN MODE) LIMIT {k}"
        cand_sql = f"SELECT COUNT(*) FROM t WHERE MATCH(body) AGAINST('{esc(phrase)}' IN BOOLEAN MODE)"
    else:
        rank_sql = f"SELECT id FROM t WHERE bm25(body) AGAINST('{esc(phrase)}') LIMIT {k}"
        cand_sql = (f"SELECT COUNT(*) FROM (SELECT id FROM t WHERE bm25(body) "
                    f"AGAINST('{esc(phrase)}') LIMIT 1000000) x")
    t0 = time.time()
    cur.execute(rank_sql)
    ids = [r[0] for r in cur.fetchall()]
    ms = (time.time() - t0) * 1000.0
    cur.execute(cand_sql)
    cand = cur.fetchone()[0]
    return ids, cand, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/wiki_zh_2f")
    ap.add_argument("--widths", default="3,4,6,8", help="CJK phrase window lengths (chars)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    csv = args.data + ".csv"
    oracle = json.load(open(args.data + ".oracle.json"))
    if args.limit:
        oracle = oracle[:args.limit]
    widths = [int(x) for x in args.widths.split(",")]

    curs, conns = {}, {}
    for engine in ("fulltext2", "bm25"):
        conn, cur, _ = build_index(cfg, csv, engine)
        conns[engine], curs[engine] = conn, cur

    # phrase sets per width (same phrases for both engines)
    sets = {}
    for w in widths:
        cases = []
        for e in oracle:
            ph = cjk_window(e["q"], w)
            if ph and e.get("ids"):
                cases.append((ph, e["ids"][0]))
        sets[w] = cases

    print(f"\n=========== CJK PHRASE PRECISION ({args.data}) ===========")
    print("  fulltext2 = exact positional phrase | bm25 = bag-of-words (gojieba tokens)\n")
    for w in widths:
        cases = sets[w]
        print(f"  --- {w}-char CJK window (n={len(cases)}, k={args.k}) ---")
        print(f"    {'engine':10s} {'P@1':>6s} {'MRR':>6s} {'found@k':>8s} "
              f"{'cand(med)':>10s} {'cand(mean)':>11s} {'p50 ms':>8s}")
        for engine, cur in curs.items():
            rr, p1, found, cand, lat = [], 0, 0, [], []
            for phrase, src in cases:
                ids, c, ms = rank_and_cand(cur, engine, phrase, args.k)
                lat.append(ms)
                cand.append(c)
                rank = ids.index(src) + 1 if src in ids else 0
                rr.append(1.0 / rank if rank else 0.0)
                p1 += rank == 1
                found += rank > 0
            n = max(1, len(cases))
            print(f"    {engine:10s} {p1/n:6.2f} {statistics.mean(rr):6.2f} {found/n:8.2f} "
                  f"{int(statistics.median(cand)):10d} {int(statistics.mean(cand)):11d} "
                  f"{sorted(lat)[len(lat)//2]:8.2f}")
        print()

    for engine, cur in curs.items():
        cur.execute(f"DROP DATABASE `pp_{engine}`")
        conns[engine].close()
    print("  Higher P@1/MRR = exact source doc at the top. bm25's cand (candidate set)")
    print("  balloons with window length — the bag-of-words superset — while fulltext2")
    print("  stays pinned to the exact phrase. The gap opens at >=4 chars (>=2 tokens).")
    print("=" * 58)


if __name__ == "__main__":
    main()

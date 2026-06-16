#!/usr/bin/env python3
"""run_matrix.py — wiki_all 1M ivfflat benchmark matrix driver.

Sweeps base column type x index quantization for ivfflat, recording index build
time, recall@k and search latency/QPS. Designed to be re-run under each MO build
config (GPU+SIMD / GPU+non-SIMD / non-GPU+SIMD / non-GPU+non-SIMD); the imported
tables persist on disk across MO restarts, so --phase import runs ONCE and
--phase matrix runs per build config.

  python run_matrix.py --phase import                      # once (any build)
  python run_matrix.py --phase matrix --tag gpu_simd       # per build config
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/eric/github/vector_benchmark/wiki_all_1M"
F32_PREFIX = f"{DATA}/gen/base.1M"                 # base.1M0..3.csv (f32 text)
I8_PREFIX  = f"{DATA}/gen_narrow/base.i8_"          # base.i8_0.csv  (int8 scaled)
U8_PREFIX  = f"{DATA}/gen_narrow/base.u8_"          # base.u8_0.csv  (uint8 scaled)

# base_type -> (database, load csv prefix)
BASES = {
    "f32":  ("wiki_f32",  F32_PREFIX),
    "f16":  ("wiki_f16",  F32_PREFIX),
    "bf16": ("wiki_bf16", F32_PREFIX),
    "int8": ("wiki_i8",   I8_PREFIX),
    "uint8":("wiki_u8",   U8_PREFIX),
}
QUANTS = ["float32", "float16", "bf16", "int8", "uint8"]

# Matrix cells: base sweep (no quantization → entries keep the base column type)
# + quant sweep (base=f32, entries downcast to the named quantization).
# NOTE: base cells must NOT pass quantization='float32'. That overrides the entry
# type to f32, so a narrow base (bf16/f16/int8/uint8) would store f32 entries and
# the re-rank would run the f32 distance kernel over upcast data — measuring an f32
# index, not the narrow one (and now an outright error, since upcasting is rejected
# in schema.go). 'none' omits the QUANTIZATION clause so entries = base type.
def cells():
    out = [("base", b, "none") for b in BASES]                    # 5 base-sweep
    out += [("quant", "f32", q) for q in QUANTS if q != "float32"] # 4 quant-sweep
    return out

DATASET = {
    "query_fbin": f"{DATA}/queries.fbin",
    "groundtruth_ibin": f"{DATA}/groundtruth.1M.neighbors.ibin",
    "id_offset": 1,
}
LISTS = 1000
PROBE = 8           # probe_limit: ~0.91 recall@10, ~0.9s/query (32 was ~4s/q, I/O-bound)
N_QUERIES = 200
K = 10
CONCURRENCY = 8
PASSES = 4          # query passes per cell: pass1=cold, warm = median(pass2..N)


def make_cfg(base_type, quant, database):
    return {
        "host": "127.0.0.1", "port": 6001, "user": "dump", "password": "111",
        "database": database, "table": "hfb",
        "base_type": base_type, "dimension": 768,
        "index": {
            "name": "idx_l2", "type": "ivfflat", "lists": LISTS,
            # 'none' (base sweep) omits the QUANTIZATION clause so the index entries
            # keep the base column type; a real quant name downcasts the entries.
            "op_type": "vector_l2_ops", "quantization": ("" if quant == "none" else quant),
            "kmeans_train_percent": 10, "kmeans_max_iteration": 20,
        },
        "env": {"probe_limit": PROBE, "ivf_preload_entries": 0},
        "dataset": DATASET,
    }


def write_cfg(cfg, name):
    d = os.path.join(HERE, "cfg", "bench")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    json.dump(cfg, open(p, "w"), indent=2)
    return p


def run(cmd, **kw):
    print("  $ " + " ".join(cmd[-4:]), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    dt = time.perf_counter() - t0
    return r.returncode, r.stdout + r.stderr, dt


def py(*a):
    return [sys.executable, os.path.join(HERE, "run_wiki.py"), *a]


def parse_recall(out):
    m = {}
    for pat, key in [(r"avg recall@\d+\s*=\s*([\d.]+)", "recall"),
                     (r"^QPS\s*=\s*([\d.]+)", "qps"),
                     (r"avg\s*=\s*([\d.]+)\s*ms", "lat_avg"),
                     (r"p50\s*=\s*([\d.]+)\s*ms", "lat_p50"),
                     (r"p95\s*=\s*([\d.]+)\s*ms", "lat_p95"),
                     (r"p99\s*=\s*([\d.]+)\s*ms", "lat_p99")]:
        g = re.search(pat, out, re.M)
        if g:
            m[key] = float(g.group(1))
    return m


def phase_import():
    results = {}
    for bt, (db, prefix) in BASES.items():
        cfg = make_cfg(bt, "float32", db)
        p = write_cfg(cfg, f"import_{bt}.json")
        print(f"\n===== IMPORT base={bt} db={db} prefix={prefix} =====", flush=True)
        rc, out, _ = run(py("create_table", "--config", p))
        if rc:
            print(out[-2000:]); print(f"!! create_table failed base={bt}"); continue
        rc, out, dt = run(py("import", "--config", p, "--input-csv-prefix", prefix))
        ok = rc == 0 and "失败" not in out
        aff = re.findall(r"affected_rows=(\d+)", out)
        rows = sum(int(x) for x in aff)
        print(f"   import base={bt}: rows={rows} time={dt:.1f}s ok={ok}")
        if not ok:
            print(out[-2000:])
        results[bt] = {"db": db, "rows": rows, "import_s": round(dt, 1), "ok": ok}
    json.dump(results, open(os.path.join(HERE, "bench_import.json"), "w"), indent=2)
    print("\nIMPORT SUMMARY:", json.dumps(results, indent=2))


def phase_matrix(tag):
    global PASSES
    rows = []
    for kind, base, quant in cells():
        db, _ = BASES[base]
        cfg = make_cfg(base, quant, db)
        name = f"{tag}_{kind}_{base}_{quant}.json"
        p = write_cfg(cfg, name)
        label = f"{kind}: base={base} quant={quant} db={db}"
        print(f"\n===== {label} =====", flush=True)
        # drop any existing index, then build (timed), then recall.
        run(py("drop_index", "--config", p))
        rc, out, build_s = run(py("create_index", "--config", p))
        if rc:
            print(out[-1500:]); rows.append({"tag": tag, "kind": kind, "base": base,
                "quant": quant, "build_s": None, "error": "build_failed"}); continue
        def run_recall():
            if base in ("int8", "uint8"):
                # int8/uint8 base columns need integer-scaled query literals (the f32
                # query can't cast to the narrow column); recall_narrow.py handles it.
                return run([sys.executable, os.path.join(HERE, "recall_narrow.py"),
                    "--database", db, "--table", "hfb", "--mode", base,
                    "--query-fbin", DATASET["query_fbin"], "--groundtruth-ibin", DATASET["groundtruth_ibin"],
                    "-n", str(N_QUERIES), "-k", str(K), "--probe", str(PROBE),
                    "--concurrency", str(CONCURRENCY), "--id-offset", str(DATASET["id_offset"])])
            return run(py("recall", "--config", p, "--gt-source", "fbin",
                         "-n", str(N_QUERIES), "-k", str(K), "--concurrency", str(CONCURRENCY)))

        # Cold/warm: pass 1 reads the freshly-built (uncached) index entry blocks
        # cold; later passes hit the warm fileservice cache. The build runs with
        # SkipMemoryCacheWrites, so pass 1 is a true cold read. Warm QPS/p50 is the
        # MEDIAN of the post-cold passes (not the single last pass) — a transient on
        # any one warm pass otherwise tanks the cell (e.g. f32 once read 61 vs ~457).
        passes_met = []
        for pi in range(PASSES):
            _rc, out, _ = run_recall()
            m = parse_recall(out)
            passes_met.append(m)
            print(f"   pass{pi+1}: qps={m.get('qps')} recall@{K}={m.get('recall')} "
                  f"p50={m.get('lat_p50')}ms", flush=True)

        def _median(vals):
            vals = sorted(v for v in vals if v is not None)
            return vals[len(vals) // 2] if vals else None

        cold = passes_met[0]
        warm_passes = passes_met[1:] or passes_met  # drop cold; fallback if PASSES==1
        warm_qps = _median([m.get("qps") for m in warm_passes])
        warm_p50 = _median([m.get("lat_p50") for m in warm_passes])
        warm_recall = next((m.get("recall") for m in warm_passes if m.get("recall") is not None),
                           cold.get("recall"))
        rec = {"tag": tag, "kind": kind, "base": base, "quant": quant,
               "build_s": round(build_s, 1),
               "recall": warm_recall,
               "qps_cold": cold.get("qps"), "qps_warm": warm_qps,
               "p50_cold": cold.get("lat_p50"), "p50_warm": warm_p50,
               "qps_passes": [m.get("qps") for m in passes_met]}
        print(f"   -> build={build_s:.1f}s recall@{K}={warm_recall} "
              f"qps_cold={cold.get('qps')} qps_warm={warm_qps} (median of {len(warm_passes)} warm) "
              f"p50: {cold.get('lat_p50')}->{warm_p50}ms")
        rows.append(rec)
        run(py("drop_index", "--config", p))
        # checkpoint after each cell
        json.dump(rows, open(os.path.join(HERE, f"bench_matrix_{tag}.json"), "w"), indent=2)
    print(f"\nMATRIX[{tag}] SUMMARY:\n" + json.dumps(rows, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["import", "matrix"])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--passes", type=int, default=PASSES,
                    help="query passes per cell (pass1=cold, last=warm)")
    a = ap.parse_args()
    if a.phase == "import":
        phase_import()
    else:
        PASSES = a.passes
        phase_matrix(a.tag)

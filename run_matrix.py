#!/usr/bin/env python3
"""run_matrix.py — wiki_all benchmark matrix driver (ivfflat / ivfpq / cagra).

Sweeps base column type x index quantization for ONE index algorithm, recording
index build time, recall@k and search latency/QPS.

run_matrix.py orchestrates run_wiki.py (the per-operation tool) for each cell:
  drop_index -> create_index (timed) -> recall x N passes -> drop_index.
run_wiki.py is config-driven and algorithm-agnostic; run_matrix.py is the sweep
driver that GENERATES a per-cell config and calls run_wiki.py for each step.

Scale (1M/10M/88M), GPU distribution mode (single/sharded) and the dataset root
are configurable. Per-scale tuning (lists, probe, graph degrees) and dataset
paths come from a committed template: cfg/templates/<scale>.json. Dataset paths
in the template are RELATIVE to --data-root, so templates stay machine-neutral.

  # import once per (scale, algo-base-set); tables persist across MO restarts
  python run_matrix.py --phase import --scale 1M --algo ivfpq
  # run the matrix per algorithm / distribution
  python run_matrix.py --phase matrix --scale 1M  --algo ivfpq
  python run_matrix.py --phase matrix --scale 10M --algo cagra --distribution sharded

Generated per-cell configs are written to cfg/generated/ (git-ignored).
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
# Dataset root for the RELATIVE paths in cfg/templates/<scale>.json. Defaults to
# this repo dir, so datasets live under (or are symlinked into) mo_vector_benchmark
# itself. Override with --data-root or $WIKI_DATA_ROOT.
DEFAULT_DATA_ROOT = os.environ.get("WIKI_DATA_ROOT", HERE)

QUANTS = ["float32", "float16", "bf16", "int8", "uint8"]

N_QUERIES = 200
K = 20
CONCURRENCY = 8
PASSES = 4          # query passes per cell: pass1=cold, warm = median(pass2..N)


# Matrix cells: base sweep (no quantization → entries keep the base column type)
# + quant sweep (base=f32/f16, entries downcast to the named quantization).
# NOTE: base cells must NOT pass quantization='float32'. That overrides the entry
# type to f32, so a narrow base (bf16/f16/int8/uint8) would store f32 entries and
# the re-rank would run the f32 distance kernel over upcast data — measuring an f32
# index, not the narrow one (and an outright error, since upcasting is rejected in
# schema.go). 'none' omits the QUANTIZATION clause so entries = base type.
def cells(algo):
    if algo == "ivfflat":
        # CPU ivfflat supports every base type + every narrow quantization.
        out = [("base", b, "none") for b in ["f32", "f16", "bf16", "int8", "uint8"]]
        out += [("quant", "f32", q) for q in QUANTS if q != "float32"]   # 4 quant
        return out
    # cuvs (ivfpq/cagra): base column is f32/f16 only; bf16 has no GPU storage;
    # int8/uint8 are L2-only (op_type is l2 here) and cannot be a base column.
    # So: f32/f16 base (native), + f16/int8/uint8 quant on f32 base,
    # + int8/uint8 quant on f16 base. (No bf16/int8/uint8 base, no bf16 quant.)
    out = [("base", "f32", "none"), ("base", "f16", "none")]
    out += [("quant", "f32", q) for q in ["float16", "int8", "uint8"]]
    out += [("quant", "f16", q) for q in ["int8", "uint8"]]
    return out


def load_template(scale):
    p = os.path.join(HERE, "cfg", "templates", f"{scale}.json")
    if not os.path.exists(p):
        raise SystemExit(f"template not found: {p}")
    return json.load(open(p))


def dbname(scale, base):
    return f"wiki{scale.lower()}_{base}"


def abspath(data_root, rel):
    return rel if os.path.isabs(rel) else os.path.join(data_root, rel)


def csv_prefix_for(base, ds, data_root):
    # f32/f16/bf16 load from the f32 text CSV (the column type does the cast);
    # int8/uint8 need integer-scaled CSVs (separate narrow set).
    if base == "int8":
        rel = ds.get("csv_prefix_int8")
    elif base == "uint8":
        rel = ds.get("csv_prefix_uint8")
    else:
        rel = ds.get("csv_prefix")
    return abspath(data_root, rel) if rel else None


def make_cfg(algo, base_type, quant, database, tmpl, data_root, distribution):
    ds, tun = tmpl["dataset"], tmpl["tuning"]
    qval = "" if quant == "none" else quant
    cfg = {
        "host": tmpl.get("host", "127.0.0.1"), "port": tmpl.get("port", 6001),
        "user": tmpl.get("user", "dump"), "password": tmpl.get("password", "111"),
        "database": database, "table": "hfb",
        "base_type": base_type, "dimension": tmpl.get("dimension", 768),
        "dataset": {
            "query_fbin": abspath(data_root, ds["query_fbin"]),
            "groundtruth_ibin": abspath(data_root, ds["groundtruth_ibin"]),
            "id_offset": ds.get("id_offset", 1),
        },
    }
    # 'none' (base sweep) omits QUANTIZATION so entries keep the base column type;
    # a real quant name downcasts the entries.
    if algo == "ivfflat":
        cfg["index"] = {
            "name": "idx_l2", "type": "ivfflat", "lists": tun["ivfflat_lists"],
            "op_type": "vector_l2_ops", "quantization": qval,
            "kmeans_train_percent": tun.get("kmeans_train_percent", 10),
            "kmeans_max_iteration": tun.get("kmeans_max_iteration", 20),
        }
        cfg["env"] = {"probe_limit": tun["probe_ivfflat"], "ivf_preload_entries": 0}
    elif algo == "ivfpq":
        cfg["index"] = {
            "name": "idx_l2", "type": "ivfpq", "lists": tun["ivfpq_lists"],
            "m": tun.get("ivfpq_m", 192), "bits_per_code": tun.get("ivfpq_bits_per_code", 8),
            "op_type": "vector_l2_ops", "quantization": qval,
            "distribution_mode": distribution, "max_index_capacity": 0,
            "kmeans_train_percent": tun.get("kmeans_train_percent", 10),
        }
        cfg["env"] = {"experimental_ivfpq_index": 1, "probe_limit": tun["probe_ivfpq"],
                      "ivfpq_batch_window": 0}
    elif algo == "cagra":
        cg = tun["cagra"]
        cfg["index"] = {
            "name": "idx_l2", "type": "cagra",
            "intermediate_graph_degree": cg["intermediate_graph_degree"],
            "graph_degree": cg["graph_degree"], "itopk_size": cg["itopk_size"],
            "op_type": "vector_l2_ops", "quantization": qval,
            "distribution_mode": distribution, "max_index_capacity": 0,
        }
        cfg["env"] = {"experimental_cagra_index": 1, "cagra_batch_window": 0,
                      "cagra_threads_build": 0, "cagra_threads_search": 0}
    else:
        raise SystemExit(f"unknown algo {algo}")
    return cfg


def write_cfg(cfg, name):
    d = os.path.join(HERE, "cfg", "generated")
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


def import_base(algo, base, db, scale, tmpl, data_root, distribution):
    """create_table (drops+recreates db) + load data for one base type.
    Returns (ok, rows)."""
    prefix = csv_prefix_for(base, tmpl["dataset"], data_root)
    if not prefix:
        print(f"!! no csv prefix configured for base={base}; skip"); return False, 0
    cfg = make_cfg(algo, base, "float32", db, tmpl, data_root, distribution)
    p = write_cfg(cfg, f"import_{scale}_{base}.json")
    rc, out, _ = run(py("create_table", "--config", p))
    if rc:
        print(out[-2000:]); print(f"!! create_table failed base={base}"); return False, 0
    rc, out, dt = run(py("import", "--config", p, "--input-csv-prefix", prefix))
    ok = rc == 0 and "失败" not in out
    rows = sum(int(x) for x in re.findall(r"affected_rows=(\d+)", out))
    print(f"   import base={base}: rows={rows} time={dt:.1f}s ok={ok}")
    if not ok:
        print(out[-2000:])
    return ok, rows


def drop_database(tmpl, db):
    """Drop the per-base database (and its table) to release memory between base
    groups. run_wiki.py has no drop command, so do it directly like create_table."""
    import pymysql
    try:
        c = pymysql.connect(host=tmpl.get("host", "127.0.0.1"), port=tmpl.get("port", 6001),
                            user=tmpl.get("user", "dump"), password=tmpl.get("password", "111"))
        with c.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{db}`")
        c.commit(); c.close()
        print(f"   dropped database {db}")
    except Exception as e:   # noqa: BLE001
        print(f"!! drop database {db} failed: {e}")


def phase_import(algo, scale, tmpl, data_root, distribution):
    # Standalone import for the --keep-tables persistent workflow. Import only the
    # base types this algo's matrix references — GPU algos need just f32/f16.
    results = {}
    for bt in sorted({b for _, b, _ in cells(algo)}):
        db = dbname(scale, bt)
        print(f"\n===== IMPORT scale={scale} base={bt} db={db} =====", flush=True)
        ok, rows = import_base(algo, bt, db, scale, tmpl, data_root, distribution)
        results[bt] = {"db": db, "rows": rows, "ok": ok}
    json.dump(results, open(os.path.join(HERE, f"bench_import_{scale}.json"), "w"), indent=2)
    print(f"\nIMPORT[{scale}] SUMMARY:", json.dumps(results, indent=2))


def run_one_cell(tag, algo, scale, kind, base, quant, db, tmpl, data_root, distribution):
    probe_flat = tmpl["tuning"]["probe_ivfflat"]
    cfg = make_cfg(algo, base, quant, db, tmpl, data_root, distribution)
    name = f"{tag}_{scale}_{kind}_{base}_{quant}.json"
    p = write_cfg(cfg, name)
    label = f"{kind}: base={base} quant={quant} db={db} dist={distribution}"
    print(f"\n===== {label} =====", flush=True)
    # drop any existing index, then build (timed), then recall.
    run(py("drop_index", "--config", p))
    rc, out, build_s = run(py("create_index", "--config", p))
    if rc:
        print(out[-1500:])
        return {"tag": tag, "scale": scale, "kind": kind, "base": base,
                "quant": quant, "build_s": None, "error": "build_failed"}

    def run_recall():
        if base in ("int8", "uint8"):
            # int8/uint8 base columns need integer-scaled query literals (the f32
            # query can't cast to the narrow column); recall_narrow.py handles it.
            return run([sys.executable, os.path.join(HERE, "recall_narrow.py"),
                "--database", db, "--table", "hfb", "--mode", base,
                "--query-fbin", abspath(data_root, tmpl["dataset"]["query_fbin"]),
                "--groundtruth-ibin", abspath(data_root, tmpl["dataset"]["groundtruth_ibin"]),
                "-n", str(N_QUERIES), "-k", str(K), "--probe", str(probe_flat),
                "--concurrency", str(CONCURRENCY),
                "--id-offset", str(tmpl["dataset"].get("id_offset", 1))])
        return run(py("recall", "--config", p, "--gt-source", "fbin",
                     "-n", str(N_QUERIES), "-k", str(K), "--concurrency", str(CONCURRENCY)))

    # Cold/warm: pass 1 reads the freshly-built (uncached) index entry blocks
    # cold; later passes hit the warm fileservice cache. Warm QPS/p50 is the
    # MEDIAN of the post-cold passes (a transient on any one warm pass otherwise
    # tanks the cell).
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
    warm_passes = passes_met[1:] or passes_met
    warm_qps = _median([m.get("qps") for m in warm_passes])
    warm_p50 = _median([m.get("lat_p50") for m in warm_passes])
    warm_recall = next((m.get("recall") for m in warm_passes if m.get("recall") is not None),
                       cold.get("recall"))
    rec = {"tag": tag, "scale": scale, "kind": kind, "base": base, "quant": quant,
           "distribution": distribution, "build_s": round(build_s, 1),
           "recall": warm_recall,
           "qps_cold": cold.get("qps"), "qps_warm": warm_qps,
           "p50_cold": cold.get("lat_p50"), "p50_warm": warm_p50,
           "qps_passes": [m.get("qps") for m in passes_met]}
    print(f"   -> build={build_s:.1f}s recall@{K}={warm_recall} "
          f"qps_cold={cold.get('qps')} qps_warm={warm_qps} (median of {len(warm_passes)} warm) "
          f"p50: {cold.get('lat_p50')}->{warm_p50}ms")
    run(py("drop_index", "--config", p))
    return rec


def phase_matrix(tag, algo, scale, tmpl, data_root, distribution, keep_tables):
    rows = []
    # Group cells by base table, preserving order. The per-base table lifecycle
    # (default) creates+loads a base table, runs all its cells, then DROPS it, so
    # only ONE base table is resident at a time. Without it, every base table for
    # the scale stays alive across the whole matrix (5 tables at once at 88M →
    # OOM). --keep-tables opts into the persistent workflow: run --phase import
    # first; tables are neither created nor dropped here.
    order, by_base = [], {}
    for c in cells(algo):
        b = c[1]
        if b not in by_base:
            by_base[b] = []; order.append(b)
        by_base[b].append(c)

    for base in order:
        db = dbname(scale, base)
        if not keep_tables:
            print(f"\n##### BASE {base} (db={db}) — import #####", flush=True)
            ok, _ = import_base(algo, base, db, scale, tmpl, data_root, distribution)
            if not ok:
                print(f"!! import failed for base={base}; skipping its cells")
                continue
        try:
            for kind, _b, quant in by_base[base]:
                rows.append(run_one_cell(tag, algo, scale, kind, base, quant, db,
                                         tmpl, data_root, distribution))
                # checkpoint after each cell
                json.dump(rows, open(os.path.join(HERE, f"bench_matrix_{tag}.json"), "w"), indent=2)
        finally:
            if not keep_tables:
                drop_database(tmpl, db)   # free the base table before the next base
    print(f"\nMATRIX[{tag}] SUMMARY:\n" + json.dumps(rows, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["import", "matrix"])
    ap.add_argument("--algo", default="ivfflat", choices=["ivfflat", "ivfpq", "cagra"],
                    help="index algorithm; selects index params + valid cell-set")
    ap.add_argument("--scale", default="1M", choices=["1M", "10M", "88M"],
                    help="dataset scale; loads cfg/templates/<scale>.json")
    ap.add_argument("--distribution", default=None, choices=["single", "sharded"],
                    help="GPU distribution mode (ivfpq/cagra); default: template's value. "
                         "Use 'sharded' when one GPU lacks memory for the dataset (e.g. 10M).")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help=f"root for relative dataset paths in the template (default: {DEFAULT_DATA_ROOT})")
    ap.add_argument("--tag", default=None, help="bench_matrix_{tag}.json (default: --algo)")
    ap.add_argument("--passes", type=int, default=PASSES,
                    help="query passes per cell (pass1=cold, warm=median(rest))")
    ap.add_argument("--keep-tables", action="store_true",
                    help="persistent workflow: do NOT import/drop tables in --phase matrix "
                         "(run --phase import first). Default drops each base table after its "
                         "cells so only one base table is resident at a time (avoids OOM).")
    a = ap.parse_args()

    tmpl = load_template(a.scale)
    distribution = a.distribution or tmpl["tuning"].get("distribution_mode", "single")
    tag = a.tag or a.algo
    PASSES = a.passes
    if a.phase == "import":
        phase_import(a.algo, a.scale, tmpl, a.data_root, distribution)
    else:
        phase_matrix(tag, a.algo, a.scale, tmpl, a.data_root, distribution, a.keep_tables)

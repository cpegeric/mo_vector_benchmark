#!/usr/bin/env python3
"""aggregate.py — compare the wiki_all 1M matrix across index algorithms.

Reads bench_matrix_{ivfflat,ivfpq,cagra}.json (written by run_matrix.py
--phase matrix --algo <x>) and prints a per-cell side-by-side of build time,
recall@k, warm QPS and warm p50. cuvs algos (ivfpq/cagra) lack the bf16/int8/
uint8 base cells and the bf16 quant cell (rejected at CREATE), shown as '-'.
"""
import json, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = ["ivfflat", "ivfpq", "cagra"]
LBL = {"ivfflat": "IVFFLAT(cpu)", "ivfpq": "IVF-PQ(gpu)", "cagra": "CAGRA(gpu)"}

data = {}
for t in TAGS:
    p = os.path.join(HERE, f"bench_matrix_{t}.json")
    if os.path.exists(p):
        data[t] = {(r["kind"], r["base"], r["quant"]): r for r in json.load(open(p))}

# Union of cells across algos. Base cells carry quant='none' (entries keep the
# base type); quant cells name the storage downcast applied to an f32/f16 base.
cells = [("base", "f32", "none"), ("base", "f16", "none"), ("base", "bf16", "none"),
         ("base", "int8", "none"), ("base", "uint8", "none"),
         ("quant", "f32", "float16"), ("quant", "f32", "bf16"),
         ("quant", "f32", "int8"), ("quant", "f32", "uint8"),
         ("quant", "f16", "int8"), ("quant", "f16", "uint8")]


def cellname(k):
    kind, base, quant = k
    return f"base={base}" if kind == "base" else f"q:{base}->{quant}"


for metric, fmt in [("build_s", "{:.1f}"), ("recall", "{:.3f}"),
                    ("qps_warm", "{:.0f}"), ("p50_warm", "{:.1f}")]:
    print(f"\n===== {metric} =====")
    print(f"{'cell':16}" + "".join(f"{LBL[t]:>14}" for t in TAGS))
    for k in cells:
        row = f"{cellname(k):16}"
        for t in TAGS:
            v = data.get(t, {}).get(k, {}).get(metric)
            row += f"{(fmt.format(v) if v is not None else '-'):>14}"
        print(row)


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


# Per-algo geomean over the cells each algo actually ran (skips '-').
print("\n===== per-algo geomean (over that algo's own cells) =====")
print(f"{'metric':12}" + "".join(f"{LBL[t]:>14}" for t in TAGS))
for metric in ["build_s", "recall", "qps_warm", "p50_warm"]:
    row = f"{metric:12}"
    for t in TAGS:
        g = geomean([r.get(metric) for r in data.get(t, {}).values()])
        row += f"{(f'{g:.2f}' if g is not None else '-'):>14}"
    print(row)

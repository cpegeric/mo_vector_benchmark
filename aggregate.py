#!/usr/bin/env python3
"""aggregate.py — 4-way comparison of the wiki_all 1M ivfflat matrix."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = ["gpu_simd", "gpu_nosimd", "nogpu_simd", "nogpu_nosimd"]
LBL = {"gpu_simd": "GPU+SIMD", "gpu_nosimd": "GPU-noSIMD",
       "nogpu_simd": "noGPU+SIMD", "nogpu_nosimd": "noGPU-noSIMD"}

data = {}
for t in TAGS:
    p = os.path.join(HERE, f"bench_matrix_{t}.json")
    if os.path.exists(p):
        data[t] = {(r["kind"], r["base"], r["quant"]): r for r in json.load(open(p))}

cells = [("base","f32","float32"),("base","f16","float32"),("base","bf16","float32"),
         ("base","int8","float32"),("base","uint8","float32"),
         ("quant","f32","float16"),("quant","f32","bf16"),
         ("quant","f32","int8"),("quant","f32","uint8")]

def cellname(k):
    kind, base, quant = k
    return f"base={base}" if kind == "base" else f"quant={quant}"

for metric, fmt in [("build_s", "{:.0f}"), ("recall", "{:.3f}"), ("lat_p50", "{:.0f}")]:
    print(f"\n===== {metric} =====")
    hdr = f"{'cell':16}" + "".join(f"{LBL[t]:>14}" for t in TAGS)
    print(hdr)
    for k in cells:
        row = f"{cellname(k):16}"
        for t in TAGS:
            r = data.get(t, {}).get(k, {})
            v = r.get(metric)
            row += f"{(fmt.format(v) if v is not None else '-'):>14}"
        print(row)

# SIMD and GPU speedup summary on build + p50 (geomean over cells)
import math
def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None

print("\n===== speedup factors (geomean across 9 cells) =====")
for metric in ["build_s", "lat_p50"]:
    def ratio(ta, tb):
        rs = []
        for k in cells:
            a = data.get(ta, {}).get(k, {}).get(metric)
            b = data.get(tb, {}).get(k, {}).get(metric)
            if a and b and a > 0:
                rs.append(b / a)
        return geomean(rs)
    print(f"  [{metric}]")
    if "gpu_simd" in data and "gpu_nosimd" in data:
        print(f"    SIMD on GPU build   (nosimd/simd): {ratio('gpu_simd','gpu_nosimd'):.2f}x slower without SIMD")
    if "nogpu_simd" in data and "nogpu_nosimd" in data:
        print(f"    SIMD on CPU build   (nosimd/simd): {ratio('nogpu_simd','nogpu_nosimd'):.2f}x slower without SIMD")
    if "gpu_simd" in data and "nogpu_simd" in data:
        print(f"    GPU vs CPU (SIMD)   (cpu/gpu):      {ratio('gpu_simd','nogpu_simd'):.2f}x")
    if "gpu_nosimd" in data and "nogpu_nosimd" in data:
        print(f"    GPU vs CPU (noSIMD) (cpu/gpu):      {ratio('gpu_nosimd','nogpu_nosimd'):.2f}x")

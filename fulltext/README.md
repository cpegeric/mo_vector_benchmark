# Fulltext / fulltext2 / bm25 benchmark suite

Generalized tools to **generate a corpus**, **generate a query workload**, and **run
benchmarks** for MatrixOne's fulltext-family retrieval indexes (`fulltext`,
`fulltext2`, `bm25`). Run everything **from the repo root** so the shared `data/` and
`cfg/` directories resolve.

## Data contract

Every corpus is three files under `data/<name>`:

| file | format | purpose |
|------|--------|---------|
| `<name>.csv` | `id,"body"` (LOAD DATA `FIELDS ',' ENCLOSED '"'`) | the documents |
| `<name>.queries.json` | `[{"terms","nterms","kind"}]` | df-sampled `common`/`rare` OR workload |
| `<name>.oracle.json` | `[{"q","ids"}]` | real sentences → the doc id they came from (ground truth) |

Any generator that emits these three files is a drop-in `--data` for every benchmark.

## 1. Generate data + queries

### From real Wikipedia (en + zh baseline) — `wiki_corpus.py`

Extracts real article plain-text from a `pages-articles(-multistream).xml.bz2` dump.
The document **body is raw plain-text** — MatrixOne tokenizes it at index time — so no
segmenter touches the corpus. Only the **query workload** is language-aware: English =
whitespace/word tokens; Chinese = jieba words (matching the `gojieba` parser's units so
sampled terms align with what the index stores).

Long articles are **chunked into `--chunk` (default 1024) token docs** — a token is a
word for en, a character for zh — so docs are uniformly sized (like the synthetic
corpus's `-t 1024`) instead of one 30KB article next to a 300-char stub. `--chunk 0`
keeps one article per doc. jieba word length for zh queries spans 2–8 chars (idioms /
named entities included).

```bash
# a single dump file is enough. point at a local file (a dir/glob of parts also works):
python fulltext/wiki_corpus.py \
    --dump /path/enwiki-latest-pages-articles-multistream1.xml-p1p41242.bz2 \
    --lang en -n 50000 -o data/wiki_en_50k

python fulltext/wiki_corpus.py \
    --dump /path/zhwiki-latest-pages-articles-multistream1.xml-p1p187712.bz2 \
    --lang zh -n 50000 -o data/wiki_zh_50k

# pass 2 files as the dataset (uncapped -n consumes them):
python fulltext/wiki_corpus.py --dump /path/…-multistream1.xml-p*.bz2 /path/…-multistream2.xml-p*.bz2 \
    --lang en -n 200000 -o data/wiki_en_2f

# or let the tool fetch dumps: --download {en,zh} scrapes the hardcoded WIKI_BASE dir for the
# CURRENT multistream parts (filenames change each dump); --download-files N picks how many.
# zh_yue / simple are small single-file editions for a quick test.
python fulltext/wiki_corpus.py --download en --download-files 2 --lang en -n 200000 -o data/wiki_en_2f
python fulltext/wiki_corpus.py --download zh_yue --lang zh -n 20000 -o data/wiki_zhyue_20k
```

Requires `pip install wikitextparser` (both langs) and `pip install jieba` (zh queries).

### Synthetic Zipfian corpus — `gen_fulltext_data.py`

Controllable size/tokens with a real Zipf df skew (jieba dict + Shakespeare vocab) and
planted 红楼梦 ground-truth sentences. Good for scaling tests where you want an exact row
count and token length.

```bash
python fulltext/gen_fulltext_data.py -n 1000000 -t 1024 -o data/ft1m
```

## 2. Run benchmarks

All take `--config ftcfg/ft2.json` (`{host,port,user,password}`) and `--data data/<name>`.

| script | what it measures |
|--------|------------------|
| `fulltext_vs_fulltext2_bench.py` | classic vs fulltext2: build (time-until-searchable) + NL `MATCH` latency |
| `fulltext2_topk_bench.py` | Block-Max WAND top-k: boolean-OR + `ORDER BY score LIMIT k` at k=10/100/1000 |
| `retrieval_topk_3way.py` | 3-way ranked top-k: `fulltext` / `fulltext2` / `bm25`, each in its native query form (`MATCH … IN BOOLEAN MODE` vs `bm25(col) AGAINST(…)`); needs `experimental_bm25_index` |
| `phrase_precision.py` | **fulltext2 vs bm25 divergence** — CJK phrase precision. A Chinese query gojieba-tokenizes into several tokens; fulltext2 matches them as an EXACT positional phrase, bm25 as BAG-OF-WORDS. Measures P@1/MRR of the true source doc + candidate-set size across CJK window lengths. The gap opens at ≥4 chars (≥2 tokens); on ranked-OR the two are identical, so this is *the* test that separates them. `--data data/wiki_zh_2f` |
| `nl_singleword_bench.py` | single-word NL latency |
| `fulltext2_cdc_ingest_bench.py` | async CDC ingest throughput / catch-up |
| `fulltext2_idxcron_test.py` | idxcron MERGE/REBUILD maintenance |
| `bm25_*` | the standalone bm25 retrieval index (cold build, seg-count, idxcron) |

```bash
python fulltext/fulltext_vs_fulltext2_bench.py --config ftcfg/ft2.json --data data/wiki_en_50k
python fulltext/fulltext2_topk_bench.py        --config ftcfg/ft2.json --data data/wiki_en_50k
```

## End-to-end example

```bash
# 1. corpus + queries from one local wiki dump
python fulltext/wiki_corpus.py --dump ~/wiki/zhwiki-...-multistream1.xml-p1p187712.bz2 \
    --lang zh -n 50000 -o data/wiki_zh_50k
# 2. benchmark both engines on it
python fulltext/fulltext_vs_fulltext2_bench.py --config ftcfg/ft2.json --data data/wiki_zh_50k
python fulltext/fulltext2_topk_bench.py        --config ftcfg/ft2.json --data data/wiki_zh_50k
```

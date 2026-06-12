# Wiki-all 向量数据集测试工具

用于 cuVS Bench Wiki-all 数据集（768 维向量）导入和测试 MatrixOne 向量搜索性能的工具。

## 测试场景说明

### SQL 查询场景

工具支持三种 SQL 查询场景，适用于不同的向量搜索需求：

| 场景 | 说明 | 适用条件 |
|------|------|----------|
| **l2_only** | 全表向量相似度搜索，不带过滤条件 | 适用于对整个数据集进行相似度检索 |
| **l2_filter** | 先按 file_id 过滤，再在同 file_id 分区内进行向量搜索 | 适用于多租户场景，按 file_id 隔离数据 |
| **l2_filter_threshold** | 在 l2_filter 基础上增加距离阈值，只返回距离小于阈值的向量 | 适用于需要过滤远距离结果的精确检索 |

### 1. 安装依赖

```bash
pip install numpy pymysql boto3
```

### 2. 准备配置文件

复制并编辑 S3 密钥（若用 S3 导入或 S3 上的 ann 文件）：

```bash
cp cfg/s3_credentials.example.json cfg/s3_credentials.json
# 填写 access_key_id、secret_access_key
```

按索引类型与数据规模选择或复制 `cfg/` 下 JSON（如 `cfg/ivfflat_10M.json`），至少确认：

| 字段 | 含义 |
|------|------|
| `host` / `port` / `user` / `password` | MatrixOne 连接 |
| `database` / `table` | 库表名 |
| `index` | 索引类型与参数（`ivfflat` / `ivfpq` / `cagra` / `hnsw`） |
| `env` | 查询会话变量，如 `probe_limit`（recall 时自动 `SET`） |
| `dataset.s3` | 可选，S3 LOAD DATA 导入路径 |
| `dataset.ann_s3` | 可选，召回用 ann 文件在 S3 上的位置 |

### 3. 下载 Wiki-all 数据集

使用 [cuVS Bench Wiki-all 数据集](https://docs.rapids.ai/api/cuvs/nightly/cuvs_bench/wiki_all_dataset/) 进行测试（真实数据，**768 维**向量，`.fbin` 为 float32 二进制格式）。

```bash
# 下载 Wiki-all 数据集 1M（约 100 万条，约 2.9GB）
curl -L -O https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar

# 下载 Wiki-all 数据集 10M（约 1000 万条）
curl -L -O https://data.rapids.ai/raft/datasets/wiki_all_10M/wiki_all_10M.tar

# 解压 tar 文件（以 1M 为例）
tar -xf wiki_all_1M.tar
tar -xf wiki_all_10M.tar

# 下载 Wiki-all 数据集 88M（完整数据集，分卷下载）
mkdir -p wiki_all_88M
curl -s https://data.rapids.ai/raft/datasets/wiki_all/wiki_all.tar.{00..9} | tar -xf - -C wiki_all_88M/

# 若已分卷下载到当前目录，也可合并解压：
# cat wiki_all.tar.* | tar -xf - -C wiki_all_88M/
```

解压后典型文件（以 10M 为例）：

| 文件 | 用途 |
|------|------|
| `base.10M.fbin` | 库内向量，对应 `cfg.dataset.base_fbin` |
| `queries.fbin` | 查询向量，对应 `dataset.query_fbin` |
| `groundtruth.10M.neighbors.ibin` | cuVS 预计算邻居，对应 `dataset.groundtruth_ibin` |

在 `cfg/*.json` 中填写上述路径后，可用 `run_wiki.py setup` / `all`（fbin INSERT）或 `gen_csv` + LOAD DATA 导入。**千万级生产灌数更推荐**在 cfg 中配置 `dataset.s3`，由 MatrixOne `LOAD DATA` 从对象存储导入，无需本地解压全部 fbin。

---

## 子命令速查

| 命令 | 作用 |
|------|------|
| `all` | 清库建表 → 导入 → 建索引 → recall |
| `setup` | 同上，**不跑** recall |
| `create_table` | 仅建库建表 |
| `import` | 仅导入（S3 > `--csv` > fbin INSERT） |
| `create_index` | 仅建索引 |
| `drop_index` | 删索引（便于调参重建） |
| `gen_csv` | fbin → CSV（供 LOAD DATA） |
| `ann` | 在线生成 ann 评测文件 |
| `recall` | 仅跑召回 / QPS |

---

## 使用场景

### 场景 1：10M 全流程（S3 导入 + 建索引 + 召回）

**适用**：已有 S3 上的 wiki CSV、固定 `cfg`、CI 或 nightly 回归。

```bash
python run_wiki.py all --config cfg/ivfflat_10M.json \
  -n 5000 -k 100 --concurrency 32
```

`all` 会：`DROP DATABASE` → 建表 → `LOAD DATA`（读 `dataset.s3`）→ 按 `index`/`env` 建索引 → `recall`（默认用 cfg 中的 GT 设置）。

若 recall 走 S3 上的 ann，cfg 需配置 `dataset.gt_source: "ann"` 与 `dataset.ann_s3`（见场景 6）。

---

### 场景 2：只准备环境，稍后再测

**适用**：先灌数、建索引，召回参数未定或需多台机器共享同一库。

```bash
python run_wiki.py setup --config cfg/ivfflat_10M.json
```

---

### 场景 3：调索引参数后只重跑召回

**适用**：改 `cfg` 里 `index`（如 `lists`、`probe_limit`）或对比不同索引类型。

```bash
python run_wiki.py drop_index   --config cfg/ivfflat_10M.json
python run_wiki.py create_index --config cfg/ivfflat_10M.json
python run_wiki.py recall --config cfg/ivfflat_10M.json \
  --gt-source ann --sql-mode l2_only -n 10000 -k 10 --concurrency 100
```

数据仍在库里，无需重新 `setup`。

---

### 场景 4：本地 CSV + LOAD DATA 导入

**适用**：无 S3、但希望比 fbin INSERT 更快；CSV 须放在 **MatrixOne 服务端可读路径**。

```bash
# 1) 从 fbin 生成 CSV（不连库）
python run_wiki.py gen_csv --config cfg/ivfpq_1M.json --output /tmp/wiki_1M.csv

# 2) 全流程
python run_wiki.py all --config cfg/ivfpq_1M.json --csv /tmp/wiki_1M.csv \
  -n 5000 -k 100 --concurrency 32
```

多分片 fbin 可用 `--output-csv-prefix /tmp/wiki_`，再用 `--input-csv-prefix /tmp/wiki_` 导入。

---

### 场景 5：小数据 fbin INSERT 导入

**适用**：1M 调试、无 S3/CSV；在 `cfg` 中配置 `dataset.base_fbin`。

```bash
python run_wiki.py setup --config cfg/ivfpq_1M.json
# 或带 recall
python run_wiki.py all --config cfg/ivfpq_1M.json -n 1000 -k 10 --concurrency 8
```

导入优先级：**S3 > `--csv` > fbin INSERT**。

---

### 场景 6：生成 ann 评测文件并上传 S3

**适用**：召回要用固定 query/GT，避免每次在线算精确 GT；支持 S1/S2/S3 三种 `--sql-mode`。

精确 GT 由 `ann` 通过 `mode=force` SQL 导出，产出：

- `query_<mode>_k<k>.fvecs`
- `groundtruth_<mode>_k<k>.ivecs`
- `id_mapping_<mode>_k<k>.txt`
- S2/S3 且 `--distribute-file-ids` 时另有 `query_<mode>_k<k>.filters.txt`（每行一个 `file_id`）

```bash
# S1 全表
python run_wiki.py ann --config cfg/ivfflat_10M.json \
  --sql-mode l2_only -n 1000 -k 10

# S2 多租户：queries 均分到表内多个 DISTINCT file_id（最多 50 个，可改 --max-distinct-file-ids）
python run_wiki.py ann --config cfg/ivfflat_10M.json \
  --sql-mode l2_filter --distribute-file-ids -n 10000 -k 10
```

将生成文件上传到 `cfg.dataset.ann_s3.prefix`，并在 cfg 里按模式登记文件名，例如：

```json
"ann_s3": {
  "prefix": "vector/wiki_ann/ivfflat_10m",
  "local_dir": "/tmp/wiki_ann_ivfflat_10m",
  "l2_only": { "query_fvecs": "query_l2_only_k100.fvecs", ... },
  "l2_filter": {
    "query_fvecs": "query_l2_filter_k100.fvecs",
    "groundtruth_ivecs": "groundtruth_l2_filter_k100.ivecs",
    "id_mapping": "id_mapping_l2_filter_k100.txt",
    "query_filters": "query_l2_filter_k100.filters.txt"
  }
}
```

`-n` 越大生成越慢，建议先用小 `-n` 验证。

---

### 场景 7：用 S3 上的 ann 做 recall（推荐常态）

**适用**：环境已由 `setup`/`all` 建好；GT 已放在 S3。

```bash
python run_wiki.py recall --config cfg/ivfflat_10M.json \
  --gt-source ann \
  --sql-mode l2_only \
  -n 10000 -k 10 --concurrency 100
```

- 按 `--sql-mode` 自动下载 `ann_s3.<mode>` 下文件到 `local_dir`
- `probe_limit` 等来自 `cfg.env`，无需手写 `--probe`
- 更新 S3 文件后加 `--ann-s3-refresh`

---

### 场景 8：用 cuVS 预计算 GT（fbin / ibin）

**适用**：已有 cuVS 的 `queries.fbin` 与 `groundtruth.*.ibin`。

**本地路径**：在 cfg 中配置 `dataset.query_fbin`、`dataset.groundtruth_ibin`、`dataset.id_offset`（本地文件存在时优先使用）。

**S3（CI / 无本地文件时）**：可同时配置 `dataset.fbin_s3`（复用 `dataset.s3` 的 bucket/endpoint 与 `cfg/s3_credentials.json`）；本地路径不存在时自动从 S3 下载到 `local_dir`：

```json
"fbin_s3": {
  "prefix": "vector/wiki_all_10m",
  "local_dir": "/tmp/wiki_fbin_10m",
  "query_fbin": "queries.fbin",
  "groundtruth_ibin": "groundtruth.10M.neighbors.ibin"
}
```

```bash
python run_wiki.py recall --config cfg/ivfflat_10M.json \
  --gt-source fbin \
  --sql-mode l2_only \
  -n 5000 -k 100 --concurrency 32
```

更新 S3 文件后加 `--fbin-s3-refresh`（或 `--ann-s3-refresh` 也会刷新 fbin 缓存）。

S2/S3 若用 ibin 本地过滤 GT，需配合 `--filter-val` 与 `--filter-file-id-base` / `--filter-distinct-file-ids`（与灌数时 `file_id` 规则一致）。

---

### 场景 9：多租户 recall（l2_filter / l2_filter_threshold）

**适用**：按 `file_id` 分区检索；与生产多租户一致。

**方式 A（推荐）**：ann 含 `query_filters`，cfg 已配 `ann_s3.l2_filter`：

```bash
python run_wiki.py recall --config cfg/ivfflat_10M.json \
  --gt-source ann \
  --sql-mode l2_filter \
  -n 10000 -k 10 --concurrency 100
```

无需 `--filter-val`；每条 query 的 `file_id` 来自 `.filters.txt`。

**方式 B**：只测某一个 `file_id`：

```bash
python run_wiki.py recall --config cfg/ivfpq_1M.json \
  --sql-mode l2_filter --filter-val 20000007 \
  -n 1000 -k 10 --concurrency 8
```

**方式 C**：无预生成 ann，在线均分多个 `file_id`（较慢）：

```bash
python run_wiki.py recall --config cfg/ivfflat_10M.json \
  --sql-mode l2_filter --distribute-file-ids \
  -n 1000 -k 10 --concurrency 100
```

**方式 D**：距离阈值场景：

```bash
python run_wiki.py recall --config cfg/ivfflat_10M.json \
  --gt-source ann --sql-mode l2_filter_threshold \
  -n 10000 -k 10 --concurrency 100
```

阈值上界在 `sql_config_simple.json` 的 `m3_l2_filter_threshold.extra.max_distance` 中配置。

---

### 场景 10：对比索引 Filter 执行方式

**适用**：评估 `pre` / `post` / `force` / `include` 对 QPS 与 recall 的影响。

```bash
python run_wiki.py recall --config cfg/ivfflat_10M.json \
  --gt-source ann --sql-mode l2_filter \
  --filter-mode pre \
  -n 5000 -k 10 --concurrency 32
```

`--filter-mode` 对应 SQL 后缀 `BY RANK WITH OPTION 'mode=...'`。`force` 可作精确检索 baseline。

---

## 评测概念（简表）

### SQL 模式 `--sql-mode`

| 值 | 含义 |
|----|------|
| `l2_only` | 全表 L2 Top-K |
| `l2_filter` | `WHERE file_id = ?` 后 Top-K |
| `l2_filter_threshold` | 同上 + L2 距离上限 |

### GT 来源 `--gt-source`（仅 `recall` / `all`）

| 值 | 含义 |
|----|------|
| `auto` | 有 fbin/ibin 用 fbin，否则 ann |
| `fbin` | cuVS 文件 |
| `ann` | fvecs + ivecs + id_mapping（+ 可选 filters） |

### 输出指标

| 指标 | 说明 |
|------|------|
| Recall@k | 检索结果与 GT 的命中率 |
| Eligible Recall | S2/S3 时分母为 `min(k, \|filtered_gt\|)` |
| QPS / P50 / P95 / P99 | 吞吐与延迟 |

---

## 常用参数

| 参数 | 说明 |
|------|------|
| `--config` | **必填**，如 `cfg/ivfflat_10M.json` |
| `-n` | 查询条数（默认 1000） |
| `-k` | Top-K（默认 10） |
| `--concurrency` | 并发（默认 4） |
| `--sql-mode` | `l2_only` / `l2_filter` / `l2_filter_threshold` |
| `--gt-source` | `auto` / `fbin` / `ann` |
| `--filter-val` | S2/S3 单一 `file_id`；有 `query_filters` 时可省略 |
| `--filter-mode` | `pre` / `post` / `force` / `include` |
| `--distribute-file-ids` | 多 `file_id` 均分（`ann` 会写 `.filters.txt`） |
| `--max-distinct-file-ids` | 默认 50；`0` 表示不限制 |
| `--ann-s3-refresh` | 强制重新下载 S3 ann |
| `--csv` / `--input-csv-prefix` | LOAD DATA 路径 |
| `--s3-endpoint` 等 | 覆盖 cfg 中 `dataset.s3` |

---

## 配置文件示例

完整示例见 `cfg/ivfflat_10M.json`。最小结构：

```json
{
  "host": "127.0.0.1",
  "port": 6001,
  "user": "dump",
  "password": "111",
  "database": "jst_app_wiki",
  "table": "historical_file_blocks_wiki_ivfflat10m",
  "index": {
    "name": "idx_l2",
    "type": "ivfflat",
    "lists": 3162,
    "op_type": "vector_l2_ops"
  },
  "env": {
    "probe_limit": 20
  },
  "dataset": {
    "s3": {
      "endpoint": "http://...",
      "bucket": "...",
      "filepath": "vector/wiki_all_10m/....csv",
      "region": "ap-guangzhou",
      "compression": "none"
    },
    "gt_source": "ann",
    "ann_s3": {
      "prefix": "vector/wiki_ann/ivfflat_10m",
      "local_dir": "/tmp/wiki_ann_ivfflat_10m",
      "l2_only": {
        "query_fvecs": "query_l2_only_k100.fvecs",
        "groundtruth_ivecs": "groundtruth_l2_only_k100.ivecs",
        "id_mapping": "id_mapping_l2_only_k100.txt"
      }
    }
  }
}
```

### 其它说明

- **`sql_config_simple.json`**：评测 SQL 模板与 S3 阈值；与 `eval_vector_search_from_table.py` 同目录。
- **灌数 `file_id` 规则**：`file_id = file_id_base + (row_idx - 1) % distinct_file_ids`（默认 `20000000` / `50`）。
- **表结构**：`id`、`file_id`、`embedding` VECF32(768) 等，见 `run_wiki.py` 建表 SQL。
- **S3 密钥顺序**：CLI `--s3-access-key-*` > cfg 内联 > `cfg/s3_credentials.json` > 环境变量。

---

## 推荐工作流（10M + IVFFLAT + ann）

```text
1. 编辑 cfg/ivfflat_10M.json（连库、S3、ann_s3、index、env）
2. cp cfg/s3_credentials.example.json → cfg/s3_credentials.json
3. python run_wiki.py setup --config cfg/ivfflat_10M.json          # 首次灌数+建索引
4. python run_wiki.py ann  --config ... --sql-mode l2_filter --distribute-file-ids -n 10000 -k 10   # 可选：本地生成 ann 再上传 S3
5. python run_wiki.py recall --config ... --gt-source ann --sql-mode l2_filter -n 10000 -k 10 --concurrency 100
6. 调 probe/lists 时：drop_index → 改 cfg → create_index → recall
```

更多子命令说明：`python run_wiki.py -h`。

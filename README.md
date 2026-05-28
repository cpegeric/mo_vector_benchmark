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

### Filter 模式

针对三种 SQL 场景支持三种不同的filter模式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **pre** | 预过滤模式 | 索引层先过滤再计算距离，减少向量计算量，性能优先 |
| **post** | 后过滤模式（默认） | 向量计算后过滤，保证精度 |
| **force** | 强制精确搜索 | 不使用索引，全表精确计算，作为召回率 baseline |
| **include** | INCLUDE 列过滤 | 依赖 `index.include` 的冗余列直接在索引层内过滤，避免回表，适合 INCLUDE 列（如 `file_id`）作为谓词列 |

### 评估指标

| 指标 | 说明 |
|------|------|
| **Recall** | 召回率，索引搜索结果与精确搜索结果的匹配度 |
| **Eligible Recall** | 过滤召回率（`l2_filter` / `l2_filter_threshold`）：分母取 `min(k, |filtered_gt|)`；本地按 `file_id` 谓词筛 `.ibin` GT 得到 `filtered_gt`，避免过滤后 GT 少于 k 时分母过大 |
| **QPS** | 每秒查询次数，反映系统吞吐量 |
| **Latency** | 查询延迟（P50/P99），反映响应速度 |

### 配置文件 `sql_config_simple.json`

与 `eval_vector_search_from_table.py` 同目录，**标准 JSON（不支持 `//` 或 `#` 注释）**。每次执行 `ann` / `run` / `wiki test` 等会由评测脚本读取，用于生成 SQL 模板、S3 距离阈值与预检行数。

| 区块 | 字段 | 说明 |
|------|------|------|
| `sql_modes` | `m1_l2_only` / `m2_l2_filter` / `m3_l2_filter_threshold` | 对应 `l2_only`、`l2_filter`、`l2_filter_threshold` 三类查询；`sql` 中含占位符 `{table}`、`{emb_col}`、`{filter_col}`、`{max_distance}`（仅 m3）等，由程序替换为实际库表与列名。 |
| `sql_modes.m3_l2_filter_threshold.extra` | `max_distance` | **S3（l2_filter_threshold）** 的 L2 距离上界，写入 SQL 与预检逻辑。可按数据规模调整，推荐100 万行量级可试 **2.5**，约 1000 万行量级可试 **2.9**（需自行按召回与数据分布调参）。 |
| `default` | `table` | 文档/默认表名参考；实际表名以命令行全局参数 `--table` 为准。 |
| `default` | `emb_col` / `filter_col` | 向量列、过滤列名，与表结构一致即可。 |
| `default` | `min_verify_partition_rows` | 跑评测前校验：每个 `file_id` 分区内行数、以及 S3 阈值球内行数，至少达到该值才认为通过预检（默认 **2000**）。 |

修改 `sql_config_simple.json` 后无需改代码；若评测进程未重启，下一次调用 `evaluate` 时会重新加载该文件。

## 快速开始

### 1. 安装依赖

```bash
pip install numpy pymysql
```

### 2. 下载 Wiki-all 数据集

使用 [cuVS Bench Wiki-all 数据集](https://docs.rapids.ai/api/cuvs/nightly/cuvs_bench/wiki_all_dataset/) 进行测试（真实数据，768 维向量）。

```bash
# 下载 Wiki-all 数据集 1M（约 100 万条，2.9GB）
curl -L -O https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar

# 下载 Wiki-all 数据集 10M（约 1000 万条）
curl -L -O https://data.rapids.ai/raft/datasets/wiki_all_10M/wiki_all_10M.tar

# 解压 tar 文件（以 1M 为例）
tar -xf wiki_all_1M.tar

# 下载 Wiki-all 数据集 88M（完整数据集，分卷下载）
curl -s https://data.rapids.ai/raft/datasets/wiki_all/wiki_all.tar.{00..9} | tar -xf - -C wiki_all_88M/

# 解压 tar 文件
cat wiki_all.tar.* | tar -xf - -C wiki_all_88M/
```

### 3. 初始化测试环境数据 wiki_all

使用 `wiki setup` 命令一键完成：创建表 → 导入数据 → 创建索引。

```bash
# 基本用法（只需指定 --fbin 以 1M 为例）
python run_vector_test.py wiki setup --fbin wiki_all_1M/base.fbin
```

**`wiki setup` 参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fbin` | - | **必需**。`.fbin` 向量文件路径，指定后自动执行创建表、导入、建索引三步 |
| `--ivf-lists` | 1000 | IVF 聚类中心数。数据量越大，建议值越大（如 100 万数据建议 1000） |
| `--batch-size` | 20000 | 批量导入大小。导入时的批处理行数 |
| `--file-id-base` | 20000000 | file_id 起始值。导入的数据会使用 50 个不同的 file_id 循环分布 |

使用例子：

```bash
# 指定 IVF lists 数量（默认 1000）
python run_vector_test.py wiki setup --fbin wiki_all_1M/base.fbin --ivf-lists 2000

# 指定数据库连接信息和表名
python run_vector_test.py --host 192.168.1.100 --database mydb wiki setup \
  --fbin wiki_all_1M/base.fbin \
  --ivf-lists 2000
```

**注意**：`--fbin` 参数只需指定一次，工具会自动完成创建表、导入数据、创建索引三个步骤。如果只需要执行其中某一步，可使用 `--create-table` 或 `--create-index` 参数。
**注意**：wiki setup 导入 wiki_all 采用批量 INSERT，耗时较长。推荐用 `run_wiki.py all` + S3 `LOAD DATA`（见下节），百万级数据可在秒级完成导入。

#### 3.1 子命令入口 `run_wiki.py`（推荐）

`run_wiki.py` 是基于 JSON 配置 (`cfg/*.json`) 的子命令入口，从 `dataset` 块读取 `base_fbin` / `query_fbin` / `groundtruth_ibin` 路径，无需每次重复传 `--fbin` 等参数。

```
python run_wiki.py <command> --config cfg/xxx.json [options]
```

| 命令 | 说明 |
|------|------|
| `all` | 顺序执行：清理旧库/建表 → 导入 → 建索引 → recall（导入支持 S3 / CSV / fbin） |
| `setup` | 仅前三步：清理旧库/建表 → 导入 → 建索引（**不跑 recall**） |
| `create_table` | 仅创建表 |
| `import` | 仅导入数据；默认走 `.fbin` INSERT，加 `--csv PATH` 或 `--input-csv-prefix PREFIX` 改走 LOAD DATA INFILE（显著更快） |
| `create_index` | 仅创建向量索引（读取 `cfg.index` 与 `cfg.env`） |
| `drop_index` | 删除索引（索引名取自 `cfg.index.name`，同时尝试清理旧名 `idx_embedding`） |
| `gen_csv` | 将 `dataset.base_fbin` 转为 6 列 CSV（LOAD DATA 兼容），不连库 |
| `ann` | 在线生成 ann 评测文件（`query/groundtruth/id_mapping`，S2/S3 还可生成 `.filters.txt`）；内部调用 `eval_vector_search_from_table.py` |
| `recall` | 仅跑召回评估（子进程调用 `eval_vector_search_from_table.py`；自动透传 `cfg` 连库参数、`cfg.env`、ann 路径） |

`dataset.base_fbin` 可为单个字符串或字符串数组，用于多 shard 数据集：

```json
"dataset": {
  "base_fbin": [
    "/data/wiki_88M/base.00.fbin",
    "/data/wiki_88M/base.01.fbin"
  ],
  "query_fbin": "/data/wiki_88M/queries.fbin",
  "groundtruth_ibin": "/data/wiki_88M/groundtruth.neighbors.ibin",
  "id_offset": 1
}
```

多 .fbin 时，全局 1-based 行号 `i` 跨文件连续递增（shard0: 1..N0，shard1: N0+1..N0+N1 ...），保持 `file_id` / `page_num` / `content` / `meta` 分布不变。

**典型用法**

```bash
# 全流程（INSERT 导入）
python run_wiki.py all --config cfg/ivfpq_1M.json -n 5000 -k 100 --concurrency 32

# 仅建表 + S3 导入 + 建索引（不跑 recall）
python run_wiki.py setup --config cfg/ivfflat_10M.json

# 全流程（S3 LOAD DATA，一步完成：清库建表 → S3 导入 → IVF 索引 → recall）
# run_vector_test.py 与 run_wiki.py 等价（均需 --config + cfg/s3_credentials.json）
python run_vector_test.py --config cfg/ivfflat_10M.json all -n 5000 -k 100 --concurrency 32
python run_wiki.py all --config cfg/ivfflat_10M.json -n 5000 -k 100 --concurrency 32
# 方式 B：CLI 传 S3 参数（密钥见统一凭证文件，见下）
python run_wiki.py all --config cfg/ivfflat_1M.json \
  --s3-endpoint oss-cn-shanghai.aliyuncs.com \
  --s3-bucket my-bucket --s3-filepath wiki/wiki_1M.csv \
  --s3-region oss-cn-shanghai -n 5000 -k 100 --concurrency 32

# 先生成单个 CSV，再用 LOAD DATA 走全流程（百万级数据导入从分钟级降到秒级）
python run_wiki.py gen_csv --config cfg/ivfpq_1M.json --output /tmp/wiki_1M.csv
python run_wiki.py all --config cfg/ivfpq_1M.json --csv /tmp/wiki_1M.csv \
    -n 5000 -k 100 --concurrency 32

# 多 CSV 分片：每个 .fbin 对应一个 CSV，再按前缀 LOAD
python run_wiki.py gen_csv --config cfg/my_sharded.json --output-csv-prefix /tmp/wiki_
#   生成：/tmp/wiki_0.csv, /tmp/wiki_1.csv, /tmp/wiki_2.csv ...
python run_wiki.py import --config cfg/my_sharded.json --input-csv-prefix /tmp/wiki_
#   按顺序对匹配到的每个 CSV 执行一次 LOAD DATA INFILE

# 迭代索引调参：只重建索引 + 重跑召回
python run_wiki.py drop_index   --config cfg/ivfpq_1M.json
python run_wiki.py create_index --config cfg/ivfpq_1M.json
python run_wiki.py recall       --config cfg/ivfpq_1M.json -n 5000 -k 100 --concurrency 32

# 带过滤的召回（eligible recall@k）
python run_wiki.py recall --config cfg/ivfpq_1M.json \
    --sql-mode l2_filter --filter-val 20000007 --filter-mode pre \
    -n 1000 -k 10 --concurrency 8
```

**过滤召回（l2_filter / l2_filter_threshold）**

S2/S3 的 SQL 带 `WHERE file_id = ?`。`file_id` 来源（三选一，按优先级）：

| 方式 | 说明 |
|------|------|
| **`--filter-val=<file_id>`** | 所有 query 共用同一个 `file_id` |
| **ann + `query_filters`（`.filters.txt`）** | 每条 query 一行 `file_id`；`cfg.ann_s3.l2_filter` 等已配置时 **无需** `--filter-val`（推荐多租户 recall） |
| **`--distribute-file-ids`** | 从表取至多 N 个 `DISTINCT file_id`，将 `num_queries` 均分到各分区（在线抽样或 `ann` 生成时写出 `.filters.txt`） |

导入数据的 `file_id` 分布（与 `gen.py` / `gen_csv` 一致）：

```
file_id = file_id_base + (row_idx - 1) % distinct_file_ids
```

默认 `file_id_base=20000000`、`distinct_file_ids=50`。用 cuVS `.ibin` 做本地 GT 过滤时，需用 `--filter-file-id-base` / `--filter-distinct-file-ids` 与生成时一致；ann 三件套 + `.filters.txt` 则 GT 已按分区导出，无需再筛。

**召回 GT 来源（`recall` / `run` / `all` 最后一步）**

| 方式 | cfg `dataset` 字段 | CLI（覆盖 cfg） |
|------|-------------------|-----------------|
| cuVS 文件 | `query_fbin`, `groundtruth_ibin`, `id_offset` | `--query-fbin`, `--groundtruth-ibin`, `--id-offset` |
| ann-benchmarks | `query_fvecs`, `groundtruth_ivecs`, `id_mapping` | `--query-fvecs`, `--groundtruth-ivecs`, `--id-mapping` |

两套都写在 cfg 时，用 **`--gt-source`** 区分（或 cfg 里 `"gt_source": "fbin"` / `"ann"`）：

| `--gt-source` | 行为 |
|---------------|------|
| `auto`（默认） | 有完整 fbin/ibin 则用 fbin，否则用 ann 三件套 |
| `fbin` | 只用 cuVS fbin/ibin |
| `ann` | 只用 ann fvecs/ivecs/id_mapping |

```bash
# cuVS 预计算 GT
python run_wiki.py recall --config cfg/ivfflat_10M.json --gt-source fbin -n 5000 -k 100 --concurrency 32

# ann 预生成 GT（路径也可写在 cfg.dataset.ann_s3，见下）
python run_wiki.py recall --config cfg/ivfflat_10M.json --gt-source ann \\
  --sql-mode l2_only -n 5000 -k 100 --concurrency 32

# S2 多 file_id recall（S3 上 ann 含 query_filters 时无需 --filter-val）
python run_wiki.py recall --config cfg/ivfflat_10M.json --gt-source ann \\
  --sql-mode l2_filter -n 10000 -k 10 --concurrency 100
```

**`recall` 与 `eval_vector_search_from_table.py`**

`run_wiki.py recall` / `run_vector_test.py run` 均通过 `run_eval()` 启动子进程执行 `eval_vector_search_from_table.py` 的 `evaluate()`，召回公式与 QPS 统计逻辑相同。`cfg/*.json` 中的 `host` / `port` / `user` / `password` / `database` / `table` 会传给 eval；日志中的表名与 `cfg.table` 一致。

**ann 文件在 S3 上（recall 前自动下载）**

在 `dataset.ann_s3` 中配置对象前缀与文件名（复用 `dataset.s3` 的 endpoint/bucket/region 与 `cfg/s3_credentials.json` 密钥）：

```json
"gt_source": "ann",
"ann_s3": {
  "prefix": "vector/wiki_ann/ivfflat_10m",
  "local_dir": "/tmp/wiki_ann_ivfflat_10m",
  "l2_only": {
    "query_fvecs": "query_l2_only_k100.fvecs",
    "groundtruth_ivecs": "groundtruth_l2_only_k100.ivecs",
    "id_mapping": "id_mapping_l2_only_k100.txt"
  },
  "l2_filter": {
    "query_fvecs": "query_l2_filter_k100.fvecs",
    "groundtruth_ivecs": "groundtruth_l2_filter_k100.ivecs",
    "id_mapping": "id_mapping_l2_filter_k100.txt",
    "query_filters": "query_l2_filter_k100.filters.txt"
  },
  "l2_filter_threshold": {
    "query_fvecs": "query_l2_filter_threshold_k100.fvecs",
    "groundtruth_ivecs": "groundtruth_l2_filter_threshold_k100.ivecs",
    "id_mapping": "id_mapping_l2_filter_threshold_k100.txt",
    "query_filters": "query_l2_filter_threshold_k100.filters.txt"
  }
}
```

`recall` 会按 `--sql-mode` 自动选用 `ann_s3.<mode>` 中的 S3 对象（文件名与 `ann` 导出一致：`query_{mode}_k{k}.*`）。评测时 `id_mapping` 每行为 `file_id` 与表主键 `id`（制表符分隔），与 SQL 返回一致。

**生成多 file_id 的 ann（再上传 S3 或本地 recall）**

```bash
pip install boto3

# 1) 在线生成：均分到表内 DISTINCT file_id，并写出 query_<mode>_k<k>.filters.txt
python run_wiki.py ann --config cfg/ivfflat_10M.json \\
  --sql-mode l2_filter --distribute-file-ids -n 10000 -k 10
# 产出示例：query_l2_filter_k10.fvecs、groundtruth_l2_filter_k10.ivecs、
#           id_mapping_l2_filter_k10.txt、query_l2_filter_k10.filters.txt

# 2) 用 cfg ann_s3 下载后 recall（多 file_id 无需 --filter-val）
python run_wiki.py recall --config cfg/ivfflat_10M.json --gt-source ann \\
  --sql-mode l2_filter -n 10000 -k 10 --concurrency 100

# 3) 不预生成 ann，在线多 file_id recall（较慢）
python run_wiki.py recall --config cfg/ivfflat_10M.json \\
  --sql-mode l2_filter --distribute-file-ids -n 10000 -k 10 --concurrency 100

# 强制重新拉取 S3：加 --ann-s3-refresh
```

等价命令：`python run_vector_test.py ann|run ...`（需 `--config` 时行为与上相同）。

召回公式变为 **eligible recall@k**：每条 query 的分母取 `min(k, |filtered_gt|)`，避免当 `.ibin` 深度不足导致过滤后邻居少于 k 时召回率无法达到 1.0。`.ibin` 的 `k_file` 越大，过滤后剩余邻居越多；若出现"可用 GT 不足 k"的警告，请使用更深的 groundtruth 文件（期望 `k_file >= k * distinct_file_ids`）。

**常用参数**

| 参数 | 适用命令 | 说明 |
|------|---------|------|
| `--config` | 全部 | JSON 配置文件路径（必填） |
| `dataset.s3` | `all` / `import` | JSON 内 S3 块：`endpoint`/`bucket`/`filepath`/`region`/`compression`（不含密钥时从凭证文件读） |
| `cfg/s3_credentials.json` | `all` / `import` | **统一 S3 密钥**（复制 `cfg/s3_credentials.example.json` 后填写，已加入 `.gitignore`） |
| `--s3-credentials-file` | `all` / `import` | 自定义密钥文件路径（默认 `cfg/s3_credentials.json`） |
| `--s3-endpoint` 等 | `all` / `import` | CLI 覆盖 JSON 的 S3 连接参数；`--s3-access-key-id` 可临时覆盖凭证文件 |
| `--csv PATH` | `all` / `import` | 走 LOAD DATA 路径的单个 CSV 文件 |
| `--input-csv-prefix PREFIX` | `all` / `import` | 匹配 `{PREFIX}*.csv`，按顺序逐个 LOAD DATA |
| `-o, --output PATH` | `gen_csv` | 输出单个 CSV 路径 |
| `--output-csv-prefix PREFIX` | `gen_csv` | 按前缀输出多个 CSV（每个 .fbin 对应 `{PREFIX}0.csv` ...） |
| `-n` / `-k` / `--concurrency` / `--sql-mode` | `all` / `recall` / `ann` | 召回/生成 ann 参数 |
| `--gt-source` | `recall` / `all` | `auto` / `fbin` / `ann`，指定 GT 来源 |
| `--ann-s3-refresh` | `recall` | 强制从 S3 重新下载 `ann_s3` 文件 |
| `--filter-val` | `all` / `recall` / `ann` | 单一 `file_id`；S2/S3 若已有 `query_filters` 或 `--distribute-file-ids` 可省略 |
| `--distribute-file-ids` | `recall` / `ann` | 多 `file_id`：均分 queries；`ann` 会写出 `.filters.txt` |
| `--max-distinct-file-ids` | `recall` / `ann` | 配合 `--distribute-file-ids`；默认 50，`0`=不限制 |
| `--filter-mode` | `all` / `recall` | `pre` / `post` / `force` / `include`，SQL 执行层过滤方式（对应 `BY RANK WITH OPTION 'mode=...'`） |
| `--filter-file-id-base` | `all` / `recall` | 本地 `.ibin` GT 过滤用 file_id_base（默认 20000000） |
| `--filter-distinct-file-ids` | `all` / `recall` | 本地 `.ibin` GT 过滤用 distinct_file_ids（默认 50） |
| `host` / `port` / `user` / `password` / `database` / `table` | `cfg/*.json` | 连库与评测表名；`recall` 会传给 `eval_vector_search_from_table.py` |
| `--expected-dim` | `gen_csv` | 期望向量维度（默认 768） |
| `--batch-size` | `all` / `import` | INSERT 批量大小（默认 20000） |
| `--file-id-base` / `--distinct-file-ids` | `all` / `import` / `gen_csv` | file_id 生成规则 |

> `run_wiki.py` 会自动从 `cfg.env.probe_limit` 设置 IVF 查询的 probe 参数，无需手动传 `--probe`，与 `vector_benchmark/gtrecall.py` 的默认行为一致。
>
> `LOAD DATA INFILE`（非 `LOCAL`）由 MatrixOne 服务端读取 CSV，CSV 文件必须放在服务端可访问的路径。若 MO 与脚本不在同一台机器，需先把 CSV 拷贝到 MO 所在机器。

**S3 凭证（统一文件）**

```bash
cp cfg/s3_credentials.example.json cfg/s3_credentials.json
# 编辑 cfg/s3_credentials.json，填入 access_key_id / secret_access_key
```

密钥读取顺序：CLI `--s3-access-key-*` > `dataset.s3` 内联 > `cfg/s3_credentials.json` > 环境变量 `MO_S3_ACCESS_KEY_ID` / `MO_S3_SECRET_ACCESS_KEY`（兼容旧用法）。

表结构：

```sql
CREATE TABLE `historical_file_blocks_wiki` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `file_id` bigint NOT NULL,
  `content` text DEFAULT NULL,
  `embedding` vecf32(768) DEFAULT NULL,
  `page_num` int NOT NULL DEFAULT 0,
  `meta` json DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_file` (`file_id`),
  FULLTEXT `idx_content`(`content`) WITH PARSER ngram
);
```
### 全局参数

`run_vector_test.py` 与各子命令的全局连库参数；**使用 `run_wiki.py` + `--config` 时以 JSON 为准**（会写入 eval 子进程）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 127.0.0.1 | 数据库主机（`cfg/*.json` 同字段） |
| `--port` | 6001 | 端口 |
| `--user` | dump | 用户名 |
| `--password` | 111 | 密码 |
| `--database` | jst_app_wiki | 数据库名 |
| `--table` | historical_file_blocks_wiki | 表名（eval SQL 与日志均使用此名） |

### 4. 生成 ANN 评测文件

生成 `query.fvecs`、`groundtruth.ivecs`、`id_mapping.txt`；S2/S3 加 `--distribute-file-ids` 时额外生成 `query_<mode>_k<k>.filters.txt`（每行一个 query 的 `file_id`），供后续多分区 recall。

```bash
# 推荐：与 cfg 一致（连库、表名、索引 env 均来自 JSON）
python run_wiki.py ann --config cfg/ivfflat_10M.json --sql-mode l2_only -n 1000 -k 10

python run_wiki.py ann --config cfg/ivfflat_10M.json \\
  --sql-mode l2_filter --distribute-file-ids -n 10000 -k 10

python run_wiki.py ann --config cfg/ivfflat_10M.json \\
  --sql-mode l2_filter_threshold --distribute-file-ids -n 10000 -k 10

# 等价：run_vector_test.py（需自行传连库参数或 --config）
python run_vector_test.py ann --config cfg/ivfflat_10M.json \\
  --sql-mode l2_filter --distribute-file-ids -n 1000 -k 10
```

**参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sql-mode` | l2_only | SQL 模式：`l2_only`、`l2_filter`、`l2_filter_threshold` |
| `-n, --num-queries` | 1000 | 查询数量 |
| `-k` | 10 | Top-K |
| `--concurrency` | 1 | 并发数 |
| `--filter-val` | - | 单一 `file_id`；多分区生成请用 `--distribute-file-ids` |
| `--distribute-file-ids` | - | 将 query 均分到多个 `DISTINCT file_id`，并写出 `.filters.txt` |
| `--max-distinct-file-ids` | 50 | 最多使用多少个不同的 `file_id`（`0`=不限制） |

**注意**：`ann` 用精确 GT SQL（`BY RANK WITH OPTION 'mode=force'`）导出，`-n` 越大越慢，可先用小 `-n` 验证。生成后可将文件上传到 `cfg.dataset.ann_s3.prefix` 对应路径。


### 5. 运行召回率/QPS 评估

运行向量搜索性能评估，输出召回率和 QPS。支持三种 SQL 场景和多种 Filter 模式。

```bash
# 推荐：run_wiki + cfg（连库、表、probe_limit、ann_s3 均来自 JSON）
python run_wiki.py recall --config cfg/ivfflat_10M.json --gt-source ann \\
  --sql-mode l2_only -n 10000 -k 10 --concurrency 100

# 全表搜索（run_vector_test，需 --config 或手动传连库参数）
python run_vector_test.py run --config cfg/ivfflat_10M.json \\
  --sql-mode l2_only -n 1000 -k 10 --concurrency 100 --skip-db-verify

# 预过滤：单一 file_id
python run_vector_test.py run --config cfg/ivfflat_10M.json \\
  --sql-mode l2_filter --filter-val 20000000 \\
  -n 1000 -k 10 --concurrency 100 --skip-db-verify

# 预过滤：多 file_id（无预生成 ann 时）
python run_wiki.py recall --config cfg/ivfflat_10M.json \\
  --sql-mode l2_filter --distribute-file-ids -n 1000 -k 10 --concurrency 100

# 调整 IVF 索引 probe 参数测试召回率
python run_vector_test.py run \
  --sql-mode l2_filter \
  -n 1000 -k 10 --concurrency 100 \
  --probe 20

# pre 模式测试（预过滤，性能优先）
python run_vector_test.py run \
  --sql-mode l2_filter \
  -n 1000 -k 10 --concurrency 100 \
  --filter-mode pre

# force 模式测试（精确搜索，作为 baseline）
python run_vector_test.py run \
  --sql-mode l2_filter \
  -n 1000 -k 10 --concurrency 100 \
  --filter-mode force
```

**参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sql-mode` | l2_only | SQL 场景：`l2_only`（全表搜索）、`l2_filter`（预过滤）、`l2_filter_threshold`（带距离阈值） |
| `--filter-mode` | - | Filter 模式：`pre`（预过滤）、`post`（后过滤）、`force`（强制精确搜索） |
| `-n, --num-queries` | 1000 | 查询数量 |
| `-k` | 10 | Top-K |
| `--concurrency` | 1 | 并发数 |
| `--gt-source` | auto | `fbin` / `ann` / `auto` |
| `--filter-val` | - | 单一 `file_id`；ann 含 `.filters.txt` 时可省略 |
| `--probe` | cfg.env | `run_wiki` 默认用 `cfg.env.probe_limit`；`run_vector_test` 可传 `--probe` |
| `--distribute-file-ids` | - | 在线多 `file_id` recall（有预生成 ann 时一般不需要） |
| `--max-distinct-file-ids` | 50 | 配合 `--distribute-file-ids` |
| `--skip-db-verify` | - | 跳过数据库预检（`run_wiki recall` 默认已跳过） |

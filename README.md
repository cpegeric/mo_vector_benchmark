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

### 评估指标

| 指标 | 说明 |
|------|------|
| **Recall** | 召回率，索引搜索结果与精确搜索结果的匹配度 |
| **QPS** | 每秒查询次数，反映系统吞吐量 |
| **Latency** | 查询延迟（P50/P99），反映响应速度 |

### 配置文件 `sql_config_simple.json`

与 `eval_vector_search_from_table.py` 同目录，**标准 JSON（不支持 `//` 或 `#` 注释）**。每次执行 `ann` / `run` / `wiki test` 等会由评测脚本读取，用于生成 SQL 模板、S3 距离阈值与预检行数。

| 区块 | 字段 | 说明 |
|------|------|------|
| `sql_modes` | `m1_l2_only` / `m2_l2_filter` / `m3_l2_filter_threshold` | 对应 `l2_only`、`l2_filter`、`l2_filter_threshold` 三类查询；`sql` 中含占位符 `{table}`、`{emb_col}`、`{filter_col}`、`{max_distance}`（仅 m3）等，由程序替换为实际库表与列名。 |
| `sql_modes.m3_l2_filter_threshold.extra` | `max_distance` | **S3（l2_filter_threshold）** 的 L2 距离上界，写入 SQL 与预检逻辑。可按数据规模调整，例如约 100 万行量级可试 **2.5**，约 1000 万行量级可试 **2.9**（需自行按召回与数据分布调参）。 |
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
**注意**：wiki setup 导入wiki_all采用批量加载方式，时间相对较长，如果想要快速加载数据，也可以手动用load data s3的方式加载更快,联系我获取csv s3 path

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

### 4. 生成 ANN 评测文件

生成 `query.fvecs` 和 `groundtruth.ivecs` 文件，用于后续评估。

```bash
# 生成全表搜索模式的 ANN 文件
python run_vector_test.py ann --sql-mode l2_only -n 1000 -k 10 --distribute-file-ids

# 生成预过滤模式的 ANN 文件
python run_vector_test.py ann \
  --sql-mode l2_filter \
  --distribute-file-ids \
  -n 1000 -k 10

# 生成距离阈值过滤模式的 ANN 文件
python run_vector_test.py ann --sql-mode l2_filter_threshold --distribute-file-ids -n 1000 -k 10
```

**参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sql-mode` | l2_only | SQL 模式：`l2_only`、`l2_filter`、`l2_filter_threshold` |
| `-n, --num-queries` | 1000 | 查询数量 |
| `-k` | 10 | Top-K |
| `--concurrency` | 1 | 并发数 |
| `--filter-val` | - | file_id 过滤值（用于 l2_filter 模式） |
| `--distribute-file-ids` | - | 将查询分布到多个不同的 file_id |
| `--max-distinct-file-ids` | 50 | 最多使用多少个不同的 file_id |

**注意**：生成ann文件是暴力搜索生成的预期结果，-n 1000值越大生成文件越慢，可以调节变小缩短生成时间，适用于快速验证测试  

## 命令详解

### 全局参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 127.0.0.1 | 数据库主机 |
| `--port` | 6001 | 端口 |
| `--user` | dump | 用户名 |
| `--password` | 111 | 密码 |
| `--database` | jst_app_wiki | 数据库名 |
| `--table` | historical_file_blocks_wiki | 表名 |


**参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-n, --num-queries` | 1000 | 查询数量 |
| `-k` | 10 | Top-K |
| `--concurrency` | 4 | 并发数 |
| `--sql-mode` | l2_only | SQL 模式：`l2_only`（全表搜索）、`l2_filter`（预过滤）、`l2_filter_threshold`（带距离阈值） |
| `--filter-val` | - | file_id 过滤值（用于 l2_filter 和 l2_filter_threshold 模式） |

**使用示例：**

```bash
# 全表搜索（默认 l2_only）
python run_vector_test.py wiki test -n 1000 -k 10 --concurrency 100

# 预过滤模式（指定 file_id）
python run_vector_test.py wiki test \
  --sql-mode l2_filter \
  --filter-val 20000000 \
  -n 1000 -k 10 --concurrency 100
```

### `wiki info` - 显示数据集信息

```bash
python run_vector_test.py wiki info
```

### `ann` - 生成 ANN 评测文件

生成 `query.fvecs` 和 `groundtruth.ivecs` 文件，用于后续评估。

```bash
# 生成全表搜索模式的 ANN 文件
python run_vector_test.py ann --sql-mode l2_only -n 1000 -k 10

# 生成预过滤模式的 ANN 文件（分布到多个 file_id）
python run_vector_test.py ann \
  --sql-mode l2_filter \
  --distribute-file-ids \
  --max-distinct-file-ids 50 \
  -n 1000 -k 10
```

**参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sql-mode` | l2_only | SQL 模式：`l2_only`、`l2_filter`、`l2_filter_threshold` |
| `-n, --num-queries` | 1000 | 查询数量 |
| `-k` | 10 | Top-K |
| `--concurrency` | 1 | 并发数 |
| `--filter-val` | - | file_id 过滤值（用于 l2_filter 模式） |
| `--distribute-file-ids` | - | 将查询分布到多个不同的 file_id |
| `--max-distinct-file-ids` | 50 | 最多使用多少个不同的 file_id |

### `run` - 运行召回率/QPS 评估

运行向量搜索性能评估，输出召回率和 QPS。支持三种 SQL 场景和多种 Filter 模式，可全面评估向量索引在不同配置下的性能表现。

```bash
# 全表搜索评估（l2_only 场景）
python run_vector_test.py run --sql-mode l2_only -n 1000 -k 10 --concurrency 100

# 预过滤模式评估（l2_filter 场景）
python run_vector_test.py run \
  --sql-mode l2_filter \
  --filter-val 20000000 \
  -n 1000 -k 10 --concurrency 100

# 多 file_id 分布测试（适合多租户场景）
python run_vector_test.py run \
  --sql-mode l2_filter \
  --distribute-file-ids \
  --max-distinct-file-ids 50 \
  -n 1000 -k 10 --concurrency 100

# 压测模式（持续 120 秒，输出 QPS 和延迟分布）
python run_vector_test.py run \
  --sql-mode l2_only \
  -n 1000 -k 10 --concurrency 100 \
  --duration 120

# 调整 IVF 索引 probe 参数测试召回率
python run_vector_test.py run \
  --sql-mode l2_filter \
  --filter-val 20000000 \
  -n 1000 -k 10 --concurrency 100 \
  --probe 20

# pre 模式测试（预过滤，性能优先）
python run_vector_test.py run \
  --sql-mode l2_filter \
  --filter-val 20000000 \
  -n 1000 -k 10 --concurrency 100 \
  --filter-mode pre

# force 模式测试（精确搜索，作为 baseline）
python run_vector_test.py run \
  --sql-mode l2_filter \
  --filter-val 20000000 \
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
| `--filter-val` | - | file_id 过滤值（用于 l2_filter 场景） |
| `--duration` | - | 压测持续时间（秒） |
| `--probe` | - | IVF 索引 probe 参数，控制查询时扫描的聚类数 |
| `--distribute-file-ids` | - | 将查询分布到多个不同的 file_id |
| `--max-distinct-file-ids` | 50 | 最多使用多少个不同的 file_id |
| `--skip-db-verify` | - | 跳过数据库预检 |

## 完整示例

```bash
#!/bin/bash

# 下载数据集（以 1M 为例）
curl -L -O https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar
tar -xf wiki_all_1M.tar

# 一键完成所有步骤（创建表、导入、建索引、测试）
python run_vector_test.py \
  --host 127.0.0.1 \
  --port 6001 \
  --user dump \
  --password 111 \
  --database jst_app_wiki \
  --table historical_file_blocks_wiki \
  wiki setup \
  --fbin wiki_all_1M/base.fbin \
  --ivf-lists 100 \
  --auto-test \
  -n 1000 \
  -k 10 \
  --concurrency 4
```

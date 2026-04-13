# 向量搜索测试工具

一套用于测试向量数据库（MatrixOne）搜索性能的工具集，支持数据生成、环境初始化、ANN 文件生成和召回率/QPS 评估。

## 组件说明

| 文件 | 说明 |
|------|------|
| `generate_historical_file_blocks.py` | 生成测试数据 CSV |
| `eval_vector_search_from_table.py` | 向量搜索召回率和 QPS 评估 |
| `run_vector_test.py` | 统一入口脚本，支持配置化 SQL 模式 |
| `sql_config_simple.json` | SQL 模式配置文件 |

## 快速开始

### 1. 安装依赖

```bash
pip install numpy pymysql
```

### 2. 一键完整测试流程

```bash
# 自动生成数据、创建表、导入数据、建索引、运行测试
python run_vector_test.py init \
  --database jst_app \
  --table historical_file_blocks \
  --auto-generate -n 100k \
  --distinct-file-ids 50 \
  --create-db \
  --create-table \
  --create-index \
  --index-type ivf \
  --auto-run \
  --sql-modes m2_l2_filter \
  --concurrency 100
```

## 命令详解

### `init` - 初始化环境

创建数据库、创建表、导入数据、创建向量索引。

```bash
# 从 CSV 文件初始化
python run_vector_test.py init \
  --database jst_app \
  --table historical_file_blocks \
  --data-file data.csv \
  --create-db \
  --create-table \
  --create-index \
  --index-type ivfflat

# 自动生成数据并初始化
python run_vector_test.py init \
  --database jst_app \
  --table historical_file_blocks \
  --auto-generate -n 1m \
  --distinct-file-ids 100 \
  --create-db \
  --create-table \
  --create-index

# 创建 IVF 索引（指定 lists 和 probe）
python run_vector_test.py init \
  --database jst_app \
  --table historical_file_blocks \
  --create-index \
  --index-type ivfflat \
  --ivf-lists 100 \
  --op-type vector_l2_ops
```

#### init 参数说明

**数据库配置**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 127.0.0.1 | 数据库主机 |
| `--port` | 6001 | 端口 |
| `--user` | dump | 用户名 |
| `--password` | 111 | 密码 |
| `--database` | jst_app | 数据库名 |
| `--table` | historical_file_blocks | 表名 |

**数据选项**
| 参数 | 说明 |
|------|------|
| `--data-file` | 指定已有 CSV 数据文件 |
| `--auto-generate` | 自动生成数据 |
| `-n, --rows` | 生成行数（如 100k, 1m, 10m） |
| `--distinct-file-ids` | 不同的 file_id 数量 |

**初始化选项**
| 参数 | 说明 |
|------|------|
| `--create-db` | 创建数据库 |
| `--create-table` | 创建表 |
| `--create-index` | 创建向量索引 |
| `--index-type` | 索引类型（默认: ivf） |
| `--ivf-lists` | IVF 聚类中心数 |
| `--op-type` | 向量操作类型（默认: vector_l2_ops） |

**自动运行选项**
| 参数 | 说明 |
|------|------|
| `--auto-run` | 初始化完成后自动运行测试 |
| `--sql-modes` | 自动运行时的 SQL 模式 |
| `--num-queries` | 查询数量 |
| `-k` | Top-K |
| `--concurrency` | 并发数 |

### `generate` - 生成测试数据

```bash
# 生成 100万行数据，50 个不同 file_id
python run_vector_test.py generate -n 1m --distinct-file-ids 50

# 生成 1000万行数据，指定输出文件
python run_vector_test.py generate -n 10m --distinct-file-ids 50 -o big_data.csv
```

### `ann` - 生成 ANN 评测文件

根据 SQL 模式生成 `query.fvecs` 和 `groundtruth.ivecs` 文件。

```bash
# 使用配置文件的 m1_l2_only 模式
python run_vector_test.py ann \
  --sql-mode m1_l2_only \
  -n 1000 \
  -k 10

# 使用 m2_l2_filter 模式，指定 filter_mode
python run_vector_test.py ann \
  --sql-mode m2_l2_filter \
  --filter-mode post \
  -n 1000 \
  -k 10

# 指定数据库连接
python run_vector_test.py ann \
  --sql-mode m2_l2_filter \
  --host 192.168.1.100 \
  --port 6001 \
  --user admin \
  --password secret \
  --database mydb \
  --table my_vectors \
  -n 1000 \
  -k 10
```

### `eval` - 运行召回率/QPS 评估

```bash
# 使用 m3_l2_filter_threshold 模式评估
python run_vector_test.py eval \
  --sql-mode m3_l2_filter_threshold \
  -n 1000 \
  -k 10 \
  --concurrency 4

# 压测模式（持续 60 秒）
python run_vector_test.py eval \
  --sql-mode m2_l2_filter \
  --filter-mode pre \
  -n 1000 \
  -k 10 \
  --concurrency 8 \
  --duration 60
```

### `list-modes` - 列出所有 SQL 模式

```bash
python run_vector_test.py list-modes

# 使用自定义配置文件
python run_vector_test.py list-modes --config my_config.json
```

### `run` - 完整测试流程

## 配置文件说明

### `sql_config_simple.json` 格式

```json
{
  "sql_modes": {
    "m1_l2_only": {
      "type": "l2_only",
      "sql": "SELECT `file_id`, `id` FROM `{table}` ORDER BY l2_distance(`{emb_col}`, %s) ASC LIMIT %s"
    },
    "m2_l2_filter": {
      "type": "l2_filter",
      "sql": "SELECT `file_id`, `id`, l2_distance(`{emb_col}`, %s) AS dist FROM `{table}` WHERE `{filter_col}` = %s ORDER BY dist ASC LIMIT %s"
    },
    "m3_l2_filter_threshold": {
      "type": "l2_filter_threshold",
      "sql": "SELECT `file_id`, `id`, l2_distance(`{emb_col}`, %s) AS dist FROM `{table}` WHERE `{filter_col}` = %s AND l2_distance(`{emb_col}`, %s) <= {max_distance} ORDER BY dist ASC LIMIT %s",
      "extra": { "max_distance": 1.77 }
    }
  },
  "default": {
    "table": "historical_file_blocks",
    "emb_col": "embedding",
    "filter_col": "file_id"
  }
}
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `type` | SQL 模式类型，决定数据抽样方式：<br>- `l2_only`: 全表搜索，不需要 file_id<br>- `l2_filter`: 预过滤后搜索，需要 file_id<br>- `l2_filter_threshold`: 带距离阈值的预过滤，需要 file_id |
| `sql` | SQL 模板，占位符说明：<br>- `{table}`: 表名<br>- `{emb_col}`: 向量列名<br>- `{filter_col}`: 过滤列名<br>- `%s`: 参数占位符（query_vec, filter_val, k）<br>- `{max_distance}`: 距离阈值（从 extra 读取） |
| `extra` | 额外参数，可在 SQL 模板中使用 `{key}` 引用 |
| `default` | 默认表/列名配置 |

### filter_mode 说明

在 `ann` 和 `eval` 命令中通过 `--filter-mode` 指定 Ground Truth SQL 的后缀：

| filter_mode | SQL 后缀 | 说明 |
|------------|----------|------|
| `force` | `BY RANK WITH OPTION 'mode=force'` | 强制精确模式（默认） |
| `pre` | `BY RANK WITH OPTION 'mode=pre'` | 预过滤模式 |
| `post` | 无后缀 | 后过滤模式 |

## SQL 执行流程

### Ground Truth 查询（用于计算召回率）

```
基础 SQL（来自配置文件）
    ↓
添加 filter_mode 后缀
    ↓
生成 Ground Truth 结果
```

### 索引查询（用于计算 QPS）

```
基础 SQL（来自配置文件，无后缀）
    ↓
执行索引查询
    ↓
生成搜索结果
```

### 召回率计算

```
Ground Truth 结果 vs 索引查询结果
    ↓
计算 Recall@K = |GT ∩ Result| / K
```

## 向量索引说明

本工具使用 IVF（Inverted File）索引进行向量近似搜索。IVF 索引通过聚类将向量空间划分为多个区域，搜索时只需探查部分聚类，大幅提升搜索速度。

### IVF 索引

```sql
CREATE INDEX idx_l2 USING ivfflat ON table(embedding) lists=1000 op_type "vector_l2_ops"
```

- 聚类近似搜索
- `lists`: 聚类中心数量，建议设置为数据量的平方根
- `op_type`: 向量操作类型，支持 `vector_l2_ops` (L2距离)
- `lists`: 聚类中心数量，通常设置为数据量的平方根
- `probe`: 搜索时探查的聚类数，越大召回率越高
- 适合大数据量场景，需要权衡召回率和速度

## 数据生成说明

生成的 CSV 格式（6 列）：

```csv
\N,20000000,"content text...","[-0.028884888,0.03186035,...]",1,"{\"seq\":1,...}"
\N,20000001,"content text...","[0.0064964294,...]",2,"{\"seq\":2,...}"
```

| 列 | 说明 |
|---|------|
| id | \N（自增） |
| file_id | 过滤 ID（默认 20000000 起始） |
| content | 文本内容 |
| embedding | 1024 维向量（VEC32） |
| page_num | 页码 |
| meta | JSON 元数据 |

## 常见问题

### 1. 数据导入失败

MatrixOne 可能需要特定格式：

```bash
# 手动导入（调整字段分隔符和引号）
mysql -h127.0.0.1 -P6001 -udump -p111 -e "
LOAD DATA INFILE '/path/to/data.csv'
INTO TABLE jst_app.historical_file_blocks
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '\"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(id, file_id, content, embedding, page_num, meta)
"
```

### 2. 召回率过低

- 检查索引类型和参数（IVF 的 lists/probe 设置）
- 检查 `filter_mode` 设置是否合理
- 确认 Ground Truth 使用的是精确模式（force）

### 3. 并发测试报错

- 检查数据库连接数限制
- 适当降低 `--concurrency` 值
- 确认网络连接稳定

## 完整示例

```bash
#!/bin/bash

# 步骤 1: 生成 100万条测试数据
python run_vector_test.py generate \
  -n 1m \
  --distinct-file-ids 50 \
  -o test_data.csv

# 步骤 2: 初始化环境（创建库、表、导入、建索引）
python run_vector_test.py init \
  --database jst_app \
  --table historical_file_blocks \
  --data-file test_data.csv \
  --create-db \
  --create-table \
  --create-index \
  --index-type ivfflat \
  --ivf-lists 1000 \
  --op-type vector_l2_ops

# 步骤 3: 生成 ANN 文件（用于后续快速测试）
python run_vector_test.py ann \
  --sql-mode m2_l2_filter \
  --filter-mode force \
  --database jst_app \
  --table historical_file_blocks \
  -n 1000 \
  -k 10

# 步骤 4: 运行评估
python run_vector_test.py eval \
  --sql-mode m2_l2_filter \
  --filter-mode pre \
  --database jst_app \
  --table historical_file_blocks \
  -n 1000 \
  -k 10 \
  --concurrency 100

# 或者使用一键流程
python run_vector_test.py init \
  --database jst_app \
  --table historical_file_blocks \
  --auto-generate -n 100k \
  --distinct-file-ids 50 \
  --create-db \
  --create-table \
  --create-index \
  --index-type ivfflat \
  --auto-run \
  --sql-modes m2_l2_filter \
  --concurrency 100
```

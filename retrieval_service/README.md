# retrieval_service

基于 LanceDB 的 BM25 + 向量混合检索 HTTP 微服务，是 page-know-how 主项目的存储/检索后端。

## 设计要点

- **零外部依赖部署**：纯 Python wheel（lancedb + pyarrow），单进程 uvicorn 即可启动；无需 Docker 即可裸跑。
- **每 policy 一张表**：物理上 `STORE_DIR/{safe_policy_id}.lance/`，互不干扰；表删除等价于 `rm -rf`。
- **服务端不依赖 jieba**：分词在客户端做（主项目 `inference/retrieval/bm25.py::tokenize`），服务端只接收已分词字符串，FTS 索引 base_tokenizer 用 `whitespace`，分词逻辑 100% 与现有 BM25 同源。
- **本地 RRF 融合**：跟主项目 `inference/retrieval/rrf.py` 同公式 `1/(k+rank)`，避开 LanceDB 内置 hybrid+reranker 的版本差异。
- **关联结构原生入索**：`kind`/`parent_chunk_index`/`derived_seq`/`relation_keys`/`hop_depth`/`source`/`clause_id` 都是表列，推理期可按高亮链路展开 / 反查；跨 policy cascade 走全局接口 `/v1/relations:lookup-dependents`。

## 表 schema（pyarrow）

| 列 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | int64 | 主键 = 原 KnowledgeChunk.index |
| `content` | string | 给 LLM 用 |
| `content_tokenized` | string | **客户端 jieba 分词后空格连接** |
| `vector` | fixed_size_list&lt;float32&gt;[dim] | 客户端 embed 后送入 |
| `heading_paths` | list&lt;list&lt;string&gt;&gt; | KnowledgeChunk 元数据 |
| `directories` | list&lt;string&gt; | KnowledgeChunk 元数据 |
| `kind` | string | original \| derived |
| `parent_chunk_index` | int64 | 派生 chunk 父 chunk_id；原始 -1 |
| `derived_seq` | int32 | 同父下的序号 |
| `relation_keys` | list&lt;struct&lt;policy_id, clause_id&gt;&gt; | 派生命中的外部条款 |
| `hop_depth` | int32 | 派生跳数 |
| `source` | string | local/remote/missing |
| `clause_id` | string | 派生 chunk 的目标条款 id |
| `built_at` | int64 | 毫秒时间戳 |

## 快速开始

### 本地裸跑

```bash
cd retrieval_service
python -m venv .venv && .venv\Scripts\activate          # Windows
# source .venv/bin/activate                              # Linux/Mac
pip install -e .
cp .env.example .env                                     # Windows: copy .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8088 --reload
```

健康检查：

```bash
curl http://127.0.0.1:8088/healthz
```

### Docker

```bash
cd retrieval_service
cp .env.example .env
docker compose up -d --build
```

数据持久化在 `./data/` 目录（容器内 `/app/data`）。

## API 概览

所有接口需带 `X-API-Key: <API_KEY>` 或 `Authorization: Bearer <API_KEY>` 头；当 `API_KEY` 为空时关闭鉴权（仅本地开发）。

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/healthz` | 健康检查 |
| POST | `/v1/policies/{policy_id}/chunks:upsert` | 整批写入（mode=overwrite\|append\|merge_by_chunk_id） |
| GET | `/v1/policies/{policy_id}/chunks` | 列出 chunks，支持 where/limit |
| DELETE | `/v1/policies/{policy_id}` | 删表 |
| POST | `/v1/policies/{policy_id}/search` | 混合检索（query_tokenized + query_vector） |
| POST | `/v1/policies/{policy_id}/relations:expand` | 取某父 chunk 的派生 chunks |
| GET | `/v1/policies/{policy_id}/relations:lookup` | 单 policy 内反查依赖某 (target_policy_id, target_clause_id) 的 chunks |
| GET | `/v1/relations:lookup-dependents` | 全局反查：哪些源 policy 引用了 target |
| GET | `/v1/policies` | 列出所有 policy |
| GET | `/v1/policies/{policy_id}/meta` | 表行数 / dim / 索引状态 |

## 与主项目的对接

主项目 `inference/retrieval/client.py` 是这个服务的官方 SDK（基于 httpx async）。`hybrid_search` / `indexer.build_for_root` / `app.py._cascade_dependent_rebuilds` 三处会调用本服务。

主项目 `.env` 需要新增：

```env
RETRIEVAL_SERVICE_URL=http://127.0.0.1:8088
RETRIEVAL_SERVICE_API_KEY=changeme
```

## 测试

```bash
pip install -e ".[dev]"
pytest -q
```

`tests/test_routes.py` 跑一整套 upsert / search / expand / lookup / drop 的端到端冒烟。

---
name: ""
overview: ""
todos:
  - id: scaffold
    content: 在仓库根目录新建 retrieval_service/ 项目，pyproject.toml + Dockerfile + README + .env.example，依赖 lancedb>=0.13 / pyarrow>=15 / fastapi / uvicorn / pydantic v2 / anyio / structlog（注意：服务端不依赖 jieba），独立 venv
    status: completed
  - id: server-schema
    content: retrieval_service/app/schema.py 定义 pyarrow schema（chunk_id/policy_id/content/content_tokenized/vector/heading_paths/directories/kind/parent_chunk_index/derived_seq/relation_keys/hop_depth/source/clause_id/built_at），并提供 ensure_table(policy_id, dim) helper
    status: completed
  - id: server-store
    content: retrieval_service/app/store.py 封装 lancedb.connect(STORE_DIR) + 表缓存 + create_fts_index(base_tokenizer=whitespace) + create_index(vector cosine) + 标量索引，全部用 anyio.to_thread 包成异步以避开 GIL 阻塞
    status: completed
  - id: client-tokenizer
    content: jieba 分词留在客户端：保留 inference/retrieval/bm25.py 文件但只暴露 tokenize() / tokenize_join(text)->str；删除 build/save/load/search 等存储侧函数。indexer 写入与 hybrid 查询共用同一份 tokenize_join，服务端只接收已分词字符串
    status: completed
  - id: server-routes
    content: retrieval_service/app/routers/ 实现 POST /v1/policies/{policy_id}/chunks:upsert（入参含 content + content_tokenized + vector）、POST /v1/policies/{policy_id}/search（入参含 query_tokenized + query_vector）、POST /v1/policies/{policy_id}/relations:expand、GET /v1/policies/{policy_id}/relations:lookup、DELETE /v1/policies/{policy_id}、GET /v1/policies/{policy_id}/meta、GET /healthz；所有路由 pydantic 入参
    status: completed
  - id: server-rerank
    content: search 路由使用 LanceDB 内置 RRFReranker(K=60)，对齐现有 inference/retrieval/rrf.py 行为；同时支持 top_n / top_m / where 过滤（kind/parent_chunk_index 等）
    status: pending
  - id: server-auth
    content: 简单 X-API-Key header 鉴权 + CORS + structlog 日志 + 健康检查；fastapi lifespan 里预热打开常用 policy 表
    status: pending
  - id: client-rewrite
    content: 主项目 inference/retrieval/ 改造：bm25.py 瘦身为只保留 tokenize/tokenize_join；删除 semantic.py / rrf.py；新增 inference/retrieval/client.py 用 httpx.AsyncClient 调用服务；hybrid.py 主入口签名不变内部走 HTTP；indexer.py 写入路径换成"客户端预计算 tokenized + embed_texts 算 vector"后调 upsert 接口
    status: pending
  - id: client-cascade
    content: app.py 的 _cascade_dependent_rebuilds 改为：调服务的 relations:lookup 端点，按 (policy_id, clause_id) 反查依赖该外部条款的 source policies；旧 _relation_targets.json 不再写
    status: pending
  - id: ab-eval
    content: tests/eval_lance_vs_legacy.py 同一组 question/policy 对比新（HTTP→LanceDB）和旧（npy+pkl）top10 chunk_id 重合率，阈值≥0.8 才切默认；测脚本本身只读，跑完输出 markdown 报告
    status: pending
  - id: deploy-docs
    content: retrieval_service/README.md 写本地启动（uvicorn）+ docker-compose（mount ./data 卷）+ 主项目 .env 增加 RETRIEVAL_SERVICE_URL / RETRIEVAL_SERVICE_API_KEY；requirements.txt 增加 httpx，移除 rank_bm25
    status: completed
isProject: false
---

# 用 LanceDB + FastAPI 服务化替换三件套：方案落地

## 选型最终结论

- **检索服务**：根目录新建 `retrieval_service/`，独立 FastAPI 项目，内嵌 LanceDB（`pip install lancedb pyarrow`，无外部 db 进程）。
- **职责边界**（保持单一分词源）：
  - **客户端（主项目）**：`jieba 分词 + embedding 计算 + HTTP 调用`（[inference/retrieval/bm25.py](inference/retrieval/bm25.py)::`tokenize()` 保持不动，[inference/embedding_client.py](inference/embedding_client.py) 保持不动）
  - **服务端**：`存储 + BM25/向量/混合检索 + 关联结构反查`；**不依赖 jieba**，只接收已分词字符串和向量
- **多租户布局**：一个 policy 一张 lance 表，存放在服务端 `STORE_DIR/{policy_id}.lance/`。
- **中文 BM25 走"预分词列"**（已查实：LanceDB 当前版本对 jieba/Lindera 的官方集成不完整；预分词列由客户端写入，服务端 FTS 用 whitespace tokenizer 处理，质量 100% 与现有 BM25 同源）。

## 总体架构

```mermaid
flowchart LR
  subgraph MainApp[main project]
    A[indexer build_for_root] --> A1[jieba tokenize_join + embed_texts]
    A1 -->|HTTP upsert<br/>content + content_tokenized + vector| C[retrieval_service /v1/.../upsert]
    B[hybrid_search question, policy_id] --> B1[jieba tokenize_join + embed_texts]
    B1 -->|HTTP search<br/>query_tokenized + query_vector| D[retrieval_service /v1/.../search]
    M[app cascade trigger] -->|HTTP lookup| E[retrieval_service /v1/.../relations:lookup]
  end
  subgraph Service[retrieval_service FastAPI no jieba]
    C --> S2[lance write + ensure FTS/vec index]
    D --> S3[hybrid query with RRFReranker]
    E --> S4[scalar filter on relation_keys]
    S2 --> ST[(STORE_DIR/{policy_id}.lance)]
    S3 --> ST
    S4 --> ST
  end
```

## 项目骨架（根目录新建 `retrieval_service/`）

```
retrieval_service/
  pyproject.toml          # lancedb>=0.13 pyarrow>=15 fastapi uvicorn pydantic>=2 anyio structlog
                          # 注意：不依赖 jieba，分词在客户端做
  Dockerfile              # python:3.12-slim + uvicorn entrypoint
  docker-compose.yml      # 单服务，挂载 ./data 卷
  .env.example            # STORE_DIR=./data  API_KEY=xxx  HOST=0.0.0.0  PORT=8088
  README.md
  app/
    __init__.py
    main.py               # FastAPI() + lifespan + 鉴权中间件 + 路由挂载
    config.py             # pydantic-settings 读 .env
    schema.py             # pyarrow schema + KnowledgeChunkRow pydantic 模型
    store.py              # LanceDB 连接、表缓存、ensure_indexes
    deps.py               # X-API-Key 校验依赖
    routers/
      health.py           # GET /healthz
      chunks.py           # upsert / delete / list
      search.py           # hybrid search（直接用 query_tokenized）
      relations.py        # expand / lookup
      meta.py             # 表 meta、stats、drop
  tests/
    test_routes.py        # httpx + pytest，端到端冒烟
```

## 核心 schema（`retrieval_service/app/schema.py`）

pyarrow（**dim 由首次 upsert 时的 vector 长度推断并落 meta**，禁止后续不一致写入）：

| 列 | 类型 | 含义 |
|---|---|---|
| `chunk_id` | int64 | = 旧 `index`，主键 |
| `content` | string | 原文，给 LLM 用 |
| `content_tokenized` | string | **客户端 jieba 分词后空格连接**，给 BM25 用 |
| `vector` | fixed_size_list<float32>[dim] | embedding（客户端送） |
| `heading_paths` | list<list<string>> | KnowledgeChunk.heading_paths |
| `directories` | list<string> | KnowledgeChunk.directories |
| `kind` | string | `original` \| `derived` |
| `parent_chunk_index` | int64 | 派生 chunk 的父 chunk_id；原始 -1 |
| `derived_seq` | int32 | 同父下的派生序号 |
| `relation_keys` | list<struct<policy_id, clause_id>> | 命中外部条款 keys |
| `hop_depth` | int32 | 派生跳数；原始 0 |
| `source` | string | `local`/`remote`/`missing` |
| `clause_id` | string | 派生 chunk 的目标条款 id |
| `built_at` | int64 | 毫秒时间戳 |

索引（`store.ensure_indexes(tbl)` 内一次性建好）：

- `tbl.create_fts_index("content_tokenized", base_tokenizer="whitespace")`
- `tbl.create_index("vector", metric="cosine")`（< 5w chunks 跳过 IVF_PQ，全量扫即可）
- `tbl.create_scalar_index("kind")` / `tbl.create_scalar_index("parent_chunk_index")`

## HTTP API 设计（`retrieval_service/app/routers/`）

所有接口走 `Authorization: Bearer <API_KEY>` 或 `X-API-Key` header，统一 JSON。

| 方法 + 路径 | 用途 | 入参（精简） | 出参 |
|---|---|---|---|
| `POST /v1/policies/{policy_id}/chunks:upsert` | 整批写入/覆盖（mode=overwrite\|append\|merge_by_chunk_id） | `chunks: [{chunk_id, content, content_tokenized, vector, heading_paths, directories, kind, parent_chunk_index, derived_seq, relation_keys, hop_depth, source, clause_id}]`, `mode` | `{written: N, table_size: M, dim: D}` |
| `POST /v1/policies/{policy_id}/search` | 混合检索（**客户端预分词**） | `query_tokenized: str, query_vector: list[float], top_n, top_m, where?: str, include?: ['relation_keys','directories',...]` | `[{chunk_id, score, content, kind, parent_chunk_index, ...}]` |
| `POST /v1/policies/{policy_id}/relations:expand` | 取某 chunk 的派生关联 chunks | `chunk_id, max_depth?` | `chunks: [...]` |
| `GET  /v1/policies/{policy_id}/relations:lookup` | 反查依赖外部条款的派生 chunks（cascade 用） | query: `target_policy_id, target_clause_id` | `[{chunk_id, parent_chunk_index, source}]` |
| `DELETE /v1/policies/{policy_id}` | 删表 | — | `{ok: true}` |
| `GET  /v1/policies/{policy_id}/meta` | 表元数据：行数、dim、built_at、has_vector_index | — | `{...}` |
| `GET  /v1/policies` | 列出所有 policy 表 | — | `[{policy_id, n_chunks, dim, built_at}]` |
| `GET  /healthz` | 健康检查 | — | `{ok: true, version: ...}` |

**search 实现关键**（`routers/search.py`，服务端无分词逻辑）：

```python
hits = (
    tbl.search(query_type="hybrid")
       .vector(req.query_vector)
       .text(req.query_tokenized)   # 客户端已做 jieba 分词，FTS 走 whitespace tokenizer
       .limit(max(req.top_n, req.top_m))
       .rerank(reranker=RRFReranker(K=60))
       .where(req.where)             # e.g. "kind = 'original'"
       .to_arrow()
)
```

## 主项目改造（最小侵入）

1. **新增 [inference/retrieval/client.py](inference/retrieval/client.py)**（约 150 行）：
   - `class RetrievalServiceClient`（httpx.AsyncClient + 连接池 + 重试 + 超时）
   - `async upsert(policy_id, chunks, mode)` / `async search(policy_id, ...)` / `async expand(...)` / `async lookup(...)` / `async drop(policy_id)`
   - 序列化 `KnowledgeChunk` → 服务端 schema 时，把 `parent_chunk_index/derived_seq/relation_keys` 这些**当前 `_serialize_chunk` 丢失的字段**补回去（这是顺带修的关键 bug）。

2. **保留并瘦身 [inference/retrieval/bm25.py](inference/retrieval/bm25.py)**（jieba 分词留在客户端）：
   - 保留 `tokenize(text) -> list[str]`（现有实现不动，含停用词、`_NON_WORD` 过滤）；
   - 新增 `tokenize_join(text) -> str`（= `" ".join(tokenize(text))`），给 indexer 与 hybrid 共用；
   - **删除** `build/save/load/search` 四个存储侧函数（统一交给服务端 LanceDB）；
   - 模块改名为可选：保留路径不变以减小 diff，仅删除存储相关函数体。

3. **重写 [inference/retrieval/indexer.py](inference/retrieval/indexer.py)**：
   - 删 `_write_chunks_jsonl` / `bm25_mod.build/save` / `semantic_mod.save` 三段落盘；
   - `build_for_root` 内对每个 chunk：先 `content_tokenized = bm25.tokenize_join(chunk.content)`，再批量 `embed_texts(contents)` 拿向量，最后 `await client.upsert(policy_id, [{content, content_tokenized, vector, ...metadata}], mode="overwrite")`；
   - `rebuild_embeddings_only`：通过 `client.list_chunks(policy_id)` 拉回现有 content（不动 tokenized 列），重算向量后 `mode="merge_by_chunk_id"` 只更 vector 列。

4. **重写 [inference/retrieval/hybrid.py](inference/retrieval/hybrid.py)**：
   - 删 `_PolicyArtifacts`、`_load_artifacts`、本地缓存等；
   - `hybrid_search(question, policy_id, top_n, top_m)` 内：
     ```python
     q_tok = bm25.tokenize_join(question)
     q_vec = (await embed_texts([question]))[0]
     resp = await client.search(
         policy_id,
         query_tokenized=q_tok,
         query_vector=q_vec,
         top_n=top_n, top_m=top_m,
     )
     return [KnowledgeChunk(**row) for row in resp]
     ```
   - 自愈逻辑（`_maybe_trigger_embedding_rebuild`）保留语义，但不再"后台重写 .npy"，改为后台调 `/meta` 检查 `dim is null` 后触发 `rebuild_embeddings_only`。

5. **删除文件**：
   - `inference/retrieval/semantic.py`、`inference/retrieval/rrf.py`（功能交给服务端）；
   - `inference/retrieval/bm25.py` **保留**（仅保留分词函数）。

6. **`app.py` 调整**：
   - `_cascade_dependent_rebuilds` 不再读 `_relation_targets.json`，改为遍历所有 policy 调 `relations:lookup` 反查；
   - 旧的 `_chunks.jsonl / _bm25.pkl / _embeddings.npy / _index_meta.json / _relation_targets.json` 路径相关代码统一删除（按你"彻底替换"的选择）。

7. **依赖变更**：
   - 主项目 `requirements.txt`：加 `httpx>=0.27`，删 `rank_bm25`，**保留 jieba**（客户端分词）；
   - 服务端 `retrieval_service/pyproject.toml`：**不含 jieba/rank_bm25**，见上面骨架。

## 高亮关联展开 / 推理期检索路径

- **推理期 hybrid 检索**：主项目 `react_loop` → `hybrid.hybrid_search` → HTTP `/search` → 服务端 RRF hybrid → 返回 `KnowledgeChunk` 列表，**签名兼容、上层无感**。
- **命中后展开**：`reasoner` 拿到 chunk 想看派生关联 → `client.expand(policy_id, chunk_id)` → 服务端 `where parent_chunk_index = ? and kind = 'derived'`。
- **跨 policy cascade**（替换 `_relation_targets.json`）：某 policy 更新时，`app.py` 遍历其它 policy 调 `relations:lookup?target_policy_id=...&target_clause_id=...`，命中即触发对应 source policy 重建。

## 部署与运行

- **本地开发**：`cd retrieval_service && uvicorn app.main:app --port 8088 --reload`，主项目 `.env` 写 `RETRIEVAL_SERVICE_URL=http://127.0.0.1:8088 RETRIEVAL_SERVICE_API_KEY=...`。
- **容器化**：`retrieval_service/Dockerfile` 用 `python:3.12-slim`，`docker-compose.yml` 挂 `./data:/app/data` 卷；主项目侧只配 URL 即可。
- **数据迁移**：保留一个一次性脚本 `retrieval_service/scripts/migrate_legacy.py`（**只读旧三件套**），把 `_chunks.jsonl + _embeddings.npy` 通过 HTTP upsert 到服务端；跑完后旧文件可删。
- **服务侧目录**：`STORE_DIR/{policy_id}.lance/`，policy_id 从主项目 `_policy_index.json` 解析后作为路径片段（注意中文/特殊字符 → 服务端做 `urlsafe_b64` 编码后入库，HTTP 上仍用原始 policy_id）。

## 风险与回退

- **风险 1：HTTP 往返多耗 5–15ms / 请求**。RAG 场景每次查询 1–2 次往返可接受，但批量 upsert 时要走 chunk 分批（建议 200 条 / 请求），避免单请求体过大；客户端用 `httpx.AsyncClient` 持久连接 + http2。
- **风险 2：服务进程崩溃单点**。LanceDB 是文件型存储，重启即可；建议 supervisord/systemd/容器 restart 策略。
- **风险 3：LanceDB hybrid 内置 RRFReranker 与现有 `rrf.py` 行为差异**。上线前用 `tests/eval_lance_vs_legacy.py` 跑 ab，top10 重合率 ≥ 0.8 才切默认。
- **回退**：保留旧三件套读取代码到一个 `inference/retrieval/_legacy.py` 文件里 + 一个开关（默认关，紧急时打开），不污染主流程。

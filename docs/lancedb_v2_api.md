# LanceDB Retrieval Service v2 接口文档（通用能力）

本文档面向需要依赖本服务 **v2 通用能力** 的调用方（服务对服务），覆盖鉴权、能力协商、集合管理、文档写入/读取、混合检索与错误语义。

## 1. 基本信息

- Base URL 示例：`http://127.0.0.1:5000`
- 版本前缀：`/v2`
- 数据格式：`application/json`
- 鉴权方式（两选一）：
  - `X-API-Key: <API_KEY>`
  - `Authorization: Bearer <API_KEY>`

> 当服务端 `API_KEY` 为空时，鉴权关闭（本地开发场景，当前服务端也没有设置API_KEY）。

### 1.1 服务端 embedding 兜底配置

默认开启服务端 embedding 兜底（用于 `documents:upsert` 缺 `vector`、`search` 缺 `query_vector`）。

相关环境变量：

- `ENABLE_SERVER_EMBEDDING_FALLBACK`：是否启用兜底，默认 `true`
- `EMBEDDING_BASE_URL`：embedding 服务地址，默认 `http://mlp.paas.dc.servyou-it.com/qwen3-embedding/v1`
- `EMBEDDING_MODEL`：embedding 模型名，默认 `qwen3-embedding`
- `EMBEDDING_API_KEY`：可选，默认空
- `EMBEDDING_TIMEOUT_SEC`：请求超时秒数，默认 `10`

关闭兜底后，API 将回到“必须由调用方显式提供向量”的模式。

## 2. 能力协商（建议每个调用方启动时先探测）

### `GET /v2/capabilities`

用于判断服务可用能力、检索策略和 schema 版本。

响应示例：

```json
{
  "api_version": "0.2.0",
  "generic_api": true,
  "legacy_relations": true,
  "legacy_ui": true,
  "retrieval_modes": ["legacy_hybrid"],
  "schema_version": 1,
  "features": {
    "relations": true,
    "hybrid": true,
    "fts": true,
    "vector_index": true,
    "scalar_index": true
  }
}
```

字段说明：

- `generic_api`：是否启用 `/v2/*` 路由（若关闭，v2 接口通常会 404）
- `retrieval_modes`：当前支持的检索策略；当前仅 `legacy_hybrid`
- `schema_version`：底层表结构版本号，用于调用方做兼容判断
- `features`：布尔能力开关快照

## 3. 核心模型（v2 语义）

### 3.1 Collection（集合）

- `collection_id`：集合主键（当前实现与 v1 的 `policy_id` 同名映射）
- `dim`：向量维度（首次写入后固定）

### 3.2 Document（文档）

写入模型（`GenericDocumentInput`）：

```json
{
  "document_id": 101,
  "content": "hybrid retrieval with rrf",
  "content_tokenized": "hybrid retrieval with rrf",
  "vector": [0.0, 1.0, 0.0, 0.0],
  "metadata": {
    "kind": "original",
    "directories": ["docs"],
    "parent_chunk_index": -1,
    "derived_seq": 0,
    "relation_keys": [],
    "hop_depth": 0,
    "source": "kb",
    "clause_id": ""
  }
}
```

约束与行为：

- `documents` 在 upsert 请求中最少 1 条（空数组会被 422 拒绝）
- `content_tokenized` 允许空；为空时服务端会回退为简单分词
- `vector` 可为空（纯 BM25）；有值时应保证同一集合维度一致
- 未传 `vector` 时，服务端会按文档 `content` 自动调用 embedding 服务补齐；若 embedding 失败或维度不匹配，会降级到原有行为（已有 `dim` 时补零向量）
- `metadata` 支持任意 JSON 结构（对象/数组/基础类型），写入后在读取与检索结果中原样返回
- `metadata` 中可扁平化为标量的叶子字段会自动落成独立列（列名前缀 `md_`），并自动尝试构建标量索引，用于 `where` 过滤

读取/检索返回模型（`GenericDocumentRecord`）：

```json
{
  "document_id": 101,
  "score": 0.016129,
  "cosine_similarity": 0.982341,
  "bm25_score": 2.147325,
  "content": "hybrid retrieval with rrf",
  "metadata": {
    "heading_paths": [],
    "directories": ["docs"],
    "kind": "original",
    "parent_chunk_index": -1,
    "derived_seq": 0,
    "relation_keys": [],
    "hop_depth": 0,
    "source": "kb",
    "clause_id": ""
  }
}
```

分数字段说明：

- `score`：最终排序分（RRF 融合分）
- `cosine_similarity`：向量召回线的余弦相似度；未命中向量线时为 `null`
- `bm25_score`：BM25/FTS 召回线的原始分；未命中 BM25 线时为 `null`

说明：上面的 `metadata` 字段仅为示例，不代表固定字段集合；调用方可以按业务需要扩展任意字段与嵌套结构。

扁平化列规则（用于 `where`）：

- 列名格式：`md_<path_sanitized>_<hash8>`
- 路径分隔：对象层级用 `__`，数组索引用数字路径
- 示例路径：`metadata.tenant` -> `md_tenant_xxxxxxxx`，`metadata.risk.level` -> `md_risk__level_xxxxxxxx`
- 实际可用列名以 `/v2/collections/{collection_id}/meta` 的 `filterable_fields` 为准

## 4. 接口总览


| 方法     | 路径                                                 | 说明                               |
| ------ | -------------------------------------------------- | -------------------------------- |
| GET    | `/v2/capabilities`                                 | 能力协商                             |
| GET    | `/v2/collections`                                  | 列出集合                             |
| GET    | `/v2/collections/{collection_id}/meta`             | 查询集合元信息                          |
| DELETE | `/v2/collections/{collection_id}`                  | 删除集合                             |
| POST   | `/v2/collectionOverwriteByPrefix`                  | 按前缀覆盖写入（清理同前缀旧集合）               |
| GET    | `/v2/collections/{collection_id}/documents`        | 列表文档                             |
| GET    | `/v2/documents`                                    | 列表文档（别名，query 带 `collection_id`） |
| POST   | `/v2/collections/{collection_id}/documents:upsert` | 写入文档                             |
| POST   | `/v2/documents:upsert`                             | 写入文档（别名，body 带 `collection_id`）  |
| POST   | `/v2/collections/{collection_id}/search`           | 混合检索                             |
| POST   | `/v2/search`                                       | 混合检索（别名，body 带 `collection_id`）  |


## 5. 详细接口说明

### 5.1 列出集合

`GET /v2/collections`

响应：

```json
{
  "collections": [
    {
      "collection_id": "generic_collection_1",
      "n_documents": 2,
      "dim": 4
    }
  ]
}
```

### 5.2 集合元信息

`GET /v2/collections/{collection_id}/meta`

- 不存在集合时返回 `404`

响应：

```json
{
  "collection_id": "generic_collection_1",
  "n_documents": 2,
  "n_original": 1,
  "n_derived": 1,
  "dim": 4,
  "has_vector_index": true,
  "has_fts_index": true,
  "built_at": 1716620000000,
  "schema_version": 1,
  "schema_fields": [],
  "filterable_fields": ["kind", "parent_chunk_index", "md_tenant_a1b2c3d4", "md_risk__level_e5f6a7b8"],
  "searchable_fields": ["content_tokenized", "content", "vector"]
}
```

### 5.3 删除集合

`DELETE /v2/collections/{collection_id}`

响应：

```json
{
  "ok": true
}
```

说明：

- `ok=false` 表示集合不存在（幂等删除）

### 5.4 列表文档

`GET /v2/collections/{collection_id}/documents`

Query 参数：

- `where`：过滤表达式（LanceDB where 语法）；可使用基础字段（如 `kind`）和扁平化后的 `md_*` 字段
- `limit`：单页返回条数，默认 `1000`，范围 `1..100000`
- `offset`：跳过的条数（用于分页），默认 `0`，范围 `>= 0`
- `include_content`：默认 `false`

别名接口：

- `GET /v2/documents?collection_id=...`

分页说明：

- 接口本身不返回总数，按 `limit + offset` 方式翻页：第 N 页（页码从 0 起）传 `offset = N * limit`。
- 当某次返回的 `documents` 数量 **小于 `limit`** 时，说明已到最后一页，停止翻页。
- 需要拉取 collection 全量文档时，循环调用直到满足上述停止条件即可；不要依赖单次大 `limit` 一把梭，量大时会有内存与超时风险。
- 集合的文档总数可通过 `GET /v2/collections/{collection_id}/meta` 的 `n_documents` 字段获取，便于客户端预估页数。

分页调用示例（拉取全量）：

```text
# page_size = 1000
GET /v2/collections/{cid}/documents?include_content=false&limit=1000&offset=0
GET /v2/collections/{cid}/documents?include_content=false&limit=1000&offset=1000
GET /v2/collections/{cid}/documents?include_content=false&limit=1000&offset=2000
... 直到返回数量 < 1000 为止
```

响应：

```json
{
  "documents": [
    {
      "document_id": 101,
      "score": 0.0,
      "cosine_similarity": null,
      "bm25_score": null,
      "content": "vector database general platform",
      "metadata": {
        "heading_paths": [],
        "directories": ["docs"],
        "kind": "original",
        "parent_chunk_index": -1,
        "derived_seq": 0,
        "relation_keys": [],
        "hop_depth": 0,
        "source": "",
        "clause_id": ""
      }
    }
  ]
}
```

### 5.5 文档写入（upsert）

`POST /v2/collections/{collection_id}/documents:upsert`

请求体：

```json
{
  "documents": [
    {
      "document_id": 101,
      "content": "vector database general platform",
      "content_tokenized": "vector database general platform",
      "vector": [0.1, 0.2, 0.3, 0.4],
      "metadata": {
        "kind": "original",
        "directories": ["docs"],
        "hop_depth": 0
      }
    }
  ],
  "mode": "overwrite",
  "expected_dim": 4
}
```

`mode` 可选值：

- `overwrite`：覆盖写
- `append`：追加写
- `merge_by_chunk_id`：按主键（`document_id/chunk_id`）合并；对已存在文档执行字段级安全合并，
  空 `content`/`content_tokenized`/`vector` 不会覆盖旧值，适合 `attempts`、`tombstone` 这类局部 metadata 更新

别名接口：

- `POST /v2/documents:upsert`
- 仅差异：`collection_id` 放在 body 内

成功响应：

```json
{
  "written": 1,
  "table_size": 1,
  "dim": 4
}
```

常见失败：

- `400`：维度不一致、未知 upsert 模式等业务参数错误
- `422`：请求体字段校验失败（例如 `documents` 为空）

### 5.6 按前缀覆盖写入（collectionOverwriteByPrefix）

`POST /v2/collectionOverwriteByPrefix`

行为说明：

- 先取 `collection_id.split("_")[0]` 作为前缀
- 删除当前库中所有“前缀相同且 `collection_id` 不同”的集合
- 再用 `overwrite` 模式写入当前 `collection_id`（与创建新 collection 的写入链路一致）

请求体：

```json
{
  "collection_id": "invoice_202407",
  "documents": [
    {
      "document_id": 1,
      "content": "latest invoice",
      "content_tokenized": "latest invoice",
      "vector": [0.1, 0.2, 0.3, 0.4],
      "metadata": {"kind": "original"}
    }
  ],
  "expected_dim": 4
}
```

成功响应：

```json
{
  "written": 1,
  "table_size": 1,
  "dim": 4,
  "dropped_collections": ["invoice_202405", "invoice_202406"]
}
```

常见失败：

- `400`：维度不一致等业务参数错误
- `422`：请求体字段校验失败（例如 `documents` 为空）
- `500`：删除或写入阶段服务内部错误

### 5.7 混合检索

`POST /v2/collections/{collection_id}/search`

请求体：

```json
{
  "query_tokenized": "hybrid retrieval",
  "query_vector": [0.0, 1.0, 0.0, 0.0],
  "top_n": 5,
  "top_m": 5,
  "rrf_k": 60,
  "where": "kind = 'original'",
  "include_content": true,
  "include_derived": true,
  "strategy": "legacy_hybrid"
}
```

字段说明：

- `query_tokenized`：BM25 查询词（空串表示不走 FTS 召回）
- `query_vector`：向量查询（空数组表示不走向量召回）
- 未传 `query_vector` 时，服务端会基于 `query_tokenized` 自动调用 embedding 服务；若 embedding 失败或维度不匹配，会降级为 BM25/FTS-only
- `top_n`：向量召回上限
- `top_m`：BM25 召回上限
- `rrf_k`：RRF 常数；不传时使用服务端配置值
- `include_derived=false`：会在 where 上附加 `kind='original'`
- 需要按 metadata 过滤时，建议先调用 `GET /v2/collections/{collection_id}/meta` 获取最新 `filterable_fields` 中的 `md_*` 列名
- `strategy`：当前仅支持 `legacy_hybrid`
- 返回 `hits[*]` 中的 `score` 为最终 RRF 融合分；`cosine_similarity` / `bm25_score` 分别是向量线与 BM25 线的原始分（某条线未命中时为 `null`）

纯向量检索（vector-only）调用方法：

- `query_vector`：传非空向量（维度需与集合 `dim` 一致）
- `query_tokenized`：传空串 `""`（避免走 BM25 召回）
- `top_n`：传正整数（例如 `5`）
- `top_m`：传 `0`（显式关闭 BM25 召回）
- `strategy`：固定 `legacy_hybrid`

纯向量检索请求示例：

```json
{
  "query_tokenized": "",
  "query_vector": [0.0, 1.0, 0.0, 0.0],
  "top_n": 5,
  "top_m": 0,
  "include_content": true,
  "include_derived": true,
  "strategy": "legacy_hybrid"
}
```

别名接口：

- `POST /v2/search`
- 仅差异：`collection_id` 放在 body 内

成功响应：

```json
{
  "hits": [
    {
      "document_id": 102,
      "score": 0.016129,
      "cosine_similarity": 1.0,
      "bm25_score": 1.386294,
      "content": "hybrid retrieval with rrf",
      "metadata": {
        "heading_paths": [],
        "directories": [],
        "kind": "derived",
        "parent_chunk_index": 101,
        "derived_seq": 0,
        "relation_keys": [],
        "hop_depth": 1,
        "source": "",
        "clause_id": ""
      }
    }
  ]
}
```

纯向量检索成功响应示例：

```json
{
  "hits": [
    {
      "document_id": 102,
      "score": 0.016393,
      "cosine_similarity": 1.0,
      "bm25_score": null,
      "content": "hybrid retrieval with rrf",
      "metadata": {
        "kind": "derived",
        "parent_chunk_index": 101,
        "hop_depth": 1
      }
    }
  ]
}
```

常见失败：

- `404`：集合不存在或未建索引（`collection not indexed`）
- `400`：策略不支持等业务参数错误
- `422`：字段类型不合法

## 6. 错误语义（调用方建议）

### 6.1 HTTP 状态码

- `200`：成功
- `400`：业务参数错误（建议调用方直接记录并告警）
- `401`：鉴权失败（`invalid api key`）
- `404`：资源不存在或集合未建立
- `422`：请求参数/请求体校验失败
- `500`：服务内部错误（可重试 + 告警）

### 6.2 统一错误体

FastAPI 默认格式：

```json
{
  "detail": "error message"
}
```

或校验错误结构：

```json
{
  "detail": [
    {
      "loc": ["body", "documents", 0, "document_id"],
      "msg": "Field required",
      "type": "missing"
    }
  ]
}
```

## 7. 推荐接入流程（服务对服务）

1. 启动时调用 `/v2/capabilities` 做能力协商（确认 `generic_api=true`）。
2. 首次写入某集合时显式传 `expected_dim`，固定向量维度；若依赖自动兜底，确保 embedding 配置可用。
3. 写入后调用 `/v2/collections/{id}/meta` 校验 `dim`、索引状态与文档量。
4. 在线检索优先走 `/v2/collections/{id}/search`，按需调 `top_n/top_m`。
5. 对 `400/401/404/422/500` 分层处理并打点监控。

## 8. curl 示例

```bash
BASE_URL="http://127.0.0.1:5000"
API_KEY="changeme"
CID="generic_collection_1"

curl -sS "${BASE_URL}/v2/capabilities" \
  -H "X-API-Key: ${API_KEY}" | jq .

curl -sS "${BASE_URL}/v2/collections/${CID}/documents:upsert" \
  -H "X-API-Key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "documents":[
      {
        "document_id":101,
        "content":"vector database general platform",
        "content_tokenized":"vector database general platform",
        "vector":[0.1,0.2,0.3,0.4],
        "metadata":{"kind":"original","directories":["docs"],"hop_depth":0}
      }
    ],
    "mode":"overwrite",
    "expected_dim":4
  }' | jq .

curl -sS "${BASE_URL}/v2/collections/${CID}/search" \
  -H "X-API-Key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_tokenized":"vector database",
    "query_vector":[0.1,0.2,0.3,0.4],
    "top_n":5,
    "top_m":5,
    "include_content":true,
    "strategy":"legacy_hybrid"
  }' | jq .

# 纯向量检索（vector-only）
curl -sS "${BASE_URL}/v2/collections/${CID}/search" \
  -H "X-API-Key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_tokenized":"",
    "query_vector":[0.1,0.2,0.3,0.4],
    "top_n":5,
    "top_m":0,
    "include_content":true,
    "strategy":"legacy_hybrid"
  }' | jq .
```

## 9. 最小 SDK 示例（Python / httpx）

适用场景：其他 Python 服务以最小成本对接 v2（能力探测 + upsert + search）。

依赖安装：

```bash
pip install httpx
```

示例代码（可直接复制为 `v2_client.py`）：

```python
from __future__ import annotations

from typing import Any

import httpx


class RetrievalV2Client:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers: dict[str, str] = {}
        if api_key:
            headers["X-API-Key"] = api_key
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs) -> Any:
        resp = self._client.request(method, path, **kwargs)
        # 统一把 4xx/5xx 转成异常，调用方可在上层统一告警/重试
        resp.raise_for_status()
        if not resp.content:
            return None
        return resp.json()

    def capabilities(self) -> dict[str, Any]:
        return self._request("GET", "/v2/capabilities")

    def upsert_documents(
        self,
        collection_id: str,
        documents: list[dict[str, Any]],
        mode: str = "overwrite",
        expected_dim: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "documents": documents,
            "mode": mode,
        }
        if expected_dim is not None:
            payload["expected_dim"] = expected_dim
        return self._request(
            "POST",
            f"/v2/collections/{collection_id}/documents:upsert",
            json=payload,
        )

    def search(
        self,
        collection_id: str,
        *,
        query_tokenized: str = "",
        query_vector: list[float] | None = None,
        top_n: int = 20,
        top_m: int = 20,
        where: str | None = None,
        include_content: bool = True,
        include_derived: bool = True,
        strategy: str = "legacy_hybrid",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query_tokenized": query_tokenized,
            "query_vector": query_vector or [],
            "top_n": top_n,
            "top_m": top_m,
            "where": where,
            "include_content": include_content,
            "include_derived": include_derived,
            "strategy": strategy,
        }
        return self._request("POST", f"/v2/collections/{collection_id}/search", json=payload)


if __name__ == "__main__":
    client = RetrievalV2Client(base_url="http://127.0.0.1:5000", api_key="changeme")
    try:
        caps = client.capabilities()
        print("capabilities:", caps)

        assert caps.get("generic_api") is True, "服务未启用 generic v2 API"
        assert "legacy_hybrid" in caps.get("retrieval_modes", []), "服务未启用 legacy_hybrid"

        collection_id = "generic_collection_1"
        docs = [
            {
                "document_id": 101,
                "content": "vector database general platform",
                "content_tokenized": "vector database general platform",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "metadata": {"kind": "original", "directories": ["docs"], "hop_depth": 0},
            }
        ]
        upsert_resp = client.upsert_documents(collection_id, docs, mode="overwrite", expected_dim=4)
        print("upsert:", upsert_resp)

        search_resp = client.search(
            collection_id,
            query_tokenized="vector database",
            query_vector=[0.1, 0.2, 0.3, 0.4],
            top_n=5,
            top_m=5,
            include_content=True,
        )
        print("search hits:", search_resp.get("hits", []))
    finally:
        client.close()
```

调用方接入建议：

- 把 `capabilities()` 放到进程启动自检，避免运行期才发现能力不匹配。
- 首次 upsert 强制传 `expected_dim`，可以更早暴露 embedding 维度错误。
- 对 `httpx.HTTPStatusError` 按状态码分类处理（尤其 `401/404/422/500`）。


"""LanceDB 表 schema + 入参/出参 pydantic 模型。

设计要点：

- 一个 ``policy_id`` 一张表（路径 ``STORE_DIR/{safe_policy_id}.lance``）；
- 列组合覆盖原 ``KnowledgeChunk``（content / heading_paths / directories）+ 关联结构
  （kind / parent_chunk_index / derived_seq / relation_keys / hop_depth / source / clause_id）；
- ``vector`` 列为 ``fixed_size_list<float32>[dim]``，``dim`` 由首次 upsert 时的 vector 长度决定；
- ``content_tokenized`` 列由**客户端 jieba 分词后空格连接**，服务端 FTS 走 whitespace tokenizer，
  与主项目 ``inference/retrieval/bm25.py::tokenize`` 保持完全同源（服务端不依赖 jieba）。
"""

from __future__ import annotations

from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------- pyarrow schema


_RELATION_KEY_TYPE = pa.struct([
    pa.field("policy_id", pa.string()),
    pa.field("clause_id", pa.string()),
])


def build_arrow_schema(dim: int) -> pa.Schema:
    """根据 embedding 维度生成 pyarrow schema。

    ``dim <= 0`` 表示尚不知向量长度（创建空表保留位）；这种情况下 ``vector`` 列退化为
    可空 list<float32>，第一次有真实向量写入时会强制 reshape。实际生产路径下 ``dim`` 应在
    首次 upsert 时由 client 给出（vector 长度），服务端推断后写入 meta 锁定。
    """

    if dim and dim > 0:
        vec_type = pa.list_(pa.float32(), dim)
    else:
        vec_type = pa.list_(pa.float32())

    return pa.schema([
        pa.field("chunk_id", pa.int64(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
        pa.field("content_tokenized", pa.string(), nullable=False),
        pa.field("vector", vec_type, nullable=True),
        pa.field("heading_paths", pa.list_(pa.list_(pa.string()))),
        pa.field("directories", pa.list_(pa.string())),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("parent_chunk_index", pa.int64(), nullable=False),
        pa.field("derived_seq", pa.int32(), nullable=False),
        pa.field("relation_keys", pa.list_(_RELATION_KEY_TYPE)),
        pa.field("hop_depth", pa.int32(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("clause_id", pa.string(), nullable=False),
        pa.field("built_at", pa.int64(), nullable=False),
    ])


# ---------------------------------------------------------------- pydantic models


class RelationKey(BaseModel):
    policy_id: str = ""
    clause_id: str = ""


class ChunkRow(BaseModel):
    """与 pyarrow schema 一一对齐的写入行。"""

    chunk_id: int
    content: str
    content_tokenized: str
    vector: list[float] = Field(default_factory=list)
    heading_paths: list[list[str]] = Field(default_factory=list)
    directories: list[str] = Field(default_factory=list)
    kind: Literal["original", "derived"] = "original"
    parent_chunk_index: int = -1
    derived_seq: int = 0
    relation_keys: list[RelationKey] = Field(default_factory=list)
    hop_depth: int = 0
    source: str = ""
    clause_id: str = ""
    built_at: int = 0

    @field_validator("kind")
    @classmethod
    def _kind_lower(cls, v: str) -> str:
        return (v or "original").lower()


UpsertMode = Literal["overwrite", "append", "merge_by_chunk_id"]


class UpsertRequest(BaseModel):
    chunks: list[ChunkRow]
    mode: UpsertMode = "overwrite"
    expected_dim: int | None = None  # 客户端可选地校验维度


class UpsertResponse(BaseModel):
    written: int
    table_size: int
    dim: int


class SearchRequest(BaseModel):
    query_tokenized: str = ""
    query_vector: list[float] = Field(default_factory=list)
    top_n: int = 20  # 向量召回上限
    top_m: int = 20  # BM25 召回上限
    rrf_k: int | None = None
    where: str | None = None
    include_content: bool = True
    # 是否把派生 chunk 也参与召回；False 时强制 where=kind='original'
    include_derived: bool = True


class SearchHit(BaseModel):
    chunk_id: int
    score: float
    content: str | None = None
    heading_paths: list[list[str]] = Field(default_factory=list)
    directories: list[str] = Field(default_factory=list)
    kind: str = "original"
    parent_chunk_index: int = -1
    derived_seq: int = 0
    relation_keys: list[RelationKey] = Field(default_factory=list)
    hop_depth: int = 0
    source: str = ""
    clause_id: str = ""


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class ExpandRequest(BaseModel):
    chunk_id: int
    include_content: bool = True


class ExpandResponse(BaseModel):
    chunks: list[SearchHit]


class LookupResponse(BaseModel):
    """单 policy 内反查：返回引用了 (target_policy_id[, target_clause_id]) 的 chunks。"""

    chunks: list[SearchHit]


class DependentEntry(BaseModel):
    source_policy_id: str
    n_hits: int


class DependentsResponse(BaseModel):
    dependents: list[DependentEntry]


class TableMeta(BaseModel):
    policy_id: str
    n_chunks: int
    n_original: int
    n_derived: int
    dim: int
    has_vector_index: bool
    has_fts_index: bool
    built_at: int
    schema_version: int


class PolicyListItem(BaseModel):
    policy_id: str
    n_chunks: int
    dim: int


class PolicyListResponse(BaseModel):
    policies: list[PolicyListItem]

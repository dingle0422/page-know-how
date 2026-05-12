"""LanceDB 连接、表生命周期、索引管理。

每张表对应一个 ``policy_id``，物理路径 ``STORE_DIR/{safe_policy_id}.lance``。

为了防止 ``policy_id`` 含中文 / 路径分隔符等不安全字符破坏文件系统，统一通过
:func:`_safe_policy_dir` 做 urlsafe-base64 编码后作为目录名；HTTP 接口里仍接受
原始 ``policy_id``。

LanceDB 操作大多是同步阻塞的；外层路由用 :func:`anyio.to_thread.run_sync` 调用本模块函数。
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from typing import Any

import pyarrow as pa

from .config import get_settings
from .schema import (
    ChunkRow,
    RelationKey,
    SearchHit,
    build_arrow_schema,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# 同进程内共享：lancedb.connect 是廉价的（懒打开），但 Table 句柄保留可减少重复 open。
_db_lock = threading.Lock()
_db: Any | None = None
_table_cache: dict[str, Any] = {}
_table_lock = threading.Lock()


def _safe_policy_dir(policy_id: str) -> str:
    """把任意 policy_id 编码为安全的目录名。"""

    raw = (policy_id or "").encode("utf-8")
    enc = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"p_{enc}"


def _decode_policy_id(safe_name: str) -> str:
    if not safe_name.startswith("p_"):
        return ""
    body = safe_name[2:]
    pad = "=" * (-len(body) % 4)
    try:
        return base64.urlsafe_b64decode(body + pad).decode("utf-8")
    except Exception:
        return ""


def _connect():
    import lancedb  # type: ignore

    settings = get_settings()
    os.makedirs(settings.store_dir, exist_ok=True)

    global _db
    with _db_lock:
        if _db is None:
            _db = lancedb.connect(settings.store_dir)
        return _db


def _table_path(policy_id: str) -> str:
    settings = get_settings()
    return os.path.join(settings.store_dir, f"{_safe_policy_dir(policy_id)}.lance")


def table_exists(policy_id: str) -> bool:
    db = _connect()
    return _safe_policy_dir(policy_id) in db.table_names()


def open_table(policy_id: str):
    """打开已有表；不存在抛 ``KeyError``。"""

    if not table_exists(policy_id):
        raise KeyError(policy_id)
    name = _safe_policy_dir(policy_id)
    with _table_lock:
        cached = _table_cache.get(policy_id)
        if cached is not None:
            return cached
        db = _connect()
        tbl = db.open_table(name)
        _table_cache[policy_id] = tbl
        return tbl


def drop_table(policy_id: str) -> bool:
    db = _connect()
    name = _safe_policy_dir(policy_id)
    if name not in db.table_names():
        return False
    db.drop_table(name)
    with _table_lock:
        _table_cache.pop(policy_id, None)
    return True


def list_policies() -> list[tuple[str, int, int]]:
    """返回 ``[(policy_id, n_chunks, dim), ...]``。"""

    db = _connect()
    out: list[tuple[str, int, int]] = []
    for safe_name in db.table_names():
        pid = _decode_policy_id(safe_name)
        if not pid:
            continue
        try:
            tbl = db.open_table(safe_name)
            n = tbl.count_rows()
            dim = _detect_dim(tbl)
        except Exception as e:
            logger.warning("[Store] 读取表 %s 失败: %s", safe_name, e)
            continue
        out.append((pid, int(n), int(dim)))
    return out


# ---------------------------------------------------------------- 数据序列化


def _row_to_arrow_dict(row: ChunkRow) -> dict[str, Any]:
    return {
        "chunk_id": int(row.chunk_id),
        "content": row.content or "",
        "content_tokenized": row.content_tokenized or "",
        "vector": list(row.vector or []),
        "heading_paths": [list(seg) for seg in (row.heading_paths or [])],
        "directories": list(row.directories or []),
        "kind": row.kind or "original",
        "parent_chunk_index": int(row.parent_chunk_index),
        "derived_seq": int(row.derived_seq),
        "relation_keys": [
            {"policy_id": k.policy_id or "", "clause_id": k.clause_id or ""}
            for k in (row.relation_keys or [])
        ],
        "hop_depth": int(row.hop_depth),
        "source": row.source or "",
        "clause_id": row.clause_id or "",
        "built_at": int(row.built_at) or int(time.time() * 1000),
    }


def _build_record_batch(rows: list[ChunkRow], dim: int) -> pa.RecordBatch:
    schema = build_arrow_schema(dim)
    arrow_dicts = [_row_to_arrow_dict(r) for r in rows]
    # 强制每条 vector 长度一致到 dim
    if dim > 0:
        for d in arrow_dicts:
            v = d["vector"]
            if not v:
                # 占位零向量，避免 fixed_size_list 长度校验失败
                d["vector"] = [0.0] * dim
            elif len(v) != dim:
                raise ValueError(
                    f"vector dim mismatch: chunk_id={d['chunk_id']} got={len(v)} expect={dim}"
                )
    return pa.RecordBatch.from_pylist(arrow_dicts, schema=schema)


# ---------------------------------------------------------------- 索引


def _detect_dim(tbl) -> int:
    """从表 schema 反推 vector 列的 fixed_size_list 长度，未建/可变长返回 0。"""

    try:
        schema: pa.Schema = tbl.schema
    except Exception:
        return 0
    field = schema.field("vector") if "vector" in schema.names else None
    if field is None:
        return 0
    t = field.type
    if pa.types.is_fixed_size_list(t):
        return int(t.list_size)
    return 0


def _has_index_on(tbl, column: str) -> bool:
    try:
        for idx in tbl.list_indices():
            cols = getattr(idx, "columns", None) or getattr(idx, "fields", None) or []
            if isinstance(cols, str):
                cols = [cols]
            if column in (cols or []):
                return True
    except Exception:
        return False
    return False


def ensure_indexes(tbl) -> dict[str, bool]:
    """在表上建好 FTS / 向量 / 标量索引（已存在则跳过）。"""

    settings = get_settings()
    n = tbl.count_rows()
    fts_ok = vec_ok = scalar_ok = False

    if n == 0:
        return {"fts": False, "vector": False, "scalar": False}

    # FTS 索引：whitespace tokenizer，因为 content_tokenized 已经是客户端 jieba 分词后的空格串
    if not _has_index_on(tbl, "content_tokenized"):
        try:
            tbl.create_fts_index(
                "content_tokenized",
                base_tokenizer="whitespace",
                with_position=False,
                replace=True,
            )
            fts_ok = True
        except Exception as e:
            logger.warning("[Store] 建 FTS 索引失败: %s", e)
    else:
        fts_ok = True

    # 向量索引：仅在 dim>0 且行数足够时建（小表全量扫足够）
    dim = _detect_dim(tbl)
    if dim > 0 and n >= 256 and not _has_index_on(tbl, "vector"):
        try:
            tbl.create_index(metric="cosine", vector_column_name="vector", replace=True)
            vec_ok = True
        except Exception as e:
            logger.warning("[Store] 建向量索引失败: %s", e)
    else:
        vec_ok = dim > 0

    # 标量索引（可关）
    if settings.enable_scalar_index:
        for col in ("kind", "parent_chunk_index"):
            try:
                tbl.create_scalar_index(col, replace=True)
            except Exception as e:
                logger.debug("[Store] 建标量索引 %s 失败（忽略）: %s", col, e)
        scalar_ok = True

    return {"fts": fts_ok, "vector": vec_ok, "scalar": scalar_ok}


# ---------------------------------------------------------------- 写入


def _infer_dim(rows: list[ChunkRow], expected: int | None) -> int:
    if expected:
        return int(expected)
    for r in rows:
        if r.vector:
            return len(r.vector)
    return 0


def upsert(policy_id: str, rows: list[ChunkRow], mode: str, expected_dim: int | None) -> dict:
    """单表 upsert。返回 ``{"written", "table_size", "dim"}``。"""

    if not rows:
        return {"written": 0, "table_size": _row_count(policy_id), "dim": _existing_dim(policy_id)}

    db = _connect()
    name = _safe_policy_dir(policy_id)

    incoming_dim = _infer_dim(rows, expected_dim)
    existing_dim = _existing_dim(policy_id)
    if existing_dim and incoming_dim and existing_dim != incoming_dim:
        raise ValueError(
            f"dim mismatch for policy={policy_id}: existing={existing_dim} incoming={incoming_dim}"
        )
    dim = incoming_dim or existing_dim
    batch = _build_record_batch(rows, dim)

    if mode == "overwrite" or name not in db.table_names():
        # overwrite 或不存在：删旧建新
        if name in db.table_names():
            db.drop_table(name)
            with _table_lock:
                _table_cache.pop(policy_id, None)
        tbl = db.create_table(name, data=batch, schema=batch.schema)
    else:
        tbl = db.open_table(name)
        if mode == "append":
            tbl.add(batch)
        elif mode == "merge_by_chunk_id":
            try:
                (
                    tbl.merge_insert("chunk_id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(batch)
                )
            except Exception as e:
                # LanceDB 老版本可能不支持 merge_insert，退化为 delete + add
                logger.info("[Store] merge_insert 不可用，退化为 delete+add: %s", e)
                ids = [str(r.chunk_id) for r in rows]
                tbl.delete(f"chunk_id IN ({','.join(ids)})")
                tbl.add(batch)
        else:
            raise ValueError(f"unknown upsert mode: {mode}")

    with _table_lock:
        _table_cache[policy_id] = tbl

    ensure_indexes(tbl)
    return {
        "written": len(rows),
        "table_size": int(tbl.count_rows()),
        "dim": int(dim),
    }


# ---------------------------------------------------------------- 读取助手


def _row_count(policy_id: str) -> int:
    if not table_exists(policy_id):
        return 0
    return int(open_table(policy_id).count_rows())


def _existing_dim(policy_id: str) -> int:
    if not table_exists(policy_id):
        return 0
    return _detect_dim(open_table(policy_id))


# ---------------------------------------------------------------- 检索


def _row_to_hit(row: dict, *, include_content: bool) -> SearchHit:
    rks = row.get("relation_keys") or []
    return SearchHit(
        chunk_id=int(row["chunk_id"]),
        score=float(row.get("_score", 0.0)),
        content=row.get("content") if include_content else None,
        heading_paths=[list(p) for p in (row.get("heading_paths") or [])],
        directories=list(row.get("directories") or []),
        kind=row.get("kind") or "original",
        parent_chunk_index=int(row.get("parent_chunk_index", -1)),
        derived_seq=int(row.get("derived_seq", 0)),
        relation_keys=[
            RelationKey(policy_id=rk.get("policy_id", ""), clause_id=rk.get("clause_id", ""))
            for rk in rks
        ],
        hop_depth=int(row.get("hop_depth", 0)),
        source=row.get("source") or "",
        clause_id=row.get("clause_id") or "",
    )


def _select_columns(include_content: bool) -> list[str]:
    cols = [
        "chunk_id",
        "heading_paths",
        "directories",
        "kind",
        "parent_chunk_index",
        "derived_seq",
        "relation_keys",
        "hop_depth",
        "source",
        "clause_id",
    ]
    if include_content:
        cols.append("content")
    return cols


def _safe_search_to_list(query, top_k: int) -> list[dict]:
    if top_k <= 0:
        return []
    try:
        return query.limit(top_k).to_list()
    except Exception as e:
        logger.warning("[Store] 检索失败: %s", e)
        return []


def hybrid_search(
    policy_id: str,
    *,
    query_tokenized: str,
    query_vector: list[float],
    top_n: int,
    top_m: int,
    rrf_k: int,
    where: str | None,
    include_content: bool,
    include_derived: bool,
) -> list[SearchHit]:
    """BM25 + 向量并行召回，本地 RRF 融合，与主项目 ``inference/retrieval/rrf.py`` 同公式。"""

    if not table_exists(policy_id):
        return []
    tbl = open_table(policy_id)
    n_total = tbl.count_rows()
    if n_total == 0:
        return []

    cols = _select_columns(include_content)
    final_where = where
    if not include_derived:
        cond = "kind = 'original'"
        final_where = f"({where}) AND ({cond})" if where else cond

    # FTS 路径
    fts_pairs: list[tuple[int, float]] = []
    if query_tokenized.strip() and top_m > 0:
        q = (
            tbl.search(query_tokenized, query_type="fts", fts_columns="content_tokenized")
            .select(cols)
        )
        if final_where:
            q = q.where(final_where, prefilter=False)
        for row in _safe_search_to_list(q, top_m):
            fts_pairs.append((int(row["chunk_id"]), float(row.get("_score", 0.0))))

    # 向量路径
    vec_pairs: list[tuple[int, float]] = []
    if query_vector and top_n > 0:
        q = tbl.search(query_vector, vector_column_name="vector").select(cols)
        if final_where:
            q = q.where(final_where, prefilter=True)
        for row in _safe_search_to_list(q, top_n):
            # LanceDB 向量搜索返回 _distance（越小越相似）；转成相似度分数仅用于排序展示
            d = float(row.get("_distance", row.get("_score", 0.0)))
            vec_pairs.append((int(row["chunk_id"]), -d))

    if not fts_pairs and not vec_pairs:
        return []

    # RRF 融合：rrf = sum(1 / (k + rank))，rank 从 1 起
    fused: dict[int, float] = {}
    for rank_list in (fts_pairs, vec_pairs):
        for rank, (cid, _s) in enumerate(rank_list, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    fused_sorted = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    keep_ids = [cid for cid, _ in fused_sorted]

    # 一次性把 chunk 内容拉回（避免逐行 select）
    if not keep_ids:
        return []
    id_filter = "chunk_id IN (" + ",".join(str(i) for i in keep_ids) + ")"
    rows = tbl.search().select(cols).where(id_filter).limit(len(keep_ids)).to_list()
    by_id = {int(r["chunk_id"]): r for r in rows}

    hits: list[SearchHit] = []
    for cid, score in fused_sorted:
        row = by_id.get(cid)
        if row is None:
            continue
        row["_score"] = score
        hits.append(_row_to_hit(row, include_content=include_content))
    return hits


def list_chunks(
    policy_id: str,
    *,
    where: str | None,
    limit: int,
    include_content: bool,
) -> list[SearchHit]:
    if not table_exists(policy_id):
        return []
    tbl = open_table(policy_id)
    cols = _select_columns(include_content)
    q = tbl.search().select(cols)
    if where:
        q = q.where(where)
    rows = q.limit(max(limit, 1)).to_list()
    return [_row_to_hit(r, include_content=include_content) for r in rows]


def expand_relations(policy_id: str, chunk_id: int, *, include_content: bool) -> list[SearchHit]:
    """返回某父 chunk 的派生 chunks（``parent_chunk_index = chunk_id and kind='derived'``）。"""

    if not table_exists(policy_id):
        return []
    tbl = open_table(policy_id)
    cols = _select_columns(include_content)
    rows = (
        tbl.search()
        .select(cols)
        .where(f"parent_chunk_index = {int(chunk_id)} AND kind = 'derived'")
        .limit(10000)
        .to_list()
    )
    return [_row_to_hit(r, include_content=include_content) for r in rows]


def lookup_relations(
    policy_id: str,
    *,
    target_policy_id: str,
    target_clause_id: str | None,
    include_content: bool,
) -> list[SearchHit]:
    """单表内反查：列出 ``relation_keys`` 中含 ``target_*`` 的派生 chunks。"""

    if not table_exists(policy_id):
        return []
    tbl = open_table(policy_id)
    cols = _select_columns(include_content)
    # 用 list_has_struct 的能力：LanceDB SQL 支持 ``array_has(relation_keys, struct_value)``，
    # 但跨版本不稳定。回退到 Python 侧过滤，性能可接受（派生 chunks 一般 < 1k）。
    rows = (
        tbl.search()
        .select(cols + ["relation_keys"] if "relation_keys" not in cols else cols)
        .where("kind = 'derived'")
        .limit(100000)
        .to_list()
    )
    out: list[SearchHit] = []
    for r in rows:
        rks = r.get("relation_keys") or []
        for rk in rks:
            if rk.get("policy_id") != target_policy_id:
                continue
            if target_clause_id and rk.get("clause_id") != target_clause_id:
                continue
            out.append(_row_to_hit(r, include_content=include_content))
            break
    return out


def lookup_dependents(target_policy_id: str, target_clause_id: str | None) -> list[tuple[str, int]]:
    """全局反查：返回 ``[(source_policy_id, n_hits), ...]``，用于 cascade 触发。"""

    out: list[tuple[str, int]] = []
    for pid, _n, _dim in list_policies():
        if pid == target_policy_id:
            continue  # 自反不计
        hits = lookup_relations(
            pid,
            target_policy_id=target_policy_id,
            target_clause_id=target_clause_id,
            include_content=False,
        )
        if hits:
            out.append((pid, len(hits)))
    return out


# ---------------------------------------------------------------- meta


def table_meta(policy_id: str) -> dict:
    if not table_exists(policy_id):
        return {}
    tbl = open_table(policy_id)
    n = int(tbl.count_rows())
    dim = _detect_dim(tbl)
    n_orig = 0
    n_derv = 0
    built_at = 0
    try:
        for col_row in (
            tbl.search().select(["kind", "built_at"]).limit(n or 1).to_list()
        ):
            if col_row.get("kind") == "derived":
                n_derv += 1
            else:
                n_orig += 1
            ba = int(col_row.get("built_at") or 0)
            if ba > built_at:
                built_at = ba
    except Exception as e:
        logger.warning("[Store] 统计 meta 失败: %s", e)
    return {
        "policy_id": policy_id,
        "n_chunks": n,
        "n_original": n_orig,
        "n_derived": n_derv,
        "dim": dim,
        "has_vector_index": _has_index_on(tbl, "vector"),
        "has_fts_index": _has_index_on(tbl, "content_tokenized"),
        "built_at": built_at,
        "schema_version": SCHEMA_VERSION,
    }

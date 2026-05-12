"""端到端冒烟：用临时 STORE_DIR + TestClient 跑一轮 upsert / search / expand / drop。"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="retrieval_test_")
    monkeypatch.setenv("STORE_DIR", tmp)
    monkeypatch.setenv("API_KEY", "")  # 关闭鉴权方便测试
    monkeypatch.setenv("ENABLE_SCALAR_INDEX", "0")  # 极小表跳过标量索引省时间

    # config 和 store 都用 lru_cache / 模块级单例缓存，需要在 monkeypatch 后重新导入
    from importlib import reload

    from app import config as cfg_mod
    from app import store as store_mod
    from app import main as main_mod

    reload(cfg_mod)
    reload(store_mod)
    reload(main_mod)

    with TestClient(main_mod.app) as c:
        yield c


def _make_chunks() -> list[dict]:
    """三个 chunk：1 原始 + 2 派生。dim=4 简化测试。"""
    return [
        {
            "chunk_id": 1,
            "content": "农产品自产自销 免税 增值税",
            "content_tokenized": "农产品 自产自销 免税 增值税",
            "vector": [1.0, 0.0, 0.0, 0.0],
            "heading_paths": [["2_涉税处理", "2.1_增值税"]],
            "directories": ["/k/2_涉税处理/2.1_增值税"],
            "kind": "original",
            "parent_chunk_index": -1,
            "derived_seq": 0,
            "relation_keys": [],
            "hop_depth": 0,
            "source": "",
            "clause_id": "",
            "built_at": 1700000000000,
        },
        {
            "chunk_id": 2,
            "content": "蔬菜主要品种目录 萝卜 胡萝卜 茄子",
            "content_tokenized": "蔬菜 主要 品种 目录 萝卜 胡萝卜 茄子",
            "vector": [0.0, 1.0, 0.0, 0.0],
            "heading_paths": [["附件", "蔬菜主要品种目录"]],
            "directories": ["/k/附件/蔬菜主要品种目录"],
            "kind": "derived",
            "parent_chunk_index": 1,
            "derived_seq": 1,
            "relation_keys": [{"policy_id": "OTHER_POL", "clause_id": "C-001"}],
            "hop_depth": 1,
            "source": "local",
            "clause_id": "C-001",
            "built_at": 1700000000000,
        },
        {
            "chunk_id": 3,
            "content": "鲜活肉蛋 流通环节 免税",
            "content_tokenized": "鲜活 肉蛋 流通 环节 免税",
            "vector": [0.0, 0.0, 1.0, 0.0],
            "heading_paths": [["附件", "鲜活肉蛋"]],
            "directories": ["/k/附件/鲜活肉蛋"],
            "kind": "derived",
            "parent_chunk_index": 1,
            "derived_seq": 2,
            "relation_keys": [{"policy_id": "OTHER_POL", "clause_id": "C-002"}],
            "hop_depth": 1,
            "source": "local",
            "clause_id": "C-002",
            "built_at": 1700000000000,
        },
    ]


def test_full_lifecycle(client: TestClient):
    pid = "test_pol_v1"

    # 1) upsert overwrite
    resp = client.post(
        f"/v1/policies/{pid}/chunks:upsert",
        json={"chunks": _make_chunks(), "mode": "overwrite", "expected_dim": 4},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["written"] == 3
    assert body["table_size"] == 3
    assert body["dim"] == 4

    # 2) meta
    meta = client.get(f"/v1/policies/{pid}/meta").json()
    assert meta["n_chunks"] == 3
    assert meta["n_original"] == 1
    assert meta["n_derived"] == 2

    # 3) search：BM25 命中"萝卜"应该返回 chunk_id=2
    resp = client.post(
        f"/v1/policies/{pid}/search",
        json={
            "query_tokenized": "萝卜",
            "query_vector": [0.0, 1.0, 0.0, 0.0],
            "top_n": 5,
            "top_m": 5,
            "include_content": True,
        },
    )
    assert resp.status_code == 200, resp.text
    hits = resp.json()["hits"]
    assert any(h["chunk_id"] == 2 for h in hits)

    # 4) expand：父=1 应该有 2 个派生
    resp = client.post(
        f"/v1/policies/{pid}/relations:expand",
        json={"chunk_id": 1, "include_content": False},
    )
    assert resp.status_code == 200
    children = resp.json()["chunks"]
    assert {c["chunk_id"] for c in children} == {2, 3}

    # 5) lookup-in-policy：找 OTHER_POL/C-001 的引用应只命中 chunk 2
    resp = client.get(
        f"/v1/policies/{pid}/relations:lookup",
        params={"target_policy_id": "OTHER_POL", "target_clause_id": "C-001"},
    )
    assert resp.status_code == 200
    chunks = resp.json()["chunks"]
    assert len(chunks) == 1 and chunks[0]["chunk_id"] == 2

    # 6) global dependents：OTHER_POL 应被 test_pol_v1 引用
    resp = client.get(
        "/v1/relations:lookup-dependents",
        params={"target_policy_id": "OTHER_POL"},
    )
    assert resp.status_code == 200
    deps = resp.json()["dependents"]
    assert any(d["source_policy_id"] == pid for d in deps)

    # 7) drop
    resp = client.delete(f"/v1/policies/{pid}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # 8) drop 后 meta 404
    assert client.get(f"/v1/policies/{pid}/meta").status_code == 404

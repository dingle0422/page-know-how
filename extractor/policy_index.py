"""
page_knowledge/_policy_index.json 的统一读写模块。

历史上索引文件是 `{policyId: dirname}` 的扁平 dict，仅供 app.py 做"已抽取过的 policy
按 dirname 反查 knowledge_root"用。本模块在保持完全向后兼容的前提下，扩展索引为：

    {
      "<policyId>": {
        "root": "<dirname relative to page_knowledge/>",
        "name": "<策略可读名>",
        "version": "<版本号>",
        "clauses": {
          "<clauseId>": "<相对 root 的目录 relpath>",
          ...
        }
      },
      ...
    }

新增 `clauses` 二级索引专门服务于 v1 关联展开（reasoner/v1/clause_locator.py）：
按 (policyId, clauseId) 直接定位本地 clause.json，避免运行时全树扫描。

向后兼容策略：
- 加载时若发现 value 是 str（老格式），就地规范化为 {"root": str, "name": "", "version": "",
  "clauses": {}}，并在内存层提供 `get_root(policy_id)` / `get_legacy_view()` 访问器。
- 写回时一律使用嵌套结构；老 reader（即未升级的 app.py）会读到 dict 而非 str，因此
  app.py 必须同步升级（已在本次修改中处理）。
- 对索引文件的并发读改写用 `threading.Lock` 包一层（同进程内安全），跨进程并发抽取由
  调用方保证不发生（实际场景下 extract_from_api 由 HTTP 入口串行触发）。
"""

import os
import json
import logging
import threading

logger = logging.getLogger(__name__)

_FILE_LOCK = threading.Lock()


def _normalize_entry(value) -> dict:
    """把任意历史格式 entry 规范化为新 schema dict。"""
    if isinstance(value, str):
        return {"root": value, "name": "", "version": "", "clauses": {}}
    if isinstance(value, dict):
        return {
            "root": value.get("root") or "",
            "name": value.get("name") or "",
            "version": value.get("version") or "",
            "clauses": dict(value.get("clauses") or {}),
        }
    logger.warning(f"policy index entry 非法类型 {type(value)}，已忽略")
    return {"root": "", "name": "", "version": "", "clauses": {}}


def load_index(path: str) -> dict[str, dict]:
    """加载并规范化索引；文件不存在或解析失败返回空 dict（不抛异常）。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"加载 policy 索引失败，将视为空: {path} ({e})")
        return {}
    if not isinstance(raw, dict):
        logger.warning(f"policy 索引根非 dict，已忽略: {path}")
        return {}
    return {pid: _normalize_entry(v) for pid, v in raw.items()}


def save_index(path: str, index: dict[str, dict]) -> None:
    """把规范化后的索引整体写回磁盘（原子替换）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except OSError as e:
        logger.error(f"保存 policy 索引失败: {path} ({e})")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def upsert_policy(
    path: str,
    policy_id: str,
    root_dirname: str,
    name: str = "",
    version: str = "",
    clauses: dict[str, str] | None = None,
) -> None:
    """合并写入单个 policy 条目。clauses 整段覆盖（每次抽取都是该 policy 的全量条款）。"""
    with _FILE_LOCK:
        index = load_index(path)
        entry = _normalize_entry(index.get(policy_id, {}))
        entry["root"] = root_dirname
        if name:
            entry["name"] = name
        if version:
            entry["version"] = version
        if clauses is not None:
            entry["clauses"] = dict(clauses)
        index[policy_id] = entry
        save_index(path, index)


def get_root_map(path: str) -> dict[str, str]:
    """老 app.py 视角：返回 {policy_id: root_dirname}。"""
    return {pid: entry.get("root", "") for pid, entry in load_index(path).items() if entry.get("root")}

import sys
import os
import json
import logging
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from extractor.builder import extract_from_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Page Know-How 单问题推理服务")

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_PAGE_KNOWLEDGE_DIR = os.path.join(_PROJECT_ROOT, "page_knowledge")
_POLICY_INDEX_FILE = os.path.join(_PAGE_KNOWLEDGE_DIR, "_policy_index.json")

# policyId -> knowledge_dir 内存缓存
_knowledge_cache: dict[str, str] = {}

MAX_CONCURRENT_REASONING = int(os.environ.get("MAX_CONCURRENT_REASONING", "10"))
_reasoning_semaphore: asyncio.Semaphore | None = None


def _get_reasoning_semaphore() -> asyncio.Semaphore:
    """懒初始化，确保在事件循环内创建 Semaphore"""
    global _reasoning_semaphore
    if _reasoning_semaphore is None:
        _reasoning_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REASONING)
        logger.info(f"全局推理并发上限: {MAX_CONCURRENT_REASONING}")
    return _reasoning_semaphore


def _load_policy_index() -> dict[str, str]:
    """从磁盘加载 policyId -> 目录名 的持久化索引"""
    if os.path.exists(_POLICY_INDEX_FILE):
        try:
            with open(_POLICY_INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"加载 policy 索引文件失败，将重建: {e}")
    return {}


def _save_policy_index(index: dict[str, str]) -> None:
    """将 policyId -> 目录名 的索引持久化到磁盘"""
    os.makedirs(_PAGE_KNOWLEDGE_DIR, exist_ok=True)
    try:
        with open(_POLICY_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error(f"保存 policy 索引文件失败: {e}")


def _is_valid_knowledge_dir(dir_path: str) -> bool:
    return os.path.isdir(dir_path) and os.path.exists(os.path.join(dir_path, "knowledge.md"))


class ReasonRequest(BaseModel):
    policyId: str
    question: str
    chunkSize: int = Field(default=5000, description="知识分块模式的字符数上限，0 表示不启用（默认 5000 启用 chunk 模式）")


class ReasonData(BaseModel):
    khObj: str = Field(description="知识点名称到章节编号的 JSON 字符串映射")
    answer: str


class ReasonResponse(BaseModel):
    data: ReasonData | None = None
    status_code: int
    message: str


def _get_or_extract_knowledge(policy_id: str) -> str:
    """
    获取 policyId 对应的知识目录，按以下优先级查找：
    1. 内存缓存
    2. 磁盘索引文件 (_policy_index.json)
    3. 均未命中则调用 API 抽取，并更新缓存与索引
    """
    # 1) 内存缓存
    if policy_id in _knowledge_cache:
        cached = _knowledge_cache[policy_id]
        if _is_valid_knowledge_dir(cached):
            logger.info(f"[命中内存缓存] policyId={policy_id} -> {cached}")
            return cached

    # 2) 磁盘索引
    disk_index = _load_policy_index()
    if policy_id in disk_index:
        dir_name = disk_index[policy_id]
        candidate = os.path.join(_PAGE_KNOWLEDGE_DIR, dir_name)
        if _is_valid_knowledge_dir(candidate):
            _knowledge_cache[policy_id] = candidate
            logger.info(f"[命中磁盘索引] policyId={policy_id} -> {candidate}")
            return candidate
        else:
            logger.warning(f"磁盘索引中记录的目录无效，将重新抽取: {candidate}")

    # 3) 调用 API 抽取
    logger.info(f"[新建抽取] policyId={policy_id}")
    knowledge_dir = extract_from_api(policy_id)

    # 更新内存缓存
    _knowledge_cache[policy_id] = knowledge_dir

    # 更新磁盘索引
    disk_index = _load_policy_index()
    dir_name = os.path.basename(knowledge_dir)
    disk_index[policy_id] = dir_name
    _save_policy_index(disk_index)

    logger.info(f"知识目录抽取完成并已索引: {knowledge_dir}")
    return knowledge_dir


def _build_kh_obj(graph) -> dict[str, str]:
    """从推理结果中构建 知识点名称 -> 章节编号 映射"""
    chunk_headings = getattr(graph, '_chunk_relevant_headings', [])
    if chunk_headings:
        return _build_kh_obj_from_headings(chunk_headings)

    all_dirs: list[str] = []
    for r in graph.all_results:
        all_dirs.extend(r.relevant_dirs)

    if graph.retrieval_mode and graph.retrieval_registry:
        for frag in graph.retrieval_registry.get_all():
            if frag.directory_path not in all_dirs:
                all_dirs.append(frag.directory_path)

    kh_map: dict[str, str] = {}
    for dir_path in all_dirs:
        rel = os.path.relpath(dir_path, graph.knowledge_root)
        if rel == ".":
            continue
        parts = rel.replace("\\", "/").split("/")
        leaf = parts[-1]
        if "_" in leaf:
            chapter_num, chapter_name = leaf.split("_", 1)
            if chapter_name not in kh_map:
                kh_map[chapter_name] = chapter_num

    return kh_map


def _build_kh_obj_from_headings(headings: list[str]) -> dict[str, str]:
    """从 relevant_headings 列表构建 khObj，只取最细粒度的叶子节点。

    例如 "2_涉税处理 > 2.1_增值税" -> {"增值税": "2.1"}
    """
    kh_map: dict[str, str] = {}
    for heading in headings:
        cleaned = heading.strip().strip("【】")
        parts = [p.strip() for p in cleaned.split(">")]
        leaf = parts[-1]
        if "_" in leaf:
            chapter_num, chapter_name = leaf.split("_", 1)
            if chapter_num and chapter_num[0].isdigit() and chapter_name not in kh_map:
                kh_map[chapter_name] = chapter_num
    return kh_map


def _import_agent_graph(version: str):
    if version == "v0":
        from reasoner.v0.agent_graph import AgentGraph
    else:
        from reasoner.v1.agent_graph import AgentGraph
    return AgentGraph


def _run_reasoning(
    question: str,
    knowledge_dir: str,
    version: str = "v1",
    check_pitfalls: bool = True,
    chunk_size: int = 0,
) -> dict:
    """执行单问题推理，返回 answer 和 kh_obj"""
    AgentGraphCls = _import_agent_graph(version)
    graph = AgentGraphCls(
        question=question,
        knowledge_root=knowledge_dir,
        max_rounds=10,
        vendor="aliyun",
        model="deepseek-v3.2",
        clean_answer=True,
        summary_batch_size=3,
        retrieval_mode=True if chunk_size == 0 else False,
        check_pitfalls=check_pitfalls,
        chunk_size=chunk_size,
    )
    result = graph.run()
    kh_obj = _build_kh_obj(graph)
    return {
        "answer": result["answer"],
        "kh_obj": kh_obj,
    }


@app.post("/api/reason", response_model=ReasonResponse)
async def reason(req: ReasonRequest):
    sem = _get_reasoning_semaphore()
    async with sem:
        try:
            knowledge_dir = await asyncio.to_thread(
                _get_or_extract_knowledge, req.policyId
            )
            result = await asyncio.to_thread(
                _run_reasoning, req.question, knowledge_dir,
                chunk_size=req.chunkSize,
            )
            return ReasonResponse(
                data=ReasonData(
                    khObj=json.dumps(result["kh_obj"], ensure_ascii=False),
                    answer=result["answer"],
                ),
                status_code=200,
                message="success",
            )
        except Exception as e:
            logger.exception(f"推理失败: {e}")
            return ReasonResponse(
                data=None,
                status_code=500,
                message=f"推理失败: {str(e)}",
            )


@app.get("/example")
async def example():
    """
    Example health check endpoint.
    """
    return {"status": "ok", "message": "Server is running."}


@app.get("/")
async def check():
    """
    Example health check endpoint.
    """
    return {"status": "ok", "message": "Server is running."}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, timeout_keep_alive=3600)

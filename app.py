import sys
import os
import json
import logging
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from extractor.builder import extract_from_api
from reasoner.agent_graph import AgentGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Page Know-How 单问题推理服务")

# policyId -> knowledge_dir 内存缓存，避免重复抽取
_knowledge_cache: dict[str, str] = {}


class ReasonRequest(BaseModel):
    policyId: str
    question: str


class ReasonData(BaseModel):
    khObj: str = Field(description="知识点名称到章节编号的 JSON 字符串映射")
    answer: str


class ReasonResponse(BaseModel):
    data: ReasonData | None = None
    status_code: int
    message: str


def _get_or_extract_knowledge(policy_id: str) -> str:
    """获取或抽取 policyId 对应的知识目录，优先使用缓存"""
    if policy_id in _knowledge_cache:
        cached = _knowledge_cache[policy_id]
        if os.path.isdir(cached) and os.path.exists(os.path.join(cached, "knowledge.md")):
            logger.info(f"命中缓存知识目录: {cached}")
            return cached

    logger.info(f"开始抽取知识目录, policyId={policy_id}")
    knowledge_dir = extract_from_api(policy_id)
    _knowledge_cache[policy_id] = knowledge_dir
    logger.info(f"知识目录抽取完成: {knowledge_dir}")
    return knowledge_dir


def _build_kh_obj(graph: AgentGraph) -> dict[str, str]:
    """从推理结果中构建 知识点名称 -> 章节编号 映射"""
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


def _run_reasoning(question: str, knowledge_dir: str) -> dict:
    """执行单问题推理，返回 answer 和 kh_obj"""
    graph = AgentGraph(
        question=question,
        knowledge_root=knowledge_dir,
        max_rounds=5,
        vendor="aliyun",
        model="deepseek-v3.2",
        clean_answer=True,
        retrieval_mode=False,
    )
    result = graph.run()
    kh_obj = _build_kh_obj(graph)
    return {
        "answer": result["answer"],
        "kh_obj": kh_obj,
    }


@app.post("/api/reason", response_model=ReasonResponse)
async def reason(req: ReasonRequest):
    try:
        knowledge_dir = await asyncio.to_thread(
            _get_or_extract_knowledge, req.policyId
        )
        result = await asyncio.to_thread(
            _run_reasoning, req.question, knowledge_dir
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import sys
import os
import json
import logging
import asyncio
import time

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
    maxRounds: int = Field(default=10, description="每个子智能体的最大 ReAct 轮次（默认 5）")
    vendor: str = Field(default="qwen3.5-122b-a10b", description="LLM 供应商（默认 qwen3.5-122b-a10b 直连）")
    model: str = Field(default="Qwen3.5-122B-A10B", description="LLM 模型名称（默认 Qwen3.5-122B-A10B）")
    cleanAnswer: bool = Field(
        default=False,
        description="启用答案清洗：在 summary 后追加一轮 LLM 调用，以咨询客服口吻输出精简结论（默认 False）",
    )
    summaryBatchSize: int = Field(
        default=3,
        description="分批并行压缩总结：指定每批包含的证据条数（如 3），启用后自动激活分层总结模式。默认 0 表示不分批",
    )
    retrievalMode: bool = Field(
        default=True,
        description="启用召回模式：子智能体仅做相关性判定并收集原始知识，避免探索阶段信息畸变（默认 False）",
    )
    checkPitfalls: bool = Field(
        default=True,
        description="启用易错点收集：第一层推理时让 LLM 同步产出易错点，"
                        "并在汇总前注入到证据/知识片段中，由总结阶段一并参考；"
                        "V1 不会追加独立的二次校验 LLM 调用",
    )
    chunkSize: int = Field(
        default=3000,
        description="知识分块模式的字符数上限，0 表示不启用（web 默认 3000 启用 chunk 模式）",
    )
    version: str = Field(
        default="v1",
        description="推理引擎版本（v0=原始版本, v1=统一EXPLORE+三层目录树，默认 v1）",
    )
    enableSkills: bool = Field(
        default=True,
        description="是否启用 skill 评估与 double-check（默认开启，对应 CLI 的 --disable-skills 取反）",
    )
    summaryCleanAnswer: bool = Field(
        default=True,
        description="启用 summary+clean 一体化："
                    "在 retrieval 分批合并阶段直接产出面向用户的客服话术答案，"
                    "跳过独立的 clean-answer 调用以减少一次 LLM 串行延迟",
    )
    answerSystemPrompt: str | None = Field(
        default=None,
        description="最终作答阶段的 system prompt 自定义内容："
                    "若不传或传空字符串，则使用内置的 SUMMARY_ANSWER_SYSTEM_PROMPT 默认版本；"
                    "中间提炼层仍固定使用 SUMMARY_EXTRACT_SYSTEM_PROMPT，不受此参数影响。"
                    "仅在 version=v1 下生效",
    )
    thinkMode: bool = Field(
        default=False,
        description="启用 think 模式：在【所有最终节点】的 summary+clean 阶段，"
                    "改用 *_AND_CLEAN_THINK 版 prompt，要求模型严格按 "
                    "<think>...</think><answer>...</answer> 双标签输出。"
                    "覆盖范围不受分批/召回/chunk 影响（非分批 SUMMARY_AND_CLEAN、"
                    "分批 BATCH_MERGE_AND_CLEAN，及其 RETRIEVAL_* 对应版本均会切换）；"
                    "中间提炼 prompt 始终保持原样不动。"
                    "需配合 summaryCleanAnswer=True 使用；仅在 version=v1 下生效",
    )


class ReasonData(BaseModel):
    khObj: str = Field(description="知识点名称到章节编号的 JSON 字符串映射")
    policyId: str
    answer: str
    skillsResult: dict[str, str] = Field(
        default_factory=dict,
        description="本次推理实际调用的 skill 结果：{skill_name: stdout}；未启用 skill 或未触发任何 skill 时为空 {}",
    )


class ReasonResponse(BaseModel):
    data: ReasonData | None = None
    status_code: int
    message: str


class KhUpdateRequest(BaseModel):
    khId: str
    version: str
    typeId: str = Field(default="0", description="区分类型：行业经验0、基础知识1、三方知识2等")


class KhUpdateData(BaseModel):
    policyId: str
    update_time: str = Field(description="毫秒级时间戳")


class KhUpdateResponse(BaseModel):
    data: KhUpdateData | None = None
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
    max_rounds: int = 5,
    vendor: str = "qwen3.5-122b-a10b",
    model: str = "Qwen3.5-122B-A10B",
    clean_answer: bool = False,
    summary_batch_size: int = 0,
    retrieval_mode: bool = False,
    check_pitfalls: bool = False,
    chunk_size: int = 0,
    enable_skills: bool = True,
    summary_clean_answer: bool = False,
    answer_system_prompt: str | None = None,
    think_mode: bool = False,
) -> dict:
    """执行单问题推理，返回 answer 和 kh_obj"""
    AgentGraphCls = _import_agent_graph(version)
    extra_kwargs = {}
    if version == "v1":
        extra_kwargs["summary_clean_answer"] = summary_clean_answer
        extra_kwargs["answer_system_prompt"] = answer_system_prompt
        extra_kwargs["think_mode"] = think_mode
    else:
        if summary_clean_answer:
            logger.warning("summaryCleanAnswer 仅在 version=v1 下生效，本次将被忽略")
        if answer_system_prompt:
            logger.warning("answerSystemPrompt 仅在 version=v1 下生效，本次将被忽略")
        if think_mode:
            logger.warning("thinkMode 仅在 version=v1 下生效，本次将被忽略")
    graph = AgentGraphCls(
        question=question,
        knowledge_root=knowledge_dir,
        max_rounds=max_rounds,
        vendor=vendor,
        model=model,
        clean_answer=clean_answer,
        summary_batch_size=summary_batch_size,
        retrieval_mode=retrieval_mode,
        check_pitfalls=check_pitfalls,
        chunk_size=chunk_size,
        enable_skills=enable_skills,
        **extra_kwargs,
    )
    result = graph.run()
    kh_obj = _build_kh_obj(graph)
    skills_result = _build_skills_result(graph)
    return {
        "answer": result["answer"],
        "kh_obj": kh_obj,
        "skills_result": skills_result,
    }


def _build_skills_result(graph) -> dict[str, str]:
    """聚合 skill_registry 中的执行结果为 {skill_name: dumped_stdout}。

    - skill 关闭 / 未触发任何 skill / 全部失败    -> 返回 {}
    - 同一 skill 多次调用                          -> stdout 以两个换行拼接
    - 失败的调用                                   -> 跳过（保证字段语义为"可用结果"）
    """
    registry = getattr(graph, "skill_registry", None)
    if registry is None or not registry.has_any():
        return {}

    bucket: dict[str, list[str]] = {}
    for rec in registry.get_all():
        if not rec.result.success:
            continue
        text = (rec.result.stdout or "").strip()
        if not text:
            continue
        bucket.setdefault(rec.skill_name, []).append(text)

    return {name: "\n\n".join(parts) for name, parts in bucket.items()}


def _create_knowledge(policy_id: str) -> str:
    """
    新建知识目录。调用 API 抽取条款并生成目录树，更新缓存与索引。
    不会删除已有目录——每个 policyId 对应唯一版本，旧版本知识仍可被推理任务使用。
    """
    logger.info(f"[新增知识] policyId={policy_id}")
    knowledge_dir = extract_from_api(policy_id)

    _knowledge_cache[policy_id] = knowledge_dir

    disk_index = _load_policy_index()
    disk_index[policy_id] = os.path.basename(knowledge_dir)
    _save_policy_index(disk_index)

    logger.info(f"知识目录新增完成并已索引: {knowledge_dir}")
    return knowledge_dir


@app.post("/api/kh/update", response_model=KhUpdateResponse)
async def kh_update(req: KhUpdateRequest):
    policy_id = f"{req.khId}_{req.version}"

    disk_index = _load_policy_index()
    if policy_id in disk_index:
        existing_dir = os.path.join(_PAGE_KNOWLEDGE_DIR, disk_index[policy_id])
        if _is_valid_knowledge_dir(existing_dir):
            logger.warning(f"policyId 已存在，拒绝重复创建: {policy_id}")
            return KhUpdateResponse(
                data=None,
                status_code=409,
                message=f"知识更新失败: policyId={policy_id} 已存在，请勿重复提交相同的 khId+version 组合",
            )

    try:
        await asyncio.to_thread(_create_knowledge, policy_id)
        return KhUpdateResponse(
            data=KhUpdateData(
                policyId=policy_id,
                update_time=str(int(time.time() * 1000)),
            ),
            status_code=200,
            message="success",
        )
    except Exception as e:
        logger.exception(f"知识更新失败: {e}")
        return KhUpdateResponse(
            data=None,
            status_code=500,
            message=f"知识更新失败: {str(e)}",
        )


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
                version=req.version,
                max_rounds=req.maxRounds,
                vendor=req.vendor,
                model=req.model,
                clean_answer=req.cleanAnswer,
                summary_batch_size=req.summaryBatchSize,
                retrieval_mode=req.retrievalMode,
                check_pitfalls=req.checkPitfalls,
                chunk_size=req.chunkSize,
                enable_skills=req.enableSkills,
                summary_clean_answer=req.summaryCleanAnswer,
                answer_system_prompt=req.answerSystemPrompt,
                think_mode=req.thinkMode,
            )
            return ReasonResponse(
                data=ReasonData(
                    khObj=json.dumps(result["kh_obj"], ensure_ascii=False),
                    answer=result["answer"],
                    skillsResult=result.get("skills_result", {}),
                    policyId=req.policyId,
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

import sys
import os
import re
import json
import logging
import asyncio
import time
from tkinter.constants import TRUE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from extractor.builder import extract_from_api
from extractor.policy_index import get_root_map as _policy_index_root_map
from utils.verbose_logger import (
    open_session as _open_verbose_session,
    VERBOSE_DEFAULT_ENABLED,
    log_event as _log_verbose_event,
)

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
    """从磁盘加载 policyId -> 目录名 的扁平视图。

    底层文件已升级为嵌套 schema（含 clauses 二级索引，供关联展开使用），
    本函数仅暴露 root 字段，保留原签名以减少调用点改动。写入由
    `extractor.policy_index.upsert_policy`（在 extract_from_api 内部）统一负责。
    """
    return _policy_index_root_map(_POLICY_INDEX_FILE)


def _is_valid_knowledge_dir(dir_path: str) -> bool:
    return os.path.isdir(dir_path) and os.path.exists(os.path.join(dir_path, "knowledge.md"))


class ReasonRequest(BaseModel):
    policyId: str
    question: str
    maxRounds: int = Field(default=10, description="每个子智能体的最大 ReAct 轮次（默认 5）")
    vendor: str = Field(default="aliyun", description="LLM 供应商（默认 qwen3.5-122b-a10b 直连）")
    model: str = Field(default="deepseek-v3.2", description="LLM 模型名称（默认 Qwen3.5-122B-A10B）")
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
    lastThink: bool = Field(
        default=True,
        description="在【全流程最后一步总结/清洗】阶段打开底层 LLM 的 enable_thinking=True，"
                    "让模型返回推理轨迹（qwen3.5/3.6 会把 <think>...</think> 写进 content；"
                    "deepseek-reasoner / deepseek-v3.2 等会返回到 message.reasoning_content，"
                    "llm/client.py 已统一前缀回注到 content）。"
                    "只作用于最终节点的 chat 调用（all_in_answer / final_summary / "
                    "batch_final_merge / retrieval_final_summary / retrieval_batch_final_merge / "
                    "clean_answer），中间 batch/chunk/探索阶段都不受影响。"
                    "与 thinkMode 正交：thinkMode 只改 prompt 模板（要求 JSON 输出结构），"
                    "lastThink 只改 chat_template_kwargs.enable_thinking（开启模型推理轨迹）。",
    )
    thinkMode: bool = Field(
        default=True,
        description="启用 think 模式：在【所有最终节点】的 summary+clean 阶段，"
                    "改用 *_AND_CLEAN_THINK 版 prompt，要求模型严格按 "
                    "JSON 对象 {\"analysis\": \"...\", \"concise_answer\": \"...\"} 输出。"
                    "字段语义：analysis = 完整客服回答（受 ≤500 字等所有硬约束），"
                    "concise_answer = 在 analysis 之上提炼出的核心结论（一句话/极简短句）。"
                    "解析后映射到响应体：think <- analysis，answer <- concise_answer。"
                    "覆盖范围不受分批/召回/chunk 影响（非分批 SUMMARY_AND_CLEAN、"
                    "分批 BATCH_MERGE_AND_CLEAN，及其 RETRIEVAL_* 对应版本均会切换）；"
                    "中间提炼 prompt 始终保持原样不动。"
                    "需配合 summaryCleanAnswer=True 使用；仅在 version=v1 下生效",
    )
    enableRelations: bool = Field(
        default=True,
        description="启用关联条款展开：当某个 chunk / 子智能体命中相关知识且其目录包含 "
                    "clause.json 中的预展开 references 时，按 LLM 多跳并发拉取外部条款，"
                    "在 chunk 模式下切分为派生 chunk 后续与原 chunk 一并进入流式 batch summary；"
                    "在 standard / retrieval 模式下 inline 追加到对应 fragment 末尾。"
                    "本地缺失时自动通过 DEFAULT_CLAUSE_API_URL 实时拉取。仅在 version=v1 下生效。",
    )
    relationMaxDepth: int = Field(
        default=5,
        description="关联展开最大跳深（含首跳）。enableRelations=False 时忽略。",
    )
    relationMaxNodes: int = Field(
        default=999,
        description="单次 chunk / 子智能体触发的关联展开 BFS 总节点数上限，超出即停止扩展（横向纵向都包含）。",
    )
    relationWorkers: int = Field(
        default=8,
        description="关联展开的调度线程数；",
    )
    relationsExpansionMode: str = Field(
        default="all",
        description="关联展开模式。enableRelations=True 时生效：\n"
                    "- 'all'（默认）：跳过对每个候选条款的 LLM 二次判定，"
                    "  只要 ClauseLocator 能定位到内容，一律入 RelationFragment 并按深度全展开。"
                    "  触发率 100%、省掉 N 次 LLM 评估调用反而更快，token 略增；"
                    "  外层闸（chunk LLM relevant_headings 命中）仍然保留。\n"
                    "- 'smart'：每个候选条款都用 LLM 判 is_relevant，准确率高但触发率受 LLM "
                    "  采样波动影响（同一问题多次跑可能命中数不一致）。",
    )
    summaryPipelineMode: str = Field(
        default="layered",
        description="batch summary 流水线模式（summaryBatchSize > 0 时生效）：\n"
                    "- 'layered'（默认）：当前实现。chunk + 关联展开走 _chunk_streaming_pipeline\n"
                    "  的'按 slot 顺序流式 batch + 后续递归压缩按层同步'；其他入口走\n"
                    "  _recursive_batch_reduce 同步分层。\n"
                    "- 'reduce_queue'：所有压缩任务统一进 ReducePipeline，凑批+回灌，"
                    "  无层间同步点。chunk 数大、batch 长尾差异显著时可见明显加速；"
                    "  代价是早 flush 的 part 经过更多次中间压缩。"
                    "  生产侧的相关性激活逻辑（chunk LLM relevant_headings 命中才展开关联）"
                    "  完全保留。",
    )
    reduceMaxPartDepth: int = Field(
        default=4,
        description="reduce_queue 模式下单 part 经过的最大中间 BATCH_SUMMARY 次数。"
                    "命中上限的 part 不再参与凑批，转入 frozen 列表直接保留到 final merge，"
                    "避免某些 part 被反复压缩造成信息严重损耗。"
                    "调大可减少 frozen 数量、让 final merge 入口更接近 batch_size；"
                    "调小则相反。summaryPipelineMode='layered' 时本字段忽略。",
    )
    verbose: bool = Field(
        default=VERBOSE_DEFAULT_ENABLED,
        description="启用 verbose 模式：以当次请求的 session 编号为文件名，"
                    "完整记录该请求下根智能体 / 子智能体 / 汇总层每一步的 LLM 输入输出。"
                    "日志落盘在 <project>/verbose_logs/ 目录；当累计大小 ≥ 20MB 时，"
                    "自动打包为 archives/verbose_logs_<起始时间>__<结束时间>.zip 并清理原始文件。"
                    "未显式传入时默认跟随环境变量 VERBOSE_TRACE 的配置。",
    )
    sessionId: str | None = Field(
        default=None,
        description="可选：显式指定 verbose session 编号（便于跨服务日志串联）；"
                    "不传则自动生成 sess-<随机>。仅在 verbose=True 时生效。",
    )


class ReasonData(BaseModel):
    khObj: str = Field(description="知识点名称到章节编号的 JSON 字符串映射")
    policyId: str
    answer: str = Field(
        description="最终面向用户的客服回答。"
                    "thinkMode=False 时：模型 *_AND_CLEAN_PROMPT 输出的完整客服回答（≤500 字）；"
                    "thinkMode=True 时：取自 LLM JSON 输出中的 concise_answer 字段（一句话核心结论）。"
                    "如 thinkMode=True 但模型未严格按 JSON 输出，则兜底为模型原始输出全文",
    )
    think: str = Field(
        default="",
        description="模型在 thinkMode 开启时 LLM JSON 输出中的 analysis 字段，"
                    "即完整面向用户的客服回答（受 ≤500 字、客服口吻、知识原文引用等所有硬约束）。"
                    "未开启 thinkMode、或开启后模型未严格按 JSON 输出/缺少 analysis 字段时为空字符串",
    )
    skillsResult: dict[str, str] = Field(
        default_factory=dict,
        description="本次推理实际调用的 skill 结果：{skill_name: stdout}；未启用 skill 或未触发任何 skill 时为空 {}",
    )
    sessionId: str | None = Field(
        default=None,
        description="本次请求的 verbose session id。verbose=True 时为真正用于落盘的 id"
                    "（调用方传了 sessionId 就用调用方的，否则由服务端自动生成 sess-xxxxx）；"
                    "verbose=False 时回显调用方传入的值（可能为 null）。"
                    "配合 logName 可直接定位到 verbose_logs/ 下对应的 jsonl 日志文件",
    )
    logName: str | None = Field(
        default=None,
        description="verbose=True 时，本次 session 对应的日志文件名，格式 "
                    "<YYYYMMDD_HHMMSS>_<sessionId>.jsonl，位于服务端 verbose_logs/ 目录下。"
                    "verbose=False 时为 null（没有落盘）",
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

    # 3) 调用 API 抽取（extract_from_api 内部已回填 _policy_index.json）
    logger.info(f"[新建抽取] policyId={policy_id}")
    knowledge_dir = extract_from_api(policy_id)
    _knowledge_cache[policy_id] = knowledge_dir
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


# think_mode 下模型应输出形如 {"analysis": "...", "concise_answer": "..."} 的合法 JSON，
# 字段映射到响应体：think <- analysis, answer <- concise_answer。
# markdown 代码块（```json ... ```）剥离用：捕获中间净 JSON 主体。
_JSON_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
# 在原文里搜首个 { ... } 主体（贪婪到最末一个 }），用于模型在前后混入了说明文字时兜底。
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse_json_obj(text: str) -> dict | None:
    """优先 json.loads 严格解析；失败再退化到 json5（容忍尾随逗号、单引号、注释等）。
    解析结果不是 dict 时一律视为失败。
    """
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        import json5  # 已在 requirements 中
        obj = json5.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _split_analysis_concise_answer(raw: str, think_mode: bool) -> tuple[str, str]:
    """解析模型在 think_mode 下输出的 JSON({analysis, concise_answer})。

    返回 (think_text, answer_text)，分别对应 ReasonData.think / ReasonData.answer：
        think  <- analysis
        answer <- concise_answer

    解析策略（按顺序尝试，命中即停）：
      1. think_mode=False 或 raw 为空：直接 ("", raw or "")
      2. raw 原样 json.loads → 失败再 json5.loads（容忍非严格 JSON）
      3. 剥离 ```json ... ``` 代码块外壳后再尝试 step 2
      4. 用正则截取 raw 中首个 {...} 主体后再尝试 step 2
      5. 仍失败：视为不遵循 JSON 格式 → ("", raw)，并打 WARNING

    解析成功（拿到 dict）后字段缺失兜底：
      - 同时有 analysis 和 concise_answer → (analysis, concise_answer)
      - 只有 analysis → (analysis, "")
      - 只有 concise_answer → ("", concise_answer)
      - 都没有 → ("", raw) 并打 WARNING
    """
    if not think_mode or not raw:
        return "", raw or ""

    raw_stripped = raw.strip()

    parsed = _try_parse_json_obj(raw_stripped)

    if parsed is None:
        cb_match = _JSON_CODEBLOCK_RE.search(raw_stripped)
        if cb_match:
            parsed = _try_parse_json_obj(cb_match.group(1).strip())

    if parsed is None:
        obj_match = _JSON_OBJECT_RE.search(raw_stripped)
        if obj_match:
            parsed = _try_parse_json_obj(obj_match.group(0))

    if parsed is None:
        logger.warning(
            "[ThinkMode] 模型输出无法解析为 JSON({analysis, concise_answer}) "
            "(json/json5/codeblock/regex 全部失败)，"
            "本次 think 字段将返回空字符串，answer 字段保留模型原始输出"
        )
        return "", raw

    analysis_val = parsed.get("analysis", "")
    concise_val = parsed.get("concise_answer", "")
    analysis_text = analysis_val.strip() if isinstance(analysis_val, str) else ""
    concise_text = concise_val.strip() if isinstance(concise_val, str) else ""

    if not analysis_text and not concise_text:
        logger.warning(
            "[ThinkMode] 模型 JSON 中未提取到任何有效字段 "
            f"(实际字段={list(parsed.keys())})，"
            "本次 think 字段将返回空字符串，answer 字段保留模型原始输出"
        )
        return "", raw

    if not analysis_text:
        logger.warning(
            "[ThinkMode] 模型 JSON 缺少 analysis 字段，think 字段将返回空字符串"
        )
    if not concise_text:
        logger.warning(
            "[ThinkMode] 模型 JSON 缺少 concise_answer 字段，answer 字段将返回空字符串"
        )

    return analysis_text, concise_text


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
    last_think: bool = False,
    enable_relations: bool = False,
    relation_max_depth: int = 5,
    relation_max_nodes: int = 50,
    relation_workers: int = 8,
    relations_expansion_mode: str = "all",
    summary_pipeline_mode: str = "layered",
    reduce_max_part_depth: int = 5,
) -> dict:
    """执行单问题推理，返回 answer 和 kh_obj"""
    AgentGraphCls = _import_agent_graph(version)
    extra_kwargs = {}
    if version == "v1":
        extra_kwargs["summary_clean_answer"] = summary_clean_answer
        extra_kwargs["answer_system_prompt"] = answer_system_prompt
        extra_kwargs["think_mode"] = think_mode
        extra_kwargs["enable_relations"] = enable_relations
        extra_kwargs["relation_max_depth"] = relation_max_depth
        extra_kwargs["relation_max_nodes"] = relation_max_nodes
        extra_kwargs["relation_workers"] = relation_workers
        extra_kwargs["relations_expansion_mode"] = relations_expansion_mode
        extra_kwargs["summary_pipeline_mode"] = summary_pipeline_mode
        extra_kwargs["reduce_max_part_depth"] = reduce_max_part_depth
    else:
        if summary_clean_answer:
            logger.warning("summaryCleanAnswer 仅在 version=v1 下生效，本次将被忽略")
        if answer_system_prompt:
            logger.warning("answerSystemPrompt 仅在 version=v1 下生效，本次将被忽略")
        if think_mode:
            logger.warning("thinkMode 仅在 version=v1 下生效，本次将被忽略")
        if enable_relations:
            logger.warning("enableRelations 仅在 version=v1 下生效，本次将被忽略")
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
        last_think=last_think,
        **extra_kwargs,
    )
    result = graph.run()
    kh_obj = _build_kh_obj(graph)
    skills_result = _build_skills_result(graph)
    think_text, answer_text = _split_analysis_concise_answer(result["answer"], think_mode)
    return {
        "answer": answer_text,
        "think": think_text,
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
    新建知识目录。调用 API 抽取条款并生成目录树，由 extract_from_api 内部回填磁盘索引。
    不会删除已有目录——每个 policyId 对应唯一版本，旧版本知识仍可被推理任务使用。
    """
    logger.info(f"[新增知识] policyId={policy_id}")
    knowledge_dir = extract_from_api(policy_id)
    _knowledge_cache[policy_id] = knowledge_dir
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
        # verbose 开启时：以 sessionId（未传则自动生成）新建独立日志文件，
        # 完整记录本次请求涉及的所有智能体步骤；退出时自动触发归档。
        session_meta = {
            "endpoint": "/api/reason",
            "policyId": req.policyId,
            "question": req.question,
            "version": req.version,
            "vendor": req.vendor,
            "model": req.model,
            "retrievalMode": req.retrievalMode,
            "chunkSize": req.chunkSize,
            "summaryBatchSize": req.summaryBatchSize,
            "enableRelations": req.enableRelations,
            "thinkMode": req.thinkMode,
        }
        with _open_verbose_session(
            session_id=req.sessionId,
            meta=session_meta,
            enabled=req.verbose,
        ) as _session:
            # verbose=True → _session 为真实 _Session 对象，取其 session_id / file_path 回显；
            # verbose=False → _session 为 None，仅回显调用方原样的 sessionId，log_name 置空。
            resp_session_id = _session.session_id if _session is not None else req.sessionId
            resp_log_name = _session.file_path.name if _session is not None else None
            try:
                knowledge_dir = await asyncio.to_thread(
                    _get_or_extract_knowledge, req.policyId
                )
                if req.verbose:
                    _log_verbose_event(
                        "knowledge_ready",
                        policyId=req.policyId,
                        knowledge_dir=knowledge_dir,
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
                    last_think=req.lastThink,
                    enable_relations=req.enableRelations,
                    relation_max_depth=req.relationMaxDepth,
                    relation_max_nodes=req.relationMaxNodes,
                    relation_workers=req.relationWorkers,
                    relations_expansion_mode=req.relationsExpansionMode,
                    summary_pipeline_mode=req.summaryPipelineMode,
                    reduce_max_part_depth=req.reduceMaxPartDepth,
                )
                if req.verbose:
                    _log_verbose_event(
                        "reason_finished",
                        answer_preview=(result.get("answer") or "")[:200],
                        kh_obj=result.get("kh_obj", {}),
                    )
                return ReasonResponse(
                    data=ReasonData(
                        khObj=json.dumps(result["kh_obj"], ensure_ascii=False),
                        answer=result["answer"],
                        think=result.get("think", ""),
                        skillsResult=result.get("skills_result", {}),
                        policyId=req.policyId,
                        sessionId=resp_session_id,
                        logName=resp_log_name,
                    ),
                    status_code=200,
                    message="success",
                )
            except Exception as e:
                logger.exception(f"推理失败: {e}")
                if req.verbose:
                    _log_verbose_event("reason_error", error=repr(e))
                return ReasonResponse(
                    data=ReasonData(
                        khObj="{}",
                        answer=f"推理失败: {str(e)}",
                        think="",
                        skillsResult={},
                        policyId=req.policyId,
                        sessionId=resp_session_id,
                        logName=resp_log_name,
                    ) if resp_log_name or resp_session_id else None,
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

import sys
import os
import re
import json
import logging
import asyncio
import time
import uuid
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from extractor.builder import extract_from_api
from extractor.policy_index import get_root_map as _policy_index_root_map
from utils.verbose_logger import (
    open_session as _open_verbose_session,
    VERBOSE_DEFAULT_ENABLED,
    log_event as _log_verbose_event,
)
from redis_server.client import RedisServerClient
import task_queue as _tq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_PAGE_KNOWLEDGE_DIR = os.path.join(_PROJECT_ROOT, "page_knowledge")
_POLICY_INDEX_FILE = os.path.join(_PAGE_KNOWLEDGE_DIR, "_policy_index.json")

# policyId -> knowledge_dir 内存缓存
_knowledge_cache: dict[str, str] = {}

# ------------------------------------------------------------------ 任务队列配置
#
# 服务改造为"单 FastAPI 进程 + 内置 worker pool + 外置 redis_server"架构：
# - 请求进来统一写入 redis_server 的 queue:reason:pending 队列；
# - 本进程启动 MAX_CONCURRENT_REASONING 个 worker 从 BLPOP 拉任务执行；
# - /api/reason/submit 立刻返回 task_id，客户端轮询 /api/reason/result/{taskId} 取结果；
# - /api/reason 保留为"提交 + 内部轮询"同步包装，对外行为保持兼容。

MAX_CONCURRENT_REASONING = int(os.environ.get("MAX_CONCURRENT_REASONING", "10"))
# 单条任务结果在 redis_server 里的寿命（秒）；超过即被清理，客户端需要重新提交。
REASON_TASK_TTL_SECONDS = float(os.environ.get("REASON_TASK_TTL_SECONDS", str(7 * 24 * 3600)))
# /api/reason 同步包装轮询任务状态的间隔（秒）。
REASON_SYNC_POLL_INTERVAL = float(os.environ.get("REASON_SYNC_POLL_INTERVAL", "0.5"))
# worker 每轮 BLPOP 的阻塞时长（秒）。
# !!! 必须小于调用 redis_server 链路上最短的网关 proxy_read_timeout，
# 否则网关会先一步返回 504 Gateway Timeout。常见 Nginx/Ingress 默认 30~60s，
# 这里默认取 10s 留足余量；如果你的网关更激进，自行调小（但不要低于 2s）。
REASON_BLPOP_TIMEOUT_SECONDS = float(os.environ.get("REASON_BLPOP_TIMEOUT_SECONDS", "10"))
# 单条 executor 的硬超时（秒）。到点 asyncio.wait_for 会把 worker 协程打断、槽位释放,
# 对应 task 被标为 failed(error="executor timeout after Xs")。
# 默认 2h；如业务 P99 接近该值，改大即可；设为 0 / 负数 = 不套超时（回退老行为）。
# 注意：executor 里 asyncio.to_thread 跑的同步代码不可取消，底层线程要靠下游
# httpx/LLM client 自身的 read timeout 防泄漏。
REASON_EXECUTOR_TIMEOUT_SECONDS = float(os.environ.get("REASON_EXECUTOR_TIMEOUT_SECONDS", str(2 * 3600)))
REDIS_SERVER_URL = os.environ.get("REDIS_SERVER_URL", "http://mlp.paas.dc.servyou-it.com/redis-server")
REDIS_SERVER_AUTH_TOKEN = os.environ.get("REDIS_SERVER_AUTH_TOKEN", "")

# 本服务进程的实例 ID。每次启动生成一次（uuid），写入每条 running 任务的 instance_id,
# 使 startup cleanup 能精确区分"本进程正在跑的 running" vs "上一代崩掉留下的僵尸"。
INSTANCE_ID: str = ""

# lifespan 内初始化；模块级暴露给路由处理函数。
_redis_client: RedisServerClient | None = None
_worker_pool: _tq.WorkerPool | None = None


def _require_redis() -> RedisServerClient:
    """路由里统一入口：lifespan 还没跑完就别来。"""
    if _redis_client is None:
        raise HTTPException(status_code=503, detail="redis_server client not ready")
    return _redis_client


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _redis_client, _worker_pool, INSTANCE_ID
    INSTANCE_ID = str(uuid.uuid4())
    logger.info(f"[lifespan] 启动，REDIS_SERVER_URL={REDIS_SERVER_URL} INSTANCE_ID={INSTANCE_ID}")
    _redis_client = RedisServerClient(
        REDIS_SERVER_URL,
        auth_token=REDIS_SERVER_AUTH_TOKEN,
        timeout_seconds=15.0,
    )
    # 服务进程启动时探一次 health，失败就立刻抛错，避免 worker 陷入重复失败循环。
    try:
        h = await _redis_client.health()
        logger.info(f"[lifespan] redis_server health={h.get('status')}")
    except Exception as e:
        logger.exception(f"[lifespan] redis_server 不可达，推理服务无法启动: {e}")
        raise

    # 启动前扫一遍 task:*，把上一代进程崩掉留下的 running 僵尸全部改写成 failed,
    # 让客户端轮询立刻能看到 "server restarted while running" 而不必死等 TTL。
    # 必须放在 WorkerPool.start() 之前：避免新 worker 刚启动就把自己正在处理的
    # running 记录（instance_id == INSTANCE_ID）被并发扫到——反正也扫不到，
    # 因为 cleanup 只处理 instance_id != INSTANCE_ID 的条目，这里只是顺序保险。
    try:
        await _tq.cleanup_stale_running_tasks(
            _redis_client,
            current_instance_id=INSTANCE_ID,
            ttl_seconds=REASON_TASK_TTL_SECONDS,
        )
    except Exception as e:
        # 清理失败不应阻塞服务启动——大不了那些僵尸记录等 TTL 到期自己消失,
        # 新任务仍然能正常入队执行。
        logger.warning(f"[lifespan] 启动清理僵尸 running 任务失败（忽略继续启动）: {e}")

    _worker_pool = _tq.WorkerPool(
        _redis_client,
        executor=_reason_executor,
        worker_count=MAX_CONCURRENT_REASONING,
        task_ttl_seconds=REASON_TASK_TTL_SECONDS,
        blpop_timeout_seconds=REASON_BLPOP_TIMEOUT_SECONDS,
        instance_id=INSTANCE_ID,
        executor_timeout_seconds=REASON_EXECUTOR_TIMEOUT_SECONDS,
    )
    await _worker_pool.start()
    try:
        yield
    finally:
        logger.info("[lifespan] 关闭 worker pool + redis client")
        if _worker_pool is not None:
            await _worker_pool.stop()
        if _redis_client is not None:
            await _redis_client.close()
        _worker_pool = None
        _redis_client = None


app = FastAPI(title="Page Know-How 单问题推理服务", lifespan=lifespan)


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
        default="v2",
        description="推理引擎版本（v0=原始版本, v1=统一EXPLORE+三层目录树, v2=KV-cache 优化 prompt, "
                    "v3=v2 基础上把最终汇总节点 system prompt 切换为 _CORPUS_SYSTEM_PROMPT 训练样本风格；"
                    "默认 v2）",
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
                    "仅作用于【最终总结/作答】节点（all_in_answer / final_summary / "
                    "batch_final_merge / retrieval_final_summary / retrieval_batch_final_merge），"
                    "不会影响中间提炼层（SUMMARY_EXTRACT_SYSTEM_PROMPT / BATCH_REDUCE_SYSTEM_PROMPT）。"
                    "拼接方式：不传或传空字符串时直接使用内置的 SUMMARY_ANSWER_SYSTEM_PROMPT；"
                    "传入非空内容时不会覆盖默认 prompt，而是按"
                    "「## 最高行为准则\\n{自定义}\\n\\n## 默认作答规范\\n{SUMMARY_ANSWER_SYSTEM_PROMPT}」"
                    "结构拼接（自定义部分优先级最高，与默认冲突时以自定义为准）。"
                    "仅在 version=v1/v2 下生效",
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
                    "JSON 对象 {\"analysis\": \"...\", \"answer\": \"...\"} 输出。"
                    "字段语义：analysis = 完整客服回答（受 ≤500 字等所有硬约束），"
                    "answer = 基于分析内容给出回答用户的最终答案。"
                    "解析后映射到响应体：think <- analysis，answer <- answer。"
                    "覆盖范围不受分批/召回/chunk 影响（非分批 SUMMARY_AND_CLEAN、"
                    "分批 BATCH_MERGE_AND_CLEAN，及其 RETRIEVAL_* 对应版本均会切换）；"
                    "中间提炼 prompt 始终保持原样不动。"
                    "需配合 summaryCleanAnswer=True 使用；仅在 version=v1/v2 下生效",
    )
    enableRelations: bool = Field(
        default=True,
        description="启用关联条款展开：当某个 chunk / 子智能体命中相关知识且其目录包含 "
                    "clause.json 中的预展开 references 时，按 LLM 多跳并发拉取外部条款，"
                    "在 chunk 模式下切分为派生 chunk 后续与原 chunk 一并进入流式 batch summary；"
                    "在 standard / retrieval 模式下 inline 追加到对应 fragment 末尾。"
                    "本地缺失时自动通过 DEFAULT_CLAUSE_API_URL 实时拉取。仅在 version=v1/v2 下生效。",
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
    pureModelResult: bool = Field(
        default=False,
        description="开启后在推理流程初始节点并行向 deepseek-v4-pro 发起一次纯大模型原生作答"
                    "（system=SUMMARY_SYSTEM_PROMPT，要求 ≤500 字 + 分「判断逻辑 / 关键证据 / 有效期限」三段），"
                    "并在 batch summary / final summary 阶段把该回答以「参考回答」小节形式注入到"
                    "用户问题下方，供推理模型在知识体系中摘取支撑 / 修正 / 冲突证据；"
                    "final 阶段会把冲突信息以「疑点」方式呈现给用户但不暴露外部模型来源。"
                    "外部请求 60s 内未返回 / 失败时自动降级为「无外部参考」。"
                    "submit / reason 两个入口均生效；仅在 version=v1/v2 下生效。",
    )
    answerRefine: bool = Field(
        default=True,
        description="启用答案精简：在整体推理流程最末一步对最终 answer 做"
                    "「结论先行 + 核心证据/因果逻辑/注意事项」结构化精简，"
                    "保留核心因果链与硬性限制条件、不引入新事实、不改变判断结论。"
                    "thinkMode=True 时：原完整 answer 内容会迁移到响应 think 字段，"
                    "精简结果写入响应 answer 字段；thinkMode=False 时直接覆盖响应 answer 字段。"
                    "与 cleanAnswer / summaryCleanAnswer / thinkMode / lastThink 完全正交。"
                    "仅在 version=v1/v2 下生效",
    )
    verbose: bool = Field(
        default=VERBOSE_DEFAULT_ENABLED,
        description="启用 verbose 模式：以当次请求的 taskId 为文件名，"
                    "完整记录该请求下根智能体 / 子智能体 / 汇总层每一步的 LLM 输入输出。"
                    "日志落盘在 <project>/verbose_logs/ 目录；当累计大小 ≥ 20MB 时，"
                    "自动打包为 archives/verbose_logs_<起始时间>__<结束时间>.zip 并清理原始文件。"
                    "未显式传入时默认跟随环境变量 VERBOSE_TRACE 的配置。",
    )
    sessionId: str | None = Field(
        default=None,
        description="兼容旧调用方的可选字段。当前 verbose 日志优先使用 taskId 命名，"
                    "本字段不再作为响应字段返回。",
    )


class ReasonData(BaseModel):
    khObj: str = Field(description="知识点名称到章节编号的 JSON 字符串映射")
    policyId: str
    answer: str = Field(
        description="最终面向用户的客服回答。"
                    "thinkMode=False 时：模型 *_AND_CLEAN_PROMPT 输出的完整客服回答（≤500 字）；"
                    "thinkMode=True 时：取自 LLM JSON 输出中的 answer 字段（面向用户的最终答案）。"
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
    taskId: str = Field(
        description="本次推理任务 ID。异步 submit/result 路径下与 /api/reason/submit 返回的 taskId 一致；"
                    "同步 /api/reason 路径下由服务端为本次调用即时生成。"
                    "配合 logName 可直接定位到 verbose_logs/ 下对应的 jsonl 日志文件",
    )
    logName: str | None = Field(
        default=None,
        description="verbose=True 时，本次 taskId 对应的日志文件名，格式 "
                    "<YYYYMMDD_HHMMSS>_<taskId>.jsonl，位于服务端 verbose_logs/ 目录下。"
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


class QueueEntry(BaseModel):
    taskId: str = Field(description="服务端为每个提交任务分配的 UUID；可用在 /api/reason/result/{taskId} 查询结果")
    question: str
    policyId: str
    state: str = Field(description='"queued"=还在 redis 队列里等 worker；"running"=已被 worker 拉起正在推理')
    enqueueTime: float = Field(description="进入 /api/reason(/submit) 的 epoch 秒（浮点）")
    startTime: float | None = Field(
        default=None,
        description="实际开始推理的 epoch 秒；queued 阶段为 null",
    )
    waitingSeconds: float = Field(description="已排队秒数：running 表示排队耗时，queued 表示当前已等待时长")
    runningSeconds: float | None = Field(
        default=None,
        description="running 状态下已推理秒数；queued 阶段为 null",
    )


class QueueStatusData(BaseModel):
    maxConcurrent: int = Field(description="内置 worker 数（对应环境变量 MAX_CONCURRENT_REASONING），即同时推理上限")
    runningCount: int
    queuedCount: int
    running: list[QueueEntry] = Field(description="按开始时间排序的正在推理任务")
    queued: list[QueueEntry] = Field(description="按入队先后排序的排队中任务（与 redis 队列 FIFO 一致）")


class QueueStatusResponse(BaseModel):
    data: QueueStatusData | None = None
    status_code: int
    message: str


class CleanQueueEntry(BaseModel):
    taskId: str
    removedFromQueue: int = Field(description="从 redis pending 队列中移除的同名 task_id 数量")
    deletedTask: bool = Field(description="是否删除了对应 task:{taskId} 记录")
    skippedReason: str | None = Field(
        default=None,
        description="未删除或清理异常时的原因；扫描期间被 worker 取走的任务会在这里说明",
    )


class CleanQueueData(BaseModel):
    removedQueueItems: int = Field(description="实际从 pending 队列中移除的队列项数量")
    deletedTaskCount: int = Field(description="成功删除 task:{taskId} 记录的任务数")
    skippedCount: int = Field(description="扫描到但未完成删除的任务数")
    tasks: list[CleanQueueEntry]


class CleanQueueResponse(BaseModel):
    data: CleanQueueData | None = None
    status_code: int
    message: str


# -------------------------------------------------- 手动清理超时 running 任务


class CleanupStaleRunningRequest(BaseModel):
    thresholdSeconds: float = Field(
        default=3600.0,
        ge=0.0,
        description=(
            "running_seconds（now - start_time）超过该阈值的任务视为超时需要清理；"
            "默认 3600（1 小时）。应略大于业务正常 P99 推理耗时，避免误杀长尾任务"
        ),
    )
    dryRun: bool = Field(
        default=False,
        description="True 时只返回命中候选，不写回 redis，便于先预览再决定是否执行真清理",
    )


class CleanupStaleRunningEntry(BaseModel):
    taskId: str
    policyId: str
    question: str
    runningSeconds: float = Field(description="截至扫描时的 running 耗时（秒）")
    startTime: float | None = None
    workerId: int | None = None
    instanceId: str | None = Field(
        default=None,
        description="该 running 记录写入时的进程 INSTANCE_ID；null 表示来自改造前版本",
    )
    matchesCurrentInstance: bool = Field(
        description=(
            "该记录的 instance_id 是否等于当前进程 INSTANCE_ID。"
            "True 代表可能是本进程 worker 真正还在跑的任务，清理后会让客户端"
            "立刻看到 failed，但同名 worker 仍可能晚一步把结果写回 done/failed（覆盖掉本次清理）。"
            "False 通常是历史进程残留的僵尸，清理是安全的。"
        ),
    )
    cleaned: bool = Field(description="是否成功写回 failed；dryRun=true 时恒为 false")
    skippedReason: str | None = Field(
        default=None,
        description="cleaned=False 时的跳过原因（例如扫描期间状态已变、写回异常等）",
    )


class CleanupStaleRunningData(BaseModel):
    thresholdSeconds: float
    dryRun: bool
    instanceId: str = Field(description="当前服务进程的 INSTANCE_ID，供比对 matchesCurrentInstance")
    matchedCount: int = Field(description="命中阈值的 running 任务数")
    cleanedCount: int = Field(description="实际写回 failed 成功的任务数（dryRun 恒为 0）")
    skippedCount: int = Field(description="命中但未清理的任务数（dryRun 下等于 matched；真清时通常为 race 跳过）")
    tasks: list[CleanupStaleRunningEntry] = Field(
        description="按 runningSeconds 降序排列（最可疑在前）的命中任务明细",
    )


class CleanupStaleRunningResponse(BaseModel):
    data: CleanupStaleRunningData | None = None
    status_code: int
    message: str


# ---------------------------------------------------- 异步化任务 submit / result


class ReasonSubmitData(BaseModel):
    taskId: str = Field(description="本次任务在服务端分配的 UUID；客户端据此轮询结果")
    status: str = Field(default=_tq.STATUS_PENDING, description="提交成功后固定为 pending")
    enqueueTime: float = Field(description="任务入队 epoch 秒")


class ReasonSubmitResponse(BaseModel):
    data: ReasonSubmitData | None = None
    status_code: int
    message: str


class ReasonResultData(BaseModel):
    taskId: str
    status: str = Field(
        description='任务状态：pending / running / done / failed'
    )
    enqueueTime: float
    startTime: float | None = Field(default=None, description="worker 拉起时间；未开始时 null")
    endTime: float | None = Field(default=None, description="结束时间；未完成时 null")
    result: "ReasonData | None" = Field(
        default=None,
        description="status=done 时为最终结果；其他状态为 null",
    )
    error: str | None = Field(
        default=None,
        description="status=failed 时的错误摘要；其他状态为 null",
    )


class ReasonResultResponse(BaseModel):
    data: ReasonResultData | None = None
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


# think_mode 下模型应输出形如 {"think": "...", "answer": "..."} 的合法 JSON，
# 历史字段名 `analysis`、`concise_answer` 仍兼容；响应体映射：
#   think  <- think（缺失时回落到 analysis）
#   answer <- answer（缺失时回落到 concise_answer）
# 模型未严格遵守 JSON 格式时，agent_graph 会先发起一次 HTML 标签格式重试再 coercion 兜底，
# 因此本函数会按顺序尝试 JSON / HTML / coercion 三种形态。
_JSON_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_HTML_THINK_BLOCK_RE = re.compile(r"<think>(?P<body>.*?)</think>", re.DOTALL | re.IGNORECASE)
_HTML_ANSWER_BLOCK_RE = re.compile(r"<answer>(?P<body>.*?)</answer>", re.DOTALL | re.IGNORECASE)


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


def _try_parse_think_answer_dict(raw: str) -> dict | None:
    """尝试把 raw 解析为 {think, answer} dict（兼容 analysis/concise_answer 旧字段）。

    顺序：原文 json/json5 → 剥 ```json``` 围栏 → 抓首个 {...} 子串。
    任意一关命中即返回；都失败返回 None。
    """
    if not raw:
        return None
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
    return parsed


def _try_parse_think_answer_html(raw: str) -> dict | None:
    """尝试把 raw 解析为 HTML 标签版 {think, answer}：
    - 必须有 <answer>...</answer>；
    - <think>...</think> 可选（无则 think 为空字符串，由调用方按字段缺失规则兜底）。
    """
    if not raw:
        return None
    am = _HTML_ANSWER_BLOCK_RE.search(raw)
    if not am:
        return None
    answer_text = (am.group("body") or "").strip()
    if not answer_text:
        return None
    tm = _HTML_THINK_BLOCK_RE.search(raw)
    think_text = (tm.group("body") or "").strip() if tm else ""
    return {"think": think_text, "answer": answer_text}


def _extract_think_answer_fields(parsed: dict) -> tuple[str, str]:
    """从 dict 中按字段优先级提取 think/answer 文本（兼容历史字段名）。"""
    think_val = parsed.get("think")
    if not (isinstance(think_val, str) and think_val.strip()):
        legacy_think = parsed.get("analysis")
        think_val = legacy_think if isinstance(legacy_think, str) else ""
    answer_val = parsed.get("answer")
    if not (isinstance(answer_val, str) and answer_val.strip()):
        legacy_answer = parsed.get("concise_answer")
        answer_val = legacy_answer if isinstance(legacy_answer, str) else ""
    think_text = think_val.strip() if isinstance(think_val, str) else ""
    answer_text = answer_val.strip() if isinstance(answer_val, str) else ""
    return think_text, answer_text


def _split_analysis_concise_answer(raw: str, think_mode: bool) -> tuple[str, str]:
    """解析最终节点 think_mode 下的模型输出，映射到 ReasonData.think / answer。

    上游 (reasoner.v2.agent_graph._chat_final_with_format_retry) 已保证 raw 形态属于
    以下三种之一：
      A. 标准 `{think, answer}` JSON（首轮成功）
      B. HTML 标签 `<think>...</think><answer>...</answer>`（首轮 JSON 失败，HTML 重试成功）
      C. coercion 兜底封装 `{"think": "<reasoning_content 或空>", "answer": "<原始 body>"}`

    本函数按以下顺序尝试解析：
      1. think_mode=False 或 raw 为空：直接 ("", raw or "")
      2. JSON dict 解析（兼容裸 JSON / ```json``` / 含解释文字 / json5 / 旧字段名 analysis、concise_answer）
      3. HTML 标签解析（<think>/<answer>，<think> 可选）
      4. 全失败：视为完全不遵循格式 → ("", raw)，并打 WARNING

    解析成功（拿到 think/answer 文本）后字段缺失兜底，按用户最终确认的策略：
      - 同时有 think 与 answer：分别映射
      - 只有 think（或 analysis）：think、answer 都填该内容（保证 answer 字段不空）
      - 只有 answer（或 concise_answer）：think="", answer=该值
      - 都为空：("", raw)，并打 WARNING
    """
    if not think_mode or not raw:
        return "", raw or ""

    parsed = _try_parse_think_answer_dict(raw)
    source: str
    think_text: str
    answer_text: str
    if parsed is not None:
        think_text, answer_text = _extract_think_answer_fields(parsed)
        source = "json"
    else:
        html_obj = _try_parse_think_answer_html(raw)
        if html_obj is not None:
            think_text = html_obj.get("think", "") or ""
            answer_text = html_obj.get("answer", "") or ""
            source = "html"
        else:
            logger.warning(
                "[ThinkMode] 模型输出无法解析为 JSON{think, answer} 或 HTML <think>/<answer>"
                "（json/json5/codeblock/regex/HTML 全部失败），"
                "本次 think 字段将返回空字符串，answer 字段保留模型原始输出"
            )
            return "", raw

    if not think_text and not answer_text:
        logger.warning(
            f"[ThinkMode] 模型 {source} 输出中未提取到任何有效字段 "
            f"(parsed={list(parsed.keys()) if parsed else 'html-empty'})，"
            "本次 think 字段将返回空字符串，answer 字段保留模型原始输出"
        )
        return "", raw

    if think_text and not answer_text:
        # 只有 think/analysis：两个字段都用同一份 think 内容，避免 answer 完全为空
        logger.warning(
            f"[ThinkMode] 模型 {source} 输出缺少 answer 字段，"
            "回落策略：think 与 answer 字段都使用 think 内容"
        )
        return think_text, think_text

    if answer_text and not think_text:
        # 只有 answer/concise_answer：think 留空
        logger.warning(
            f"[ThinkMode] 模型 {source} 输出缺少 think/analysis 字段，"
            "回落策略：think 字段填空字符串，answer 字段保留 answer 内容"
        )
        return "", answer_text

    return think_text, answer_text


def _import_agent_graph(version: str):
    if version == "v0":
        from reasoner.v0.agent_graph import AgentGraph
    elif version == "v2":
        from reasoner.v2.agent_graph import AgentGraph
    elif version == "v3":
        from reasoner.v3.agent_graph import AgentGraph
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
    pure_model_result: bool = False,
    answer_refine: bool = False,
) -> dict:
    """执行单问题推理，返回 answer 和 kh_obj"""
    AgentGraphCls = _import_agent_graph(version)
    extra_kwargs = {}
    if version in ("v1", "v2", "v3"):
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
        extra_kwargs["pure_model_result"] = pure_model_result
        extra_kwargs["answer_refine"] = answer_refine
    else:
        if summary_clean_answer:
            logger.warning("summaryCleanAnswer 仅在 version=v1/v2/v3 下生效，本次将被忽略")
        if answer_system_prompt:
            logger.warning("answerSystemPrompt 仅在 version=v1/v2/v3 下生效，本次将被忽略")
        if think_mode:
            logger.warning("thinkMode 仅在 version=v1/v2/v3 下生效，本次将被忽略")
        if enable_relations:
            logger.warning("enableRelations 仅在 version=v1/v2/v3 下生效，本次将被忽略")
        if pure_model_result:
            logger.warning("pureModelResult 仅在 version=v1/v2/v3 下生效，本次将被忽略")
        if answer_refine:
            logger.warning("answerRefine 仅在 version=v1/v2/v3 下生效，本次将被忽略")
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


# ---------------------------------------------------------------------- Executor


async def _reason_executor(request_payload: dict) -> dict:
    """WorkerPool 注入的业务执行器。

    输入：submit 阶段落到 redis 的 ``request`` 字段（原 ReasonRequest 的 JSON 表示，
    字段名为 camelCase，见 ReasonRequest 定义）。
    输出：与 ReasonData 字段一致的 dict，便于 result / 同步包装接口直接构造 ReasonData。

    异常会被 WorkerPool 捕获并写入 task.error / status=failed，这里只管正常路径抛干净的异常。
    """
    policy_id = request_payload["policyId"]
    question = request_payload["question"]
    task_id = request_payload.get("taskId") or str(uuid.uuid4())
    req_verbose = bool(request_payload.get("verbose", VERBOSE_DEFAULT_ENABLED))

    session_meta = {
        "endpoint": "/api/reason/submit",
        "taskId": task_id,
        "policyId": policy_id,
        "question": question,
        "version": request_payload.get("version", "v1"),
        "vendor": request_payload.get("vendor"),
        "model": request_payload.get("model"),
        "retrievalMode": request_payload.get("retrievalMode"),
        "chunkSize": request_payload.get("chunkSize"),
        "summaryBatchSize": request_payload.get("summaryBatchSize"),
        "enableRelations": request_payload.get("enableRelations"),
        "thinkMode": request_payload.get("thinkMode"),
        "pureModelResult": request_payload.get("pureModelResult"),
        "answerRefine": request_payload.get("answerRefine"),
    }
    with _open_verbose_session(
        session_id=task_id,
        meta=session_meta,
        enabled=req_verbose,
    ) as _session:
        resp_log_name = _session.file_path.name if _session is not None else None

        knowledge_dir = await asyncio.to_thread(_get_or_extract_knowledge, policy_id)
        if req_verbose:
            _log_verbose_event(
                "knowledge_ready",
                policyId=policy_id,
                knowledge_dir=knowledge_dir,
            )

        result = await asyncio.to_thread(
            _run_reasoning, question, knowledge_dir,
            version=request_payload.get("version", "v1"),
            max_rounds=request_payload.get("maxRounds", 10),
            vendor=request_payload.get("vendor", "aliyun"),
            model=request_payload.get("model", "deepseek-v3.2"),
            clean_answer=request_payload.get("cleanAnswer", False),
            summary_batch_size=request_payload.get("summaryBatchSize", 3),
            retrieval_mode=request_payload.get("retrievalMode", True),
            check_pitfalls=request_payload.get("checkPitfalls", True),
            chunk_size=request_payload.get("chunkSize", 3000),
            enable_skills=request_payload.get("enableSkills", True),
            summary_clean_answer=request_payload.get("summaryCleanAnswer", True),
            answer_system_prompt=request_payload.get("answerSystemPrompt"),
            think_mode=request_payload.get("thinkMode", True),
            last_think=request_payload.get("lastThink", True),
            enable_relations=request_payload.get("enableRelations", True),
            relation_max_depth=request_payload.get("relationMaxDepth", 5),
            relation_max_nodes=request_payload.get("relationMaxNodes", 999),
            relation_workers=request_payload.get("relationWorkers", 8),
            relations_expansion_mode=request_payload.get("relationsExpansionMode", "all"),
            summary_pipeline_mode=request_payload.get("summaryPipelineMode", "layered"),
            reduce_max_part_depth=request_payload.get("reduceMaxPartDepth", 4),
            pure_model_result=request_payload.get("pureModelResult", False),
            answer_refine=request_payload.get("answerRefine", True),
        )

        if req_verbose:
            _log_verbose_event(
                "reason_finished",
                answer_preview=(result.get("answer") or "")[:200],
                kh_obj=result.get("kh_obj", {}),
            )

        # 字段名对齐 ReasonData（camelCase）。result / 同步包装接口据此直接构造。
        return {
            "khObj": json.dumps(result["kh_obj"], ensure_ascii=False),
            "answer": result["answer"],
            "think": result.get("think", ""),
            "skillsResult": result.get("skills_result", {}),
            "policyId": policy_id,
            "taskId": task_id,
            "logName": resp_log_name,
        }


# ----------------------------------------------------------- 异步提交 / 结果查询


def _task_to_result_data(task: dict) -> ReasonResultData:
    r = task.get("result") or None
    if r and not r.get("taskId"):
        r = {**r, "taskId": task["task_id"]}
    reason_data = ReasonData(**r) if r else None
    return ReasonResultData(
        taskId=task["task_id"],
        status=task["status"],
        enqueueTime=task["enqueue_time"],
        startTime=task.get("start_time"),
        endTime=task.get("end_time"),
        result=reason_data,
        error=task.get("error"),
    )


@app.post("/api/reason/submit", response_model=ReasonSubmitResponse)
async def reason_submit(req: ReasonRequest):
    """立刻返回 task_id 的异步提交接口；实际推理由 worker pool 异步完成。"""
    client = _require_redis()
    try:
        task = await _tq.submit_task(
            client,
            request_payload=req.model_dump(mode="json"),
            ttl_seconds=REASON_TASK_TTL_SECONDS,
        )
        return ReasonSubmitResponse(
            data=ReasonSubmitData(
                taskId=task["task_id"],
                status=task["status"],
                enqueueTime=task["enqueue_time"],
            ),
            status_code=200,
            message="success",
        )
    except Exception as e:
        logger.exception(f"/api/reason/submit 失败: {e}")
        return ReasonSubmitResponse(
            data=None,
            status_code=500,
            message=f"提交失败: {e}",
        )


@app.get("/api/reason/result/{task_id}", response_model=ReasonResultResponse)
async def reason_result(task_id: str):
    """按 task_id 查询任务状态 / 结果。调用方自行轮询（推荐 1~2s 一次）。

    状态语义：
      - pending：已入队、等待 worker；
      - running：worker 已拉起、正在推理；
      - done   ：推理完成，result 字段可读；
      - failed ：推理失败，error 字段给出摘要；
      - 任务不存在：task_id 无效或已过期（默认 24h 后回收）→ 返回 404。
    """
    client = _require_redis()
    task = await _tq.get_task(client, task_id)
    if task is None:
        return ReasonResultResponse(
            data=None,
            status_code=404,
            message=f"task_id={task_id} 不存在或已过期",
        )
    return ReasonResultResponse(
        data=_task_to_result_data(task),
        status_code=200,
        message="success",
    )


#: /api/reason 单次最长执行时间：按"自测接口、优先级最高"的定位给到 1 天。
#: 超过后 asyncio.wait_for 会抛 TimeoutError，按普通失败路径返回。
REASON_IMMEDIATE_TIMEOUT_SECONDS = float(
    os.environ.get("REASON_IMMEDIATE_TIMEOUT_SECONDS", str(24 * 3600))
)


@app.post("/api/reason", response_model=ReasonResponse)
async def reason(req: ReasonRequest):
    """自测专用同步接口：**不进队列、立即执行**，优先级高于所有 queued / running 任务。

    - 不调用 submit_task / 不写 task:{id} / 不占用 WorkerPool 的 worker 槽位，
      因此不会被 MAX_CONCURRENT_REASONING 限流，也不会被前面排队的任务阻塞；
    - 直接在本请求协程内 await 业务执行器，同一套逻辑、同一套返回字段；
    - 超时按 1 天算（REASON_IMMEDIATE_TIMEOUT_SECONDS），到点视为失败。

    注意：本接口定位是"开发者自测 / 手动触发"。生产批量调用仍应使用
    /api/reason/submit + /api/reason/result/{taskId} 的异步路径。
    """
    _require_redis()  # 仅做存活校验，verbose 日志等仍依赖 redis。

    request_payload = req.model_dump(mode="json")
    request_payload["taskId"] = str(uuid.uuid4())
    try:
        result = await asyncio.wait_for(
            _reason_executor(request_payload),
            timeout=REASON_IMMEDIATE_TIMEOUT_SECONDS,
        )
        return ReasonResponse(
            data=ReasonData(**result),
            status_code=200,
            message="success",
        )
    except asyncio.CancelledError:
        # 客户端主动断开：本接口没有把任务落到 redis，断了就是断了，不再兜底。
        logger.info("/api/reason 客户端提前断开，立即执行路径随之取消")
        raise
    except asyncio.TimeoutError:
        logger.error(
            f"/api/reason 立即执行超过 {REASON_IMMEDIATE_TIMEOUT_SECONDS}s，按失败返回"
        )
        return ReasonResponse(
            data=ReasonData(
                khObj="{}",
                answer=f"推理失败: 超过 {int(REASON_IMMEDIATE_TIMEOUT_SECONDS)}s 超时上限",
                think="",
                skillsResult={},
                policyId=req.policyId,
                taskId=request_payload["taskId"],
                logName=None,
            ),
            status_code=500,
            message="推理超时",
        )
    except Exception as e:
        logger.exception("/api/reason 立即执行失败")
        return ReasonResponse(
            data=ReasonData(
                khObj="{}",
                answer=f"推理失败: {type(e).__name__}: {e}",
                think="",
                skillsResult={},
                policyId=req.policyId,
                taskId=request_payload["taskId"],
                logName=None,
            ),
            status_code=500,
            message=f"推理失败: {type(e).__name__}: {e}",
        )


@app.get("/api/requestQueueStatus", response_model=QueueStatusResponse)
async def request_queue_status():
    """查询当前推理任务的队列状态（从 redis_server 实时读取）。

    - running：worker 已拉起、正在推理的任务；
    - queued ：仍停留在 redis 队列 ``queue:reason:pending`` 里、还没 pop 的任务；
    两组都按"先提交先排前"的顺序返回（running 按 start_time，queued 按入队顺序）。
    """
    try:
        client = _require_redis()
        now = time.time()
        queued_tasks, running_tasks = await asyncio.gather(
            _tq.list_queued_tasks(client),
            _tq.list_running_tasks(client),
        )

        def _to_entry(t: dict, state: str) -> QueueEntry:
            enqueue_time = float(t["enqueue_time"])
            start_time = t.get("start_time")
            if state == "running" and start_time is not None:
                waiting = max(0.0, float(start_time) - enqueue_time)
                running_secs = max(0.0, now - float(start_time))
            else:
                waiting = max(0.0, now - enqueue_time)
                running_secs = None
            req = t.get("request") or {}
            return QueueEntry(
                taskId=t["task_id"],
                question=req.get("question", ""),
                policyId=req.get("policyId", ""),
                state=state,
                enqueueTime=enqueue_time,
                startTime=start_time,
                waitingSeconds=round(waiting, 3),
                runningSeconds=round(running_secs, 3) if running_secs is not None else None,
            )

        queued = [_to_entry(t, "queued") for t in queued_tasks]
        running = [_to_entry(t, "running") for t in running_tasks]

        return QueueStatusResponse(
            data=QueueStatusData(
                maxConcurrent=MAX_CONCURRENT_REASONING,
                runningCount=len(running),
                queuedCount=len(queued),
                running=running,
                queued=queued,
            ),
            status_code=200,
            message="success",
        )
    except Exception as e:
        logger.exception(f"查询请求队列状态失败: {e}")
        return QueueStatusResponse(
            data=None,
            status_code=500,
            message=f"查询请求队列状态失败: {str(e)}",
        )


@app.post("/api/cleanQueue", response_model=CleanQueueResponse)
async def clean_queue():
    """清除 redis 中所有仍在 pending 队列里的任务。

    只处理还停留在 ``queue:reason:pending`` 的任务；已经被 worker 取走的 running 任务
    不会被删除。如需清理卡住的 running 任务，使用 ``/api/reason/cleanupStaleRunning``。
    """
    try:
        client = _require_redis()
        entries = await _tq.clean_queued_tasks(client)
        removed = sum(int(e.get("removed_from_queue") or 0) for e in entries)
        deleted = sum(1 for e in entries if e.get("deleted_task"))
        skipped = sum(
            1
            for e in entries
            if not e.get("deleted_task") or e.get("skipped_reason")
        )

        tasks = [
            CleanQueueEntry(
                taskId=str(e.get("task_id", "")),
                removedFromQueue=int(e.get("removed_from_queue") or 0),
                deletedTask=bool(e.get("deleted_task")),
                skippedReason=e.get("skipped_reason"),
            )
            for e in entries
        ]

        logger.info(
            f"[/api/cleanQueue] removed_queue_items={removed} "
            f"deleted_tasks={deleted} skipped={skipped}"
        )
        return CleanQueueResponse(
            data=CleanQueueData(
                removedQueueItems=removed,
                deletedTaskCount=deleted,
                skippedCount=skipped,
                tasks=tasks,
            ),
            status_code=200,
            message="success",
        )
    except Exception as e:
        logger.exception(f"清理 pending 队列失败: {e}")
        return CleanQueueResponse(
            data=None,
            status_code=500,
            message=f"清理 pending 队列失败: {str(e)}",
        )


@app.post(
    "/api/reason/cleanupStaleRunning",
    response_model=CleanupStaleRunningResponse,
)
async def cleanup_stale_running(req: CleanupStaleRunningRequest):
    """手动清理 redis 中 running 耗时超过阈值的卡死任务。

    用途
    ----
    - 服务**未重启**、某些 task 因下游长尾或逻辑卡死持续停在 running,
      占满 ``runningCount`` 让新任务排不到 worker；
    - 想让轮询 ``/api/reason/result/{taskId}`` 的客户端立刻感知到 failed。
    - 启动期遗留的僵尸 running 由 lifespan 自动清理；本接口作为运行时兜底。

    行为
    ----
    1. 扫描 redis 的 ``task:*``；
    2. 挑出 ``status=running`` 且 ``now - start_time > thresholdSeconds`` 的任务；
    3. 写回前做一次二次 GET 校验，若 worker 刚好完成已变 done/failed 则跳过;
    4. 命中项改写为 ``status=failed``、``error="manually cleaned: running over <N>s"``、
       ``end_time=now``，供客户端轮询立刻感知。

    重要限制
    --------
    - 只动 redis，**不会** cancel 仍在跑的 worker 协程；若 worker 真在跑，
      写回 done 的时间可能晚于本次清理，客户端可能看到 failed→done 跳变。
    - 因此建议先用 ``dryRun=true`` 预览、对照 ``matchesCurrentInstance``:
      False 的条目（上一代进程僵尸）清理是安全的；True 的条目是本进程 worker,
      请结合 runningSeconds 判断是否真的卡死。
    - 判据是时间阈值，与 ``instance_id`` 无关；与 lifespan 的启动清理互补。
    """
    try:
        client = _require_redis()
        entries = await _tq.cleanup_running_tasks_by_age(
            client,
            threshold_seconds=req.thresholdSeconds,
            current_instance_id=INSTANCE_ID,
            dry_run=req.dryRun,
            ttl_seconds=REASON_TASK_TTL_SECONDS,
        )
        matched = len(entries)
        cleaned = sum(1 for e in entries if e.get("cleaned"))
        skipped = matched - cleaned

        tasks_out: list[CleanupStaleRunningEntry] = []
        for e in entries:
            worker_id_raw = e.get("worker_id")
            worker_id: int | None
            try:
                worker_id = int(worker_id_raw) if worker_id_raw is not None else None
            except (TypeError, ValueError):
                worker_id = None
            tasks_out.append(
                CleanupStaleRunningEntry(
                    taskId=str(e.get("task_id", "")),
                    policyId=str(e.get("policy_id", "")),
                    question=str(e.get("question", "")),
                    runningSeconds=float(e.get("running_seconds", 0.0)),
                    startTime=e.get("start_time"),
                    workerId=worker_id,
                    instanceId=e.get("instance_id"),
                    matchesCurrentInstance=bool(e.get("matches_current_instance")),
                    cleaned=bool(e.get("cleaned")),
                    skippedReason=e.get("skipped_reason"),
                )
            )

        logger.info(
            f"[/api/reason/cleanupStaleRunning] threshold={req.thresholdSeconds}s "
            f"dry_run={req.dryRun} matched={matched} cleaned={cleaned} skipped={skipped}"
        )

        return CleanupStaleRunningResponse(
            data=CleanupStaleRunningData(
                thresholdSeconds=req.thresholdSeconds,
                dryRun=req.dryRun,
                instanceId=INSTANCE_ID,
                matchedCount=matched,
                cleanedCount=cleaned,
                skippedCount=skipped,
                tasks=tasks_out,
            ),
            status_code=200,
            message="success",
        )
    except Exception as e:
        logger.exception(f"清理超时 running 任务失败: {e}")
        return CleanupStaleRunningResponse(
            data=None,
            status_code=500,
            message=f"清理超时 running 任务失败: {str(e)}",
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

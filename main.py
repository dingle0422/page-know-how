import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_extract(args):
    has_input = bool(getattr(args, "input", None))
    has_policy = bool(getattr(args, "policy_id", None))

    if has_input and has_policy:
        print("错误：--input 与 --policy-id 不能同时使用")
        raise SystemExit(1)
    if not has_input and not has_policy:
        print("错误：必须指定 --input 或 --policy-id 之一")
        raise SystemExit(1)

    if has_policy:
        from extractor.builder import extract_from_api
        result_dir = extract_from_api(args.policy_id, getattr(args, "api_url", None))
    else:
        from extractor.builder import extract
        result_dir = extract(args.input)

    print(f"\n知识抽取完成！")
    print(f"输出目录: {result_dir}")


def _import_engine(version: str):
    """根据版本号动态导入 engine 模块"""
    if version == "v0":
        from reasoner.v0.engine import run_single_question, run_reasoning
    elif version == "v2":
        from reasoner.v2.engine import run_single_question, run_reasoning
    else:
        from reasoner.v1.engine import run_single_question, run_reasoning
    return run_single_question, run_reasoning


def cmd_reason(args):
    has_single = bool(getattr(args, "single_question", None))
    has_batch = bool(getattr(args, "questions", None))

    if has_single and has_batch:
        print("错误：--single-question 与 --questions 不能同时使用")
        raise SystemExit(1)
    if not has_single and not has_batch:
        print("错误：必须指定 --single-question 或 --questions 之一")
        raise SystemExit(1)

    has_knowledge_dir = bool(getattr(args, "knowledge_dir", None))
    has_policy_id = bool(getattr(args, "policy_id", None))

    if has_knowledge_dir and has_policy_id:
        print("错误：--knowledge-dir 与 --policy-id 不能同时使用")
        raise SystemExit(1)
    if not has_knowledge_dir and not has_policy_id:
        print("错误：必须指定 --knowledge-dir 或 --policy-id 之一")
        raise SystemExit(1)

    if has_policy_id:
        from extractor.builder import extract_from_api
        print(f"通过 policy-id 抽取知识目录: {args.policy_id}")
        args.knowledge_dir = extract_from_api(args.policy_id, getattr(args, "api_url", None))
        print(f"知识目录抽取完成: {args.knowledge_dir}")

    version = getattr(args, "version", "v1")
    run_single_question, run_reasoning = _import_engine(version)
    print(f"使用 reasoner {version} 版本")

    enable_skills = not getattr(args, "disable_skills", False)
    common_kwargs = dict(
        knowledge_dir=args.knowledge_dir,
        max_rounds=args.max_rounds,
        vendor=args.vendor,
        model=args.model,
        clean_answer=args.clean_answer,
        summary_batch_size=args.summary_batch_size,
        retrieval_mode=args.retrieval_mode,
        check_pitfalls=args.check_pitfalls,
        enable_skills=enable_skills,
        last_think=args.last_think,
    )
    if version in ("v1", "v2"):
        common_kwargs["chunk_size"] = args.chunk_size
        common_kwargs["summary_clean_answer"] = args.summary_clean_answer
        common_kwargs["think_mode"] = args.think_mode
        common_kwargs["answer_system_prompt"] = args.answer_system_prompt
        common_kwargs["enable_relations"] = args.enable_relations
        common_kwargs["relation_max_depth"] = args.relation_max_depth
        common_kwargs["relation_max_nodes"] = args.relation_max_nodes
        common_kwargs["relation_workers"] = args.relation_workers
        common_kwargs["relations_expansion_mode"] = args.relations_expansion_mode
        common_kwargs["summary_pipeline_mode"] = args.summary_pipeline_mode
        common_kwargs["reduce_max_part_depth"] = args.reduce_max_part_depth
    else:
        if args.summary_clean_answer:
            print("警告：--summary-clean-answer 仅在 --version v1/v2 下生效，本次将被忽略")
        if args.think_mode:
            print("警告：--think-mode 仅在 --version v1/v2 下生效，本次将被忽略")
        if args.answer_system_prompt:
            print("警告：--answer-system-prompt 仅在 --version v1/v2 下生效，本次将被忽略")
        if args.enable_relations:
            print("警告：--enable-relations 仅在 --version v1/v2 下生效，本次将被忽略")

    verbose_trace = getattr(args, "verbose_trace", False)
    session_id = getattr(args, "session_id", None)
    session_meta = {
        "entrypoint": "cli",
        "command": "reason",
        "version": version,
        "policyId": getattr(args, "policy_id", None),
        "knowledgeDir": args.knowledge_dir,
        "vendor": args.vendor,
        "model": args.model,
        "mode": "single" if has_single else "batch",
    }

    from utils.verbose_logger import open_session as _open_verbose_session
    with _open_verbose_session(session_id=session_id, meta=session_meta, enabled=verbose_trace):
        if has_single:
            run_single_question(question=args.single_question, **common_kwargs)
        else:
            if not args.question_column:
                print("错误：使用 --questions 时必须同时指定 --question-column")
                raise SystemExit(1)
            output_path = run_reasoning(
                questions_file=args.questions,
                question_column=args.question_column,
                output_path=args.output,
                max_workers=args.max_workers,
                **common_kwargs,
            )
            print(f"\n推理完成！")
            print(f"结果文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="文档知识结构化抽取 + 渐进式披露推理框架"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="启用详细日志")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # extract 子命令
    extract_parser = subparsers.add_parser("extract", help="从文档抽取知识并构建目录结构")
    extract_parser.add_argument(
        "--input", "-i",
        help="输入文档路径（支持 .docx、.txt 和 .json clause_list 格式，与 --policy-id 互斥）"
    )
    extract_parser.add_argument(
        "--policy-id", "-p",
        help="从 API 接口获取条款数据的 policyId（与 --input 互斥）"
    )
    extract_parser.add_argument(
        "--api-url",
        help="API 接口地址（仅配合 --policy-id 使用，默认为内置地址）"
    )

    # reason 子命令
    reason_parser = subparsers.add_parser("reason", help="基于知识目录进行推理")
    reason_parser.add_argument(
        "--single-question", "-s",
        help="直接传入单个问题字符串进行推理（与 --questions 互斥）"
    )
    reason_parser.add_argument(
        "--questions", "-q",
        help="问题集文件路径（支持 .csv 和 .xlsx，与 --single-question 互斥）"
    )
    reason_parser.add_argument(
        "--question-column", "-c",
        help="问题集中问题所在列的名称（使用 --questions 时必填）"
    )
    reason_parser.add_argument(
        "--knowledge-dir", "-k",
        help="page_knowledge 下的独立知识目录路径（与 --policy-id 互斥）"
    )
    reason_parser.add_argument(
        "--policy-id", "-p",
        help="通过 policyId 从 API 获取条款并自动抽取知识目录（与 --knowledge-dir 互斥）"
    )
    reason_parser.add_argument(
        "--api-url",
        help="API 接口地址（仅配合 --policy-id 使用，默认为内置地址）"
    )
    reason_parser.add_argument(
        "--output", "-o",
        help="结果输出文件路径（.xlsx）。若文件已存在则自动断点续跑，跳过已成功的问题"
    )
    reason_parser.add_argument(
        "--max-rounds", "-r", type=int, default=5,
        help="每个子智能体的最大 ReAct 轮次（默认 5）"
    )
    reason_parser.add_argument(
        "--vendor", default="qwen3.5-122b-a10b",
        # choices=["aliyun", "servyou", "qwen3.5-122b-a10b", "qwen3.5-27b"],
        help="LLM 供应商（默认 qwen3.5-122b-a10b 直连）"
    )
    reason_parser.add_argument(
        "--model", default="Qwen3.5-122B-A10B",
        help="LLM 模型名称（默认 Qwen3.5-122B-A10B）"
    )
    reason_parser.add_argument(
        "--max-workers", "-w", type=int, default=1,
        help="问题粒度的最大并行推理 worker 数量（默认 1，即串行）"
    )
    reason_parser.add_argument(
        "--clean-answer", action="store_true", default=False,
        help="启用答案清洗：在 summary 后追加一轮 LLM 调用，以咨询客服口吻输出精简结论"
    )
    reason_parser.add_argument(
        "--summary-batch-size", type=int, default=0,
        help="分批并行压缩总结：指定每批包含的证据条数（如 3），启用后自动激活分层总结模式。默认 0 表示不分批"
    )
    reason_parser.add_argument(
        "--retrieval-mode", action="store_true", default=False,
        help="启用召回模式：子智能体仅做相关性判定并收集原始知识，避免探索阶段信息畸变"
    )
    reason_parser.add_argument(
        "--check-pitfalls", action="store_true", default=False,
        help="启用易错点检查：在总结后追加一轮 LLM 调用，基于探索中收集的易错点对答案做逐条校验"
    )
    reason_parser.add_argument(
        "--chunk-size", type=int, default=0,
        help="启用知识分块模式：按指定字符数上限对知识目录树进行程序化分块，"
             "每个块并行推理后汇总。默认 0 表示不启用（使用原有 ReactAgent 探索模式）"
    )
    reason_parser.add_argument(
        "--version", default="v1", choices=["v0", "v1", "v2"],
        help="推理引擎版本（v0=原始版本, v1=统一EXPLORE+三层目录树, v2=KV-cache 优化 prompt, 默认 v1）"
    )
    reason_parser.add_argument(
        "--disable-skills", action="store_true", default=False,
        help="关闭 skill 功能：默认开启，会在推理前对问题做 skill 评估，并在 summary 后做 skill double-check 优化答案"
    )
    reason_parser.add_argument(
        "--summary-clean-answer", action="store_true", default=False,
        help="启用 summary+clean 一体化（仅 v1 生效，与所有 summary 模式兼容："
             "标准/召回 × 单次/分批 × chunk）。在最终 summary 或 batch_merge 阶段"
             "直接输出面向用户的客服话术答案，跳过独立的 clean-answer 调用以减少一次 LLM 串行延迟。"
             "开启后即覆盖原 --clean-answer 的清洗效果，无需再额外开 --clean-answer"
    )
    reason_parser.add_argument(
        "--think-mode", action="store_true", default=True,
        help="启用 think 模式（仅 v1 生效，需配合 --summary-clean-answer 使用）："
             "在【所有最终节点】的 summary+clean 阶段，改用 *_AND_CLEAN_THINK 版 prompt，"
             "要求模型严格按 JSON {\"analysis\": \"...\", \"concise_answer\": \"...\"} 输出。"
             "字段语义：analysis=完整客服回答（≤500 字等所有硬约束），"
             "concise_answer=核心结论一句话；解析后映射到响应体 think<-analysis、answer<-concise_answer。"
             "覆盖范围不受分批/召回/chunk 影响（非分批 SUMMARY_AND_CLEAN、"
             "分批 BATCH_MERGE_AND_CLEAN，及其 RETRIEVAL_* 对应版本均会切换）；"
             "中间提炼 prompt 始终保持原样不动"
    )
    reason_parser.add_argument(
        "--last-think", action="store_true", default=False,
        help="在【全流程最后一步总结/清洗】阶段打开底层 LLM 的 enable_thinking=True，"
             "让模型返回推理轨迹（qwen3.5/3.6 会把 <think>...</think> 写进 content；"
             "deepseek-reasoner / deepseek-v3.2 等会返回到 message.reasoning_content，"
             "已在 llm/client.py 统一前缀回注到 content）。"
             "只作用于最终节点的 chat 调用（all_in_answer / final_summary / "
             "batch_final_merge / retrieval_final_summary / retrieval_batch_final_merge / "
             "clean_answer），中间 batch/chunk/探索阶段都不受影响。"
             "与 --think-mode 正交：think-mode 只改 prompt 模板（要求 JSON 输出结构），"
             "last-think 只改 chat_template_kwargs.enable_thinking（开启模型推理轨迹）"
    )
    reason_parser.add_argument(
        "--answer-system-prompt", default=None,
        help="最终作答阶段的 system prompt 自定义内容（仅 v1 生效）。"
             "不传或传空字符串则使用内置的 SUMMARY_ANSWER_SYSTEM_PROMPT 默认版本；"
             "中间提炼层仍固定使用 SUMMARY_EXTRACT_SYSTEM_PROMPT，不受此参数影响"
    )
    reason_parser.add_argument(
        "--enable-relations", action="store_true", default=False,
        help="启用关联条款展开（仅 v1 生效）：当 chunk/子智能体命中相关知识且其目录包含 "
             "clause.json 中的预展开 references 时，按 LLM 多跳并发拉取外部条款；"
             "chunk 模式下切分为派生 chunk 与原 chunk 一并进入流式 batch summary，"
             "standard/retrieval 模式下 inline 追加到对应 fragment 末尾。"
             "本地缺失时自动通过 DEFAULT_CLAUSE_API_URL 实时拉取"
    )
    reason_parser.add_argument(
        "--relation-max-depth", type=int, default=5,
        help="关联展开最大跳深（含首跳）。--enable-relations=False 时忽略。默认 5"
    )
    reason_parser.add_argument(
        "--relation-max-nodes", type=int, default=50,
        help="单次 chunk/子智能体触发的关联展开 BFS 总节点数上限，超出即停止扩展。默认 50"
    )
    reason_parser.add_argument(
        "--relation-workers", type=int, default=8,
        help="关联展开的调度线程数；候选评估池容量为本值的 2 倍。默认 8"
    )
    reason_parser.add_argument(
        "--relations-expansion-mode", default="all", choices=["all", "smart"],
        help="关联展开模式（--enable-relations=True 时生效）：\n"
             "'all'（默认）：跳过对每个候选条款的 LLM 二次判定，"
             "只要 ClauseLocator 能定位到内容，一律入 RelationFragment 并按深度全展开。"
             "触发率 100%%、省掉 N 次 LLM 评估调用反而更快，token 略增；"
             "外层闸（chunk LLM relevant_headings 命中）仍然保留。\n"
             "'smart'：每个候选条款都用 LLM 判 is_relevant，准确率高但触发率受 LLM "
             "采样波动影响（同一问题多次跑可能命中数不一致）"
    )
    reason_parser.add_argument(
        "--summary-pipeline-mode", default="reduce_queue", choices=["layered", "reduce_queue"],
        help="batch summary 流水线模式（--summary-batch-size > 0 时生效）：\n"
             "'layered'：chunk + 关联展开走 _chunk_streaming_pipeline 的"
             "'按 slot 顺序流式 batch + 后续递归压缩按层同步'；其他入口走 "
             "_recursive_batch_reduce 同步分层。\n"
             "'reduce_queue'（默认）：所有压缩任务统一进 ReducePipeline，凑批+回灌，"
             "无层间同步点。chunk 数大、batch 长尾差异显著时可见明显加速；"
             "代价是早 flush 的 part 经过更多次中间压缩"
    )
    reason_parser.add_argument(
        "--reduce-max-part-depth", type=int, default=4,
        help="reduce_queue 模式下单 part 经过的最大中间 BATCH_SUMMARY 次数。"
             "命中上限的 part 不再参与凑批，转入 frozen 列表直接保留到 final merge。"
             "--summary-pipeline-mode=layered 时本字段忽略。默认 4"
    )
    from utils.verbose_logger import VERBOSE_DEFAULT_ENABLED as _VT_DEFAULT
    reason_parser.add_argument(
        "--verbose-trace", action="store_true", default=_VT_DEFAULT,
        help="启用 verbose 模式：以当次请求的 session 编号为文件名，"
             "完整记录根智能体/子智能体/汇总层每一步的 LLM 输入输出。"
             "日志落盘在 <project>/verbose_logs/ 目录；当累计大小 ≥ 20MB 时，"
             "自动打包为 archives/verbose_logs_<起>__<止>.zip 并清理原始文件。"
             "未显式传入时默认跟随环境变量 VERBOSE_TRACE 的配置。"
             "注意：与顶层日志级别 --verbose/-v 是两个不同的开关"
    )
    reason_parser.add_argument(
        "--session-id", default=None,
        help="可选：显式指定 verbose session 编号（便于跨服务日志串联）；"
             "不传则自动生成 sess-<随机>。仅在 --verbose-trace 开启时生效"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "reason":
        cmd_reason(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

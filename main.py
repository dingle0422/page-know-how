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
    )
    if version == "v1":
        common_kwargs["chunk_size"] = args.chunk_size
        common_kwargs["summary_clean_answer"] = args.summary_clean_answer
    elif args.summary_clean_answer:
        print("警告：--summary-clean-answer 仅在 --version v1 下生效，本次将被忽略")

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
        "--vendor", default="aliyun",
        # choices=["aliyun", "servyou"],
        help="LLM 供应商（默认 aliyun）"
    )
    reason_parser.add_argument(
        "--model", default="deepseek-v3.2",
        help="LLM 模型名称（默认 deepseek-v3.2）"
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
        "--version", default="v1", choices=["v0", "v1"],
        help="推理引擎版本（v0=原始版本, v1=统一EXPLORE+三层目录树，v1探索太激进建议开启召回模式, 默认 v1）"
    )
    reason_parser.add_argument(
        "--disable-skills", action="store_true", default=False,
        help="关闭 skill 功能：默认开启，会在推理前对问题做 skill 评估，并在 summary 后做 skill double-check 优化答案"
    )
    reason_parser.add_argument(
        "--summary-clean-answer", action="store_true", default=False,
        help="启用 summary+clean 一体化（仅 v1 + retrieval-mode + summary-batch-size>0 时生效）："
             "在 retrieval 分批合并阶段直接输出面向用户的客服话术答案，跳过独立的 clean-answer 调用以减少一次 LLM 串行延迟"
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

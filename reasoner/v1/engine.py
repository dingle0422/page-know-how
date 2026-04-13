import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from reasoner.v1.agent_graph import AgentGraph

logger = logging.getLogger(__name__)

TRACE_SEPARATOR = "\n\n====== 智能体探索追踪 ======\n\n"


def load_questions(filepath: str, column_name: str) -> list[tuple[int, str]]:
    """从 csv 或 xlsx 文件加载问题列表，返回 [(原始行号, 问题), ...]。行号从 1 开始。"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"不支持的问题集文件类型: {ext}，仅支持 .csv 和 .xlsx")

    if column_name not in df.columns:
        raise ValueError(
            f"列名 '{column_name}' 不存在于问题集中。"
            f"可用列名: {list(df.columns)}"
        )

    series = df[column_name].dropna().astype(str)
    questions = [(int(idx) + 1, text) for idx, text in series.items()]
    logger.info(f"加载了 {len(questions)} 个问题")
    return questions


def _sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """将文本列中的换行符替换为可控标记，确保 CSV 每条记录只占一行。"""
    text_cols = ["答案", "完整输出（含追踪）"]
    df = df.copy()
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("\r\n", "\\n", regex=False)
            df[col] = df[col].astype(str).str.replace("\n", "\\n", regex=False)
    return df


def _flush_results(results: list[dict], output_path: str) -> None:
    """原子写入：先写临时文件再重命名，防止写入中途崩溃导致文件损坏。"""
    base, ext = os.path.splitext(output_path)
    tmp_path = base + "_tmp" + ext
    try:
        df = pd.DataFrame(results)
        if ext.lower() in (".xlsx", ".xls"):
            df.to_excel(tmp_path, index=False, engine="openpyxl")
        else:
            df = _sanitize_for_csv(df)
            df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        os.replace(tmp_path, output_path)
    except Exception as e:
        logger.error(f"结果文件写入失败: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_existing_results(output_path: str) -> tuple[list[dict], set[str]]:
    """从已有结果文件加载已完成的记录，用于断点续跑。返回 (已有记录列表, 已成功的问题集合)。"""
    if not os.path.exists(output_path):
        return [], set()
    try:
        ext = os.path.splitext(output_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(output_path, engine="openpyxl")
        else:
            df = pd.read_csv(output_path, encoding="utf-8-sig")
        records = df.to_dict("records")
        completed = {r["问题"] for r in records if r.get("状态") == "成功"}
        logger.info(f"断点续跑：从已有结果文件加载了 {len(records)} 条记录（其中 {len(completed)} 条成功）")
        return records, completed
    except Exception as e:
        logger.warning(f"加载已有结果文件失败，将从头开始: {e}")
        return [], set()


def _process_single_question(
    question: str,
    original_index: int,
    display_pos: int,
    total: int,
    knowledge_dir: str,
    max_rounds: int,
    vendor: str,
    model: str,
    clean_answer: bool = False,
    summary_batch_size: int = 0,
    retrieval_mode: bool = False,
    check_pitfalls: bool = False,
) -> dict:
    """处理单个问题并返回结果字典，供串行和并行模式共用。
    original_index: 问题在原始输入文件中的数据行索引（0-based）。
    display_pos: 用于日志显示的顺序位置（1-based）。
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"[{display_pos}/{total}] (idx={original_index}) {question[:80]}...")
    logger.info(f"{'='*60}")

    q_start = time.time()
    try:
        graph = AgentGraph(
            question=question,
            knowledge_root=knowledge_dir,
            max_rounds=max_rounds,
            vendor=vendor,
            model=model,
            clean_answer=clean_answer,
            summary_batch_size=summary_batch_size,
            retrieval_mode=retrieval_mode,
            check_pitfalls=check_pitfalls,
        )
        result = graph.run()
        answer = result["answer"]
        trace_log = result["trace_log"]
        relevant_chapters = result.get("relevant_chapters", [])
        full_output = answer + TRACE_SEPARATOR + trace_log
        elapsed = round(time.time() - q_start, 1)

        logger.info(f"问题 idx={original_index} 推理完成，耗时 {elapsed}s")
        return {
            "原始序号": original_index,
            "问题": question,
            "答案": answer,
            "相关章节": ", ".join(relevant_chapters),
            "完整输出（含追踪）": full_output,
            "耗时(秒)": elapsed,
            "状态": "成功",
        }

    except Exception as e:
        elapsed = round(time.time() - q_start, 1)
        logger.error(f"问题 idx={original_index} 推理失败 ({elapsed}s): {e}")
        return {
            "原始序号": original_index,
            "问题": question,
            "答案": f"推理失败: {e}",
            "完整输出（含追踪）": f"推理失败: {e}",
            "耗时(秒)": elapsed,
            "状态": "失败",
        }


def run_single_question(
    question: str,
    knowledge_dir: str,
    max_rounds: int = 5,
    vendor: str = "aliyun",
    model: str = "deepseek-v3.2",
    clean_answer: bool = False,
    summary_batch_size: int = 0,
    retrieval_mode: bool = False,
    check_pitfalls: bool = False,
) -> dict:
    """
    对单个问题执行推理，直接返回结果字典，同时打印答案到终端。
    返回 {"问题", "答案", "完整输出（含追踪）", "状态"}。
    """
    if not os.path.isdir(knowledge_dir):
        raise ValueError(f"知识目录不存在: {knowledge_dir}")
    knowledge_md = os.path.join(knowledge_dir, "knowledge.md")
    if not os.path.exists(knowledge_md):
        raise ValueError(f"知识目录根无 knowledge.md: {knowledge_dir}")

    logger.info(f"推理问题: {question}")
    graph = AgentGraph(
        question=question,
        knowledge_root=knowledge_dir,
        max_rounds=max_rounds,
        vendor=vendor,
        model=model,
        clean_answer=clean_answer,
        summary_batch_size=summary_batch_size,
        retrieval_mode=retrieval_mode,
        check_pitfalls=check_pitfalls,
    )
    result = graph.run()
    answer = result["answer"]
    trace_log = result["trace_log"]
    relevant_chapters = result.get("relevant_chapters", [])
    full_output = answer + TRACE_SEPARATOR + trace_log

    print("\n" + "=" * 60)
    print("【答案】")
    print(answer)
    if relevant_chapters:
        print(f"\n【相关章节】{', '.join(relevant_chapters)}")
    print("=" * 60)

    return {
        "问题": question,
        "答案": answer,
        "相关章节": ", ".join(relevant_chapters),
        "完整输出（含追踪）": full_output,
        "状态": "成功",
    }


def _run_sequential(
    pending: list[tuple[int, int, str]],
    total: int,
    results: list[dict],
    output_path: str,
    knowledge_dir: str,
    max_rounds: int,
    vendor: str,
    model: str,
    clean_answer: bool = False,
    summary_batch_size: int = 0,
    retrieval_mode: bool = False,
    check_pitfalls: bool = False,
) -> None:
    """串行逐个处理待推理问题。"""
    for original_index, display_pos, question in pending:
        result_dict = _process_single_question(
            question, original_index, display_pos, total,
            knowledge_dir, max_rounds, vendor, model, clean_answer, summary_batch_size,
            retrieval_mode, check_pitfalls,
        )
        results.append(result_dict)
        _flush_results(results, output_path)
        _log_progress(results, total, output_path)


def _run_parallel(
    pending: list[tuple[int, int, str]],
    total: int,
    results: list[dict],
    output_path: str,
    knowledge_dir: str,
    max_rounds: int,
    vendor: str,
    model: str,
    max_workers: int,
    clean_answer: bool = False,
    summary_batch_size: int = 0,
    retrieval_mode: bool = False,
    check_pitfalls: bool = False,
) -> None:
    """并行处理待推理问题，线程安全地收集结果并实时落盘。"""
    lock = threading.Lock()
    effective_workers = min(max_workers, len(pending))
    logger.info(f"并行推理模式：{len(pending)} 个问题，{effective_workers} 个 worker")

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(
                _process_single_question,
                question, original_index, display_pos, total,
                knowledge_dir, max_rounds, vendor, model, clean_answer, summary_batch_size,
                retrieval_mode, check_pitfalls,
            ): (original_index, question)
            for original_index, display_pos, question in pending
        }

        for future in as_completed(futures):
            original_index, question = futures[future]
            try:
                result_dict = future.result()
            except Exception as e:
                logger.error(f"问题 idx={original_index} worker 异常: {e}")
                result_dict = {
                    "原始序号": original_index,
                    "问题": question,
                    "答案": f"推理失败: {e}",
                    "完整输出（含追踪）": f"推理失败: {e}",
                    "耗时(秒)": 0,
                    "状态": "失败",
                }

            with lock:
                results.append(result_dict)
                _flush_results(results, output_path)
                _log_progress(results, total, output_path)


def _log_progress(results: list[dict], total: int, output_path: str) -> None:
    done = sum(1 for r in results if r.get("状态") == "成功")
    failed = sum(1 for r in results if r.get("状态") == "失败")
    logger.info(f"已实时保存 ({done} 成功 / {failed} 失败 / {total} 总计): {output_path}")


def run_reasoning(
    questions_file: str,
    question_column: str,
    knowledge_dir: str,
    max_rounds: int = 5,
    vendor: str = "aliyun",
    model: str = "deepseek-v3.2",
    output_path: str = None,
    max_workers: int = 1,
    clean_answer: bool = False,
    summary_batch_size: int = 0,
    retrieval_mode: bool = False,
    check_pitfalls: bool = False,
) -> str:
    """
    推理引擎主入口。
    加载问题集，对每个问题执行 AgentGraph 推理，每完成一题实时写入结果文件。
    当 max_workers > 1 时启用问题粒度的并行推理。
    默认输出 CSV 格式，若 output_path 指定 .xlsx 则输出 Excel。
    支持断点续跑：指定已有的 output_path 时自动跳过已成功的问题。
    返回结果文件路径。
    """
    if not os.path.isdir(knowledge_dir):
        raise ValueError(f"知识目录不存在: {knowledge_dir}")

    knowledge_md = os.path.join(knowledge_dir, "knowledge.md")
    if not os.path.exists(knowledge_md):
        raise ValueError(f"知识目录根无 knowledge.md: {knowledge_dir}")

    indexed_questions = load_questions(questions_file, question_column)

    if output_path is None:
        output_dir = os.path.dirname(os.path.abspath(questions_file))
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        output_path = os.path.join(output_dir, f"results_{timestamp}.csv")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    results, completed_questions = _load_existing_results(output_path)

    pending: list[tuple[int, int, str]] = []
    skipped = 0
    for display_pos, (original_index, question) in enumerate(indexed_questions, 1):
        if question in completed_questions:
            skipped += 1
            logger.info(f"[{display_pos}/{len(indexed_questions)}] 跳过已完成 (idx={original_index}): {question[:60]}...")
        else:
            pending.append((original_index, display_pos, question))

    if skipped:
        logger.info(f"已跳过 {skipped} 个已完成的问题，剩余 {len(pending)} 个待推理")

    if not pending:
        logger.info("所有问题均已完成，无需推理")
        return output_path

    total = len(indexed_questions)
    total_start = time.time()

    if max_workers <= 1:
        _run_sequential(
            pending, total, results, output_path,
            knowledge_dir, max_rounds, vendor, model, clean_answer, summary_batch_size,
            retrieval_mode, check_pitfalls,
        )
    else:
        _run_parallel(
            pending, total, results, output_path,
            knowledge_dir, max_rounds, vendor, model, max_workers, clean_answer, summary_batch_size,
            retrieval_mode, check_pitfalls,
        )

    total_elapsed = round(time.time() - total_start, 1)
    logger.info(f"批量推理完成，总耗时 {total_elapsed}s，结果文件: {output_path}")
    return output_path

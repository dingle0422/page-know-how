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


def cmd_serve(args):
    """启动 FastAPI 服务（等价于直接 ``python app.py``）。"""
    import uvicorn

    web_stream_mode = getattr(args, "web_stream_mode", None)
    if web_stream_mode is not None:
        os.environ["WEB_STREAM_MODE"] = "1" if web_stream_mode else "0"

    host = getattr(args, "host", "0.0.0.0")
    port = int(getattr(args, "port", 5000))

    # 延迟 import：避免 extract / reason 子命令也加载整个 Web 服务依赖树。
    from app import app as web_app

    uvicorn.run(web_app, host=host, port=port, timeout_keep_alive=3600)


def main():
    parser = argparse.ArgumentParser(
        description="文档知识结构化抽取 + 渐进式披露推理框架"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="启用详细日志")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # serve 子命令
    serve_parser = subparsers.add_parser(
        "serve", help="启动 FastAPI 服务（默认端口 5000）"
    )
    serve_parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=5000, help="监听端口，默认 5000")
    serve_parser.add_argument(
        "--web-stream-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否覆盖 WEB_STREAM_MODE 环境变量；不传则沿用 env，传则显式开关",
    )

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

    args = parser.parse_args()
    setup_logging(args.verbose)

    # 不传子命令时默认起服务（host=0.0.0.0, port=5000）。
    if args.command is None or args.command == "serve":
        cmd_serve(args)
    elif args.command == "extract":
        cmd_extract(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

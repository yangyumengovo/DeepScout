"""命令行入口：deepscout research "问题" [--mock] ｜  deepscout serve"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="deepscout", description="DeepScout —— 深度调研 Agent"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_research = sub.add_parser("research", help="执行一次深度调研")
    p_research.add_argument("question", help="研究问题，例如：2026 年国产大模型格局如何？")
    p_research.add_argument("--mock", action="store_true", help="离线演示模式（无需 API key 与网络）")
    p_research.add_argument("--subqueries", type=int, default=None, help="子查询数量（默认读配置）")

    p_serve = sub.add_parser("serve", help="启动 FastAPI 服务")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    p_ui = sub.add_parser("ui", help="启动 Web UI（Gradio，需安装 .[ui] 可选依赖）")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=7860)

    args = parser.parse_args(argv)

    if args.cmd == "research":
        from .runner import run_research
        from .settings import Settings

        settings = Settings()
        if args.subqueries:
            settings.max_subqueries = args.subqueries

        print(f"研究问题：{args.question}{'（离线演示模式）' if args.mock else ''}")
        result = run_research(
            args.question,
            settings=settings,
            mock=args.mock,
            on_event=lambda e: print(f"  -> 节点完成：{e['name']}", flush=True),
        )
        if result.get("status") != "done":
            print(f"\n[失败] {result.get('error')}")
            sys.exit(1)

        print("\n" + "=" * 64)
        print(result["report_md"])
        print("=" * 64)
        st = result["stats"]
        print(
            f"\n产物目录：{result['run_dir']}"
            f"  | LLM 调用 {st['llm_calls']} 次 | 工具调用 {st['tool_calls']} 次"
            f" | token {st['tokens']} | 用时 {st['wall_s']}s"
        )

    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("deepscout.api.app:app", host=args.host, port=args.port)

    elif args.cmd == "ui":
        try:
            from .webui import main as ui_main
        except ImportError as e:
            print(f"缺少 Gradio 依赖：{e}\n请运行 pip install '.[ui]' 后重试")
            sys.exit(1)
        ui_main(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

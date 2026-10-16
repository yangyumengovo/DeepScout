"""运行器：组装依赖、驱动 LangGraph、落盘产物（report.md / sources.json / stats.json / trace.jsonl）。"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from .evidence import get_reranker
from .graph import RunContext, build_research_graph
from .llm import make_llm
from .search import make_search
from .settings import Settings
from .tools import ToolBox
from .trace import RunTrace


def run_research(
    question: str,
    settings: Settings | None = None,
    mock: bool = False,
    on_event=None,
) -> dict:
    """执行一次完整调研，返回包含 report_md / sources / stats / run_dir 的结果字典。

    on_event(node 事件回调) 用于 CLI 进度展示与 API 的 SSE 推送。
    任何异常都会被记录到结果与 trace 中而不是中断（服务场景不能 500）。
    """
    settings = settings or Settings()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    run_dir = Path(settings.runs_dir) / run_id
    trace = RunTrace(run_dir, model="mock" if mock else settings.llm_model)
    t0 = time.perf_counter()
    final: dict = {"question": question}

    try:
        ctx = RunContext(
            settings=settings,
            llm=make_llm(settings, mock),
            toolbox=ToolBox(make_search(settings, mock)),
            trace=trace,
            reranker=get_reranker(settings) if not mock else None,
        )
        graph = build_research_graph(ctx)
        for update in graph.stream({"question": question}, config={"recursion_limit": 100}):
            for node, payload in update.items():
                if node.startswith("__"):
                    continue
                if isinstance(payload, dict):
                    final.update(payload)
                if on_event:
                    on_event({"type": "node", "name": node})
        final["status"] = "done"
    except Exception as e:  # noqa: BLE001 —— 服务化需要把错误带回给调用方
        final["status"] = "error"
        final["error"] = f"{type(e).__name__}: {e}"
        trace.error(final["error"])
        traceback.print_exc()

    finally:
        stats = trace.stats(time.perf_counter() - t0)
        final["stats"] = stats
        if final.get("status") == "done" and final.get("report_body"):
            title = final.get("outline", {}).get("title", question)
            header = (
                f"# {title}\n\n"
                f"> 研究问题：{question}  \n"
                f"> 模型：{trace.model} · LLM 调用 {stats['llm_calls']} 次 · "
                f"token {stats['tokens']} · 估算成本 ${stats['est_cost_usd']} · "
                f"用时 {stats['wall_s']}s\n\n---\n\n"
            )
            footer = "\n\n---\n### 运行统计\n" + "\n".join(
                f"- {k}: {v}" for k, v in stats.items() if k != "nodes"
            )
            report_md = header + final.pop("report_body") + footer
            final["report_md"] = report_md
            trace.artifact("report.md", report_md)
            trace.artifact(
                "sources.json", json.dumps(final.get("sources", []), ensure_ascii=False, indent=2)
            )
        trace.artifact("stats.json", json.dumps(stats, ensure_ascii=False, indent=2))
        final["run_dir"] = str(run_dir)
        trace.close()

    return final

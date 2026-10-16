"""Web UI（Gradio）：输入研究问题 -> 实时节点进度 -> 渲染 Markdown 报告。

启动：deepscout ui --port 7860（需 pip install '.[ui]'）
"""

from __future__ import annotations

import threading
import time


def create_ui():
    import gradio as gr

    from .runner import run_research
    from .settings import Settings

    def research(question: str, mock: bool):
        state: dict = {"events": [], "done": False, "result": None}

        def worker() -> None:
            try:
                state["result"] = run_research(
                    question,
                    settings=Settings(),
                    mock=mock,
                    on_event=lambda e: state["events"].append(e),
                )
            except Exception as e:  # noqa: BLE001
                state["result"] = {"status": "error", "error": str(e)}
            finally:
                state["done"] = True

        threading.Thread(target=worker, daemon=True).start()
        seen = 0
        while not state["done"]:
            time.sleep(0.4)
            while seen < len(state["events"]):
                seen += 1
                names = [e["name"] for e in state["events"][:seen]]
                yield "运行中：" + " -> ".join(names), ""
        result = state["result"] or {}
        if result.get("status") != "done":
            yield f"❌ 失败：{result.get('error')}", ""
            return
        stats = result.get("stats", {})
        yield (
            f"✅ 完成 · LLM 调用 {stats.get('llm_calls')} 次 · token {stats.get('tokens')} · "
            f"用时 {stats.get('wall_s')}s",
            result.get("report_md", ""),
        )

    with gr.Blocks(title="DeepScout · 深度调研 Agent") as demo:
        gr.Markdown("# DeepScout · 深度调研 Agent\n规划 → 检索取证 → 证据库检索 → 撰写 → 引用核验")
        with gr.Row():
            question = gr.Textbox(
                label="研究问题",
                value="RAG 系统中混合检索相比纯向量检索的优势是什么？",
                scale=4,
            )
            mock = gr.Checkbox(label="离线演示模式（无需 API key）", value=False, scale=1)
        btn = gr.Button("开始调研", variant="primary")
        progress = gr.Markdown("等待开始…")
        report = gr.Markdown()
        btn.click(fn=research, inputs=[question, mock], outputs=[progress, report])
    return demo


def main(host: str = "127.0.0.1", port: int = 7860) -> None:
    demo = create_ui()
    demo.queue().launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()

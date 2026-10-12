"""运行追踪：每次调研的 LLM 调用、工具调用、节点耗时全部记入 trace.jsonl。

失败归因全靠它：报告出错时，先看 trace 判断是检索没查到、工具失败、还是生成幻觉，
而不是盯着最终报告猜。
"""

from __future__ import annotations

import json
from pathlib import Path

from .settings import estimate_cost_usd


class RunTrace:
    def __init__(self, run_dir: Path, model: str = ""):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self._fh = (self.run_dir / "trace.jsonl").open("a", encoding="utf-8")
        self.llm_calls = 0
        self.tool_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.node_seconds: dict[str, float] = {}

    def _write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()

    def llm(self, phase: str, resp) -> None:
        self.llm_calls += 1
        self.prompt_tokens += resp.usage.get("prompt_tokens", 0) or 0
        self.completion_tokens += resp.usage.get("completion_tokens", 0) or 0
        self._write(
            {
                "kind": "llm",
                "phase": phase,
                "model": resp.model or self.model,
                "latency_s": round(resp.latency_s, 3),
                "usage": resp.usage,
                "tool_calls": [tc.name for tc in resp.tool_calls],
                "content_head": (resp.content or "")[:160],
            }
        )

    def tool(self, name: str, ok: bool) -> None:
        self.tool_calls += 1
        self._write({"kind": "tool", "name": name, "ok": ok})

    def node(self, name: str, seconds: float) -> None:
        self.node_seconds[name] = round(seconds, 3)
        self._write({"kind": "node", "name": name, "seconds": round(seconds, 3)})

    def error(self, message: str) -> None:
        self._write({"kind": "error", "error": message})

    def artifact(self, name: str, text: str) -> Path:
        p = self.run_dir / name
        p.write_text(text, encoding="utf-8")
        return p

    def stats(self, wall_s: float) -> dict:
        tokens = self.prompt_tokens + self.completion_tokens
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens": tokens,
            "est_cost_usd": round(
                estimate_cost_usd(self.model, self.prompt_tokens, self.completion_tokens), 6
            ),
            "wall_s": round(wall_s, 2),
            "nodes": dict(self.node_seconds),
        }

    def close(self) -> None:
        self._fh.close()

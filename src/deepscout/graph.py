"""LangGraph 调研引擎：plan -> retrieve -> index -> outline -> write -> verify -> assemble。

为什么要用图编排而不是一个大 while 循环（面试常问）：
1. 每个节点职责单一，可单独测试、单独重跑；
2. 状态显式流动，配合 checkpointer 未来可断点续跑；
3. 节点耗时/失败在 trace 中天然可归因。

其中 retrieve 节点内部是一个真正的 Function Calling 循环：
模型自主决定调用 fetch_page / web_search 工具，工具结果回填后继续推理，
直到模型认为证据足够、输出调研笔记为止（有最大步数保护）。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import prompts
from .evidence import EvidenceIndex, build_evidence, filter_by_anchors, get_embedder, question_anchors
from .json_utils import extract_json
from .llm import LLMClient
from .settings import Settings
from .tools import ToolBox
from .trace import RunTrace

_CITE_RE = re.compile(r"\[(E\d+)\]")


class ResearchState(TypedDict, total=False):
    question: str
    subqueries: list[str]
    notes: list[dict]
    pages: list[dict]
    chunks: list[dict]
    outline: dict
    sections: list[dict]
    verification: dict
    report_body: str
    sources: list[dict]


@dataclass
class RunContext:
    """一次运行的全部依赖（依赖注入，便于测试替换）。"""

    settings: Settings
    llm: LLMClient
    toolbox: ToolBox
    trace: RunTrace
    index: EvidenceIndex | None = field(default=None)


def build_research_graph(ctx: RunContext):
    s = ctx.settings

    def call_llm(phase, messages, tools=None, temperature=0.3, expect_json=False):
        """统一 LLM 调用入口：记 trace；JSON 解析失败自动重试一次。"""
        resp = ctx.llm.complete(messages, tools, temperature)
        ctx.trace.llm(phase, resp)
        if expect_json and extract_json(resp.content) is None:
            retry_msgs = messages + [
                {"role": "assistant", "content": resp.content or ""},
                {"role": "user", "content": "上面的输出无法解析为 JSON。请重新输出，只输出合法 JSON。"},
            ]
            resp2 = ctx.llm.complete(retry_msgs, None, temperature)
            ctx.trace.llm(phase + ":json-retry", resp2)
            if extract_json(resp2.content) is not None:
                return resp2
        return resp

    # ---------- 1. 规划：研究问题 -> 子查询 ----------
    def plan_node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        if s.disable_planner:  # 消融：跳过规划，用原问题作为唯一子查询
            ctx.trace.node("plan", time.perf_counter() - t0)
            return {"subqueries": [state["question"]]}
        msgs = prompts.plan_messages(state["question"], s.max_subqueries, s.report_language)
        resp = call_llm("plan", msgs, temperature=0.2, expect_json=True)
        data = extract_json(resp.content) or {}
        subs = [str(q).strip()[:200] for q in data.get("subqueries", []) if str(q).strip()]
        subs = subs[: s.max_subqueries] or [state["question"]]
        ctx.trace.node("plan", time.perf_counter() - t0)
        return {"subqueries": subs}

    # ---------- 2. 检索取证：每个子查询一个 Function Calling 循环 ----------
    def retrieve_node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        notes: list[dict] = []
        pages: list[dict] = []
        seen_urls: set[str] = set()
        tool_schemas = ctx.toolbox.schemas(["fetch_page", "web_search"])

        for sq in state["subqueries"]:
            results_raw = ctx.toolbox.call("web_search", query=sq)
            try:
                results = json.loads(results_raw)
            except json.JSONDecodeError:
                results = []
            ctx.trace.tool("web_search", isinstance(results, list) and len(results) > 0)
            if not isinstance(results, list):
                results = []

            msgs = prompts.search_agent_messages(
                state["question"],
                sq,
                prompts.format_search_results(results),
                avoid_urls=sorted(seen_urls),
            )
            fetched_this_query = 0
            last_content = ""
            for _step in range(s.max_tool_steps):
                resp = call_llm(f"retrieve:{sq[:24]}", msgs, tools=tool_schemas, temperature=0.2)
                last_content = resp.content or ""
                if not resp.tool_calls:
                    break
                msgs.append(
                    {
                        "role": "assistant",
                        "content": resp.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id or f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                                },
                            }
                            for i, tc in enumerate(resp.tool_calls)
                        ],
                    }
                )
                for i, tc in enumerate(resp.tool_calls):
                    if tc.name == "fetch_page" and fetched_this_query >= s.max_fetch_per_query:
                        out = json.dumps({"error": "已达到本子问题的抓取条数上限"}, ensure_ascii=False)
                        ctx.trace.tool("fetch_page", False)
                    else:
                        out = ctx.toolbox.call(tc.name, **tc.arguments)
                        try:
                            out_obj = json.loads(out)
                        except json.JSONDecodeError:
                            out_obj = {"error": "工具输出不可解析"}
                        ok = "error" not in out_obj
                        ctx.trace.tool(tc.name, ok)
                        if tc.name == "fetch_page" and ok:
                            fetched_this_query += 1
                            url = out_obj.get("url", "")
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                pages.append(out_obj)
                    msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id or f"call_{i}",
                            "content": prompts.tool_result_digest(out),
                        }
                    )
            notes.append({"subquery": sq, "note": last_content or "（未获得可用信息）"})

        ctx.trace.node("retrieve", time.perf_counter() - t0)
        return {"notes": notes, "pages": pages}

    # ---------- 3. 证据索引：分块 + 混合检索 ----------
    def index_node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        chunks = build_evidence(state.get("pages", []), s)
        ctx.index = EvidenceIndex(chunks, get_embedder(s))
        ctx.trace.node("index", time.perf_counter() - t0)
        return {
            "chunks": [
                {"eid": c.eid, "url": c.url, "title": c.title, "text": c.text[:200]} for c in chunks
            ]
        }

    # ---------- 4. 大纲 ----------
    def outline_node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        digest = "\n".join(
            f"【{n['subquery']}】{n['note'][:300]}" for n in state.get("notes", [])
        )
        resp = call_llm(
            "outline",
            prompts.outline_messages(state["question"], digest or "（无笔记）", s.report_language),
            temperature=0.3,
            expect_json=True,
        )
        data = extract_json(resp.content) or {}
        title = str(data.get("title") or state["question"])[:80]
        sections = [str(x).strip()[:60] for x in data.get("sections", []) if str(x).strip()]
        sections = sections[:6] or ["综述"]
        ctx.trace.node("outline", time.perf_counter() - t0)
        return {"outline": {"title": title, "sections": sections}}

    # ---------- 5. 分章撰写（RAG：从证据库检索后引用写作） ----------
    def write_node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        # v1.3 硬校验：题目含明确对象锚点时，仅允许引用提到该对象的证据
        anchors = question_anchors(state["question"]) if s.strict_object_grounding else []
        sections_out = []
        for title in state["outline"]["sections"]:
            hits = ctx.index.search(f"{state['question']} {title}", k=8) if ctx.index else []
            eligible = filter_by_anchors([c for c, _ in hits], anchors)
            ev_text = prompts.format_evidence(eligible)
            resp = call_llm(
                f"write:{title[:20]}",
                prompts.section_messages(state["question"], title, ev_text, s.report_language),
                temperature=0.3,
            )
            sections_out.append({"title": title, "text": (resp.content or "").strip()})
        ctx.trace.node("write", time.perf_counter() - t0)
        return {"sections": sections_out}

    # ---------- 6. 引用核验（差异化能力：报告里的引用逐条对回证据原文） ----------
    def verify_node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        if s.disable_verify:  # 消融：跳过核验（用于量化核验模块本身的成本与价值）
            ctx.trace.node("verify", time.perf_counter() - t0)
            return {"verification": {"results": [], "supported_rate": None, "skipped": True}}
        chunk_map = {c.eid: c for c in (ctx.index.chunks if ctx.index else [])}
        pairs = []
        seen: set[str] = set()
        for sec in state.get("sections", []):
            for m in _CITE_RE.finditer(sec["text"]):
                eid = m.group(1)
                if eid in seen or eid not in chunk_map:
                    continue
                seen.add(eid)
                start = max(0, m.start() - 60)
                claim = sec["text"][start : m.end() + 20].replace("\n", " ")
                pairs.append({"evidence_id": eid, "claim": claim, "evidence": chunk_map[eid].text[:260]})

        if not pairs:
            ctx.trace.node("verify", time.perf_counter() - t0)
            return {"verification": {"results": [], "supported_rate": None}}

        pairs_text = "\n\n".join(
            f"[{p['evidence_id']}] 证据原文：{p['evidence']}\n被用于支撑的论断：……{p['claim']}……"
            for p in pairs
        )
        resp = call_llm(
            "verify", prompts.verify_messages(pairs_text), temperature=0.0, expect_json=True
        )
        data = extract_json(resp.content) or {}
        verdicts = {
            str(r.get("evidence_id")): r
            for r in data.get("results", [])
            if isinstance(r, dict) and r.get("evidence_id")
        }
        results = []
        for p in pairs:
            v = verdicts.get(p["evidence_id"])
            results.append(
                {
                    "evidence_id": p["evidence_id"],
                    "claim": p["claim"][:120],
                    "verdict": (v.get("verdict", "unverified") if v else "unverified"),
                    "reason": (str(v.get("reason", "")) if v else "模型未返回该条核验结果")[:160],
                }
            )
        n_ok = sum(1 for r in results if r["verdict"] == "supported")
        ctx.trace.node("verify", time.perf_counter() - t0)
        return {"verification": {"results": results, "supported_rate": round(n_ok / len(results), 3)}}

    # ---------- 7. 组装：E 编号 -> 全文序号，生成来源列表与核验附录 ----------
    def assemble_node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        chunk_map = {c.eid: c for c in (ctx.index.chunks if ctx.index else [])}

        ordered: list[str] = []
        for sec in state.get("sections", []):
            for m in _CITE_RE.finditer(sec["text"]):
                if m.group(1) not in ordered:
                    ordered.append(m.group(1))
        num = {eid: i + 1 for i, eid in enumerate(ordered)}

        def relabel(text: str) -> str:
            return _CITE_RE.sub(lambda m: f"[{num.get(m.group(1), '?')}]", text)

        sections_md = "\n\n".join(
            f"## {sec['title']}\n\n{relabel(sec['text'])}" for sec in state.get("sections", [])
        )

        sources_lines, sources = [], []
        for eid in ordered:
            c = chunk_map.get(eid)
            if not c:
                continue
            sources_lines.append(f"{num[eid]}. [{c.title}]({c.url})（证据 {eid}）")
            sources.append({"no": num[eid], "eid": eid, "title": c.title, "url": c.url})

        verif = state.get("verification", {})
        icons = {"supported": "✅ supported", "partial": "⚠️ partial", "unsupported": "❌ unsupported"}
        rows = ["| 证据 | 论断摘要 | 核验结果 | 说明 |", "|---|---|---|---|"]
        for r in verif.get("results", []):
            icon = icons.get(r["verdict"], "· 未核验")
            rows.append(f"| {r['evidence_id']} | {r['claim'][:60]} | {icon} | {r['reason'][:80]} |")
        rate = verif.get("supported_rate")
        if rate is not None:
            n_ok = sum(1 for r in verif.get("results", []) if r["verdict"] == "supported")
            rate_text = f"引用核验通过率：{round(rate * 100)}%（{n_ok}/{len(verif.get('results', []))}）"
        else:
            rate_text = "本报告无引用，无法核验"

        body = (
            sections_md
            + "\n\n## 参考来源\n\n"
            + ("\n".join(sources_lines) or "（无）")
            + "\n\n## 引用核验\n\n"
            + rate_text
            + "\n\n"
            + "\n".join(rows)
        )
        ctx.trace.node("assemble", time.perf_counter() - t0)
        return {"report_body": body, "sources": sources}

    g = StateGraph(ResearchState)
    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("index", index_node)
    g.add_node("outline", outline_node)
    g.add_node("write", write_node)
    g.add_node("verify", verify_node)
    g.add_node("assemble", assemble_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "index")
    g.add_edge("index", "outline")
    g.add_edge("outline", "write")
    g.add_edge("write", "verify")
    g.add_edge("verify", "assemble")
    g.add_edge("assemble", END)
    return g.compile()

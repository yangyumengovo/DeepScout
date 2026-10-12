"""版本化 Prompt 集中管理。

约定：每个角色的 system prompt 中包含一个稳定的「角色标记」（如「研究规划师」），
- 真实运行时用于在 trace 中区分调用阶段；
- FakeLLM 靠它匹配离线剧本。
Prompt 的任何修改都应递增版本号并同步记录到 eval/ 的调优记录中。
"""

from __future__ import annotations

PROMPT_VERSION = "v1"

# ---- 角色标记（FakeLLM 剧本匹配依赖，勿随意改动）----
MARKER_PLAN = "研究规划师"
MARKER_SEARCH = "资料检索员"
MARKER_OUTLINE = "大纲设计师"
MARKER_WRITE = "报告撰写者"
MARKER_VERIFY = "引用核验员"

_JSON_ONLY = "只输出 JSON，不要输出任何其他文字或代码围栏。"


def plan_messages(question: str, n: int, language: str = "中文") -> list[dict]:
    system = (
        f"你是{MARKER_PLAN}。给定一个研究问题，把它拆解成 {n} 个互相补充、可直接用于搜索引擎的子查询，"
        f"覆盖问题的不同侧面（背景/方法/对比/最新进展等）。子查询用{language}与英文关键词混合，利于检索。"
        f'{_JSON_ONLY} 格式：{{"subqueries": ["...", "..."]}}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"研究问题：{question}"},
    ]


def search_agent_messages(question: str, subquery: str, results_text: str) -> list[dict]:
    system = (
        f"你是{MARKER_SEARCH}，负责为一个研究子问题收集证据。你可以调用工具：\n"
        "- fetch_page(url)：抓取网页正文，判断是否值得引用；\n"
        "- web_search(query)：更换关键词重新搜索。\n"
        "工作方式：先从下方搜索结果中挑选最相关的 1-3 条调用 fetch_page 阅读；"
        "若都不相关，可换关键词调用 web_search。信息足够后，输出调研笔记：3-6 条要点，"
        "每条要点末尾以 [SRC: 页面URL] 标注来源；只陈述页面中真实出现的信息，不要编造。"
    )
    user = (
        f"总研究问题：{question}\n当前子问题：{subquery}\n\n搜索结果：\n{results_text}\n\n"
        "请开始（先决定要抓取哪些链接）。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def outline_messages(question: str, notes_digest: str, language: str = "中文") -> list[dict]:
    system = (
        f"你是{MARKER_OUTLINE}。基于调研笔记，为研究报告设计大纲：1 个标题 + 3-5 个章节，"
        f"章节顺序要有逻辑（背景->方法->对比->结论）。用{language}。{_JSON_ONLY} "
        '格式：{"title": "...", "sections": ["...", "..."]}'
    )
    user = f"研究问题：{question}\n\n调研笔记摘要：\n{notes_digest}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def section_messages(
    question: str, section_title: str, evidence_text: str, language: str = "中文"
) -> list[dict]:
    system = (
        f"你是{MARKER_WRITE}。撰写报告的一个章节，要求：\n"
        f"1. 150-300 字，用{language}，信息密度高，不写空话；\n"
        "2. 每一条具体事实/数据后必须紧跟证据编号引用，如 [E3]；只引用下方给出的编号，禁止编造；\n"
        "3. 若下方证据不足以支撑本章节，直接说明「现有资料不足以回答该方面」，不要猜测。"
    )
    user = (
        f"总研究问题：{question}\n本章节标题：{section_title}\n\n可用证据：\n{evidence_text}\n\n"
        f"请撰写「{section_title}」章节正文。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def verify_messages(pairs_text: str) -> list[dict]:
    system = (
        f"你是{MARKER_VERIFY}。给你若干「论断 - 证据」对，逐条判断证据是否能支撑论断：\n"
        '- supported：证据明确支撑；partial：部分支撑或不够具体；unsupported：证据不支撑或无关。\n'
        f"严格依据证据原文判断，宁可判 partial 也不要放水。{_JSON_ONLY} "
        '格式：{"results": [{"evidence_id": "E3", "verdict": "supported", "reason": "..."}, ...]}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"待核验对：\n{pairs_text}"},
    ]


def format_search_results(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '')}\n   {r.get('url', '')}\n   {r.get('snippet', '')[:150]}")
    return "\n".join(lines) if lines else "（无搜索结果）"


def format_evidence(chunks: list) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[{c.eid}] 来源：{c.title}（{c.url}）\n摘录：{c.text[:300]}")
    return "\n\n".join(lines) if lines else "（无证据，请如实说明信息不足）"


def tool_result_digest(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n...(截断，原文 {len(text)} 字符)"

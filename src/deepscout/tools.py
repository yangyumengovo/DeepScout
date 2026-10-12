"""工具层：注册即得 JSON Schema（Function Calling 协议）+ 参数校验 + 安全抓取。

设计要点（面试常问）：
- 用 pydantic 模型声明参数 -> 自动生成 OpenAI tools schema，模型侧与校验侧永不脱节；
- 工具执行永不抛异常：错误以 JSON 字符串返回给模型，让 Agent 自己决定重试或放弃；
- 网页抓取有超时、体积上限、正文抽取与长度截断，防止上下文被单页撑爆。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from pydantic import BaseModel, Field

from .search import SearchProvider

_DEMO_URL_RE = re.compile(r"^demo://[a-z0-9-_]+(/[a-z0-9-_]+)*$")


class WebSearchArgs(BaseModel):
    query: str = Field(..., description="搜索关键词，中英文均可")
    max_results: int = Field(5, ge=1, le=10, description="返回结果条数")


class FetchPageArgs(BaseModel):
    url: str = Field(..., description="要抓取正文的网页地址")


def fetch_page(url: str) -> dict:
    """抓取网页正文：demo:// 走内置语料，http(s):// 走真实网络并抽取正文。"""
    if url.startswith("demo://"):
        return _fetch_demo(url)
    return _fetch_http(url)


def _fetch_demo(url: str) -> dict:
    from importlib import resources

    if not _DEMO_URL_RE.match(url):
        return {"url": url, "title": "", "text": "", "error": "非法 demo 地址"}
    stem = url.removeprefix("demo://").split("/")[-1]
    index = json.loads(
        (resources.files("deepscout") / "sample_data" / "index.json").read_text("utf-8")
    )
    entry = next((e for e in index if e["url"] == url), None)
    if entry is None:
        return {"url": url, "title": "", "text": "", "error": "演示语料中不存在该页面"}
    text = (resources.files("deepscout") / "sample_data" / f"{stem}.md").read_text("utf-8")
    return {"url": url, "title": entry["title"], "text": text[:6000]}


def _fetch_http(url: str, max_bytes: int = 2_000_000) -> dict:
    import httpx

    try:
        resp = httpx.get(
            url,
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepScoutBot/0.1)"},
        )
        resp.raise_for_status()
    except Exception as e:
        return {"url": url, "title": url, "text": "", "error": f"抓取失败: {e}"}

    content = resp.content[:max_bytes]
    html = content.decode(resp.encoding or "utf-8", errors="ignore")
    title, text = _extract_readable(html)
    return {"url": str(resp.url), "title": title or url, "text": text[:6000]}


def _extract_readable(html: str) -> tuple[str, str]:
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False) or ""
        title = ""
        if text:
            meta = trafilatura.extract_metadata(html)
            title = (getattr(meta, "title", "") or "") if meta else ""
        if text:
            return title, text
    except Exception:
        pass
    # 兜底：bs4 粗抽取
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return (soup.title.string.strip() if soup.title and soup.title.string else ""), soup.get_text(
        " ", strip=True
    )


class ToolBox:
    """工具注册箱：schema 生成、参数校验、统一调用入口。"""

    def __init__(self, search_provider: SearchProvider):
        self._tools: dict[str, dict] = {}
        self._register(
            "web_search",
            "按关键词搜索网页，返回标题、链接与摘要列表",
            WebSearchArgs,
            lambda **kw: search_provider.search(kw["query"], kw["max_results"]),
        )
        self._register(
            "fetch_page",
            "抓取指定网页的正文文本（用于深入阅读与引用取证）",
            FetchPageArgs,
            lambda **kw: fetch_page(kw["url"]),
        )

    def _register(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        func: Callable,
    ) -> None:
        self._tools[name] = {
            "description": description,
            "args_model": args_model,
            "func": func,
        }

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        """生成 OpenAI Function Calling 的 tools 参数。"""
        names = names or list(self._tools)
        out = []
        for n in names:
            t = self._tools[n]
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": n,
                        "description": t["description"],
                        "parameters": t["args_model"].model_json_schema(),
                    },
                }
            )
        return out

    def call(self, name: str, **kwargs) -> str:
        """执行工具：参数先过 pydantic 校验；任何异常都转成模型可读的错误 JSON。"""
        t = self._tools.get(name)
        if t is None:
            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        try:
            args = t["args_model"](**kwargs)
            result = t["func"](**args.model_dump())
        except Exception as e:
            return json.dumps({"error": f"工具执行失败: {e}"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

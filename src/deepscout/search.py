"""搜索供应商抽象：ddgs（免 key）/ tavily / mock（离线演示语料）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from .settings import Settings


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 5) -> list[dict]: ...


class DDGSProvider:
    """DuckDuckGo 搜索（ddgs 包），无需 API key。国内网络可能不稳定，可切换 tavily。"""

    name = "ddgs"

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        from ddgs import DDGS  # 延迟导入

        out: list[dict] = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    out.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href") or r.get("url", ""),
                            "snippet": r.get("body") or r.get("snippet") or r.get("content", ""),
                        }
                    )
        except Exception:
            # 网络失败时返回空结果，由上层决定是否换关键词重试
            return out
        return out


class TavilyProvider:
    """Tavily 搜索 API（有免费额度），对 LLM 友好。"""

    name = "tavily"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Tavily 需要设置 TAVILY_API_KEY")
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        import httpx

        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=20,
        )
        resp.raise_for_status()
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
            for r in resp.json().get("results", [])
        ]


def _load_sample_index() -> list[dict]:
    from importlib import resources

    pkg = resources.files("deepscout") / "sample_data" / "index.json"
    return json.loads(pkg.read_text("utf-8"))


class MockSearchProvider:
    """离线演示：在内置示例语料上做关键词匹配，让全流程在无网络环境下可跑通。"""

    name = "mock"

    def __init__(self):
        self.entries = _load_sample_index()

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        scored = []
        for e in self.entries:
            score = sum(1 for kw in e["keywords"] if kw in query)
            scored.append((score, e))
        scored.sort(key=lambda x: -x[0])
        top = scored[:max_results]
        if top and top[0][0] == 0:
            # 无命中时返回第一条，保证流程可继续（演示语料库只有 4 条）
            top = scored[:1]
        return [
            {"title": e["title"], "url": e["url"], "snippet": e["snippet"]} for _, e in top
        ]


def make_search(settings: Settings, mock: bool = False) -> SearchProvider:
    provider = "mock" if mock else settings.search_provider
    if provider == "mock":
        return MockSearchProvider()
    if provider == "tavily":
        return TavilyProvider(settings.tavily_api_key)
    if provider == "ddgs":
        return DDGSProvider()
    raise ValueError(f"未知搜索供应商: {provider}（可选 ddgs / tavily / mock）")

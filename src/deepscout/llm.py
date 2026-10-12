"""LLM 客户端抽象层。

三层设计：
1. ``OpenAICompatClient`` —— 对接任意 OpenAI 兼容服务（DeepSeek / GLM / Qwen / 本地 vLLM），
   换供应商只需改环境变量，业务代码零改动。
2. ``DiskCacheClient`` —— 装饰器模式包一层磁盘缓存（请求哈希 -> 响应），
   开发期重复调试不重复花钱，评测期可一键关闭。
3. ``FakeLLM`` —— 确定性脚本回放，让整条流水线在无网络、无 API key 的环境下可运行，
   也是单元测试与 CI 的基础。
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from .settings import Settings


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCallRequest:
    name: str
    arguments: dict
    id: str = ""


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    usage: dict = field(default_factory=dict)  # prompt_tokens / completion_tokens
    model: str = ""
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "tool_calls": [tc.__dict__ for tc in self.tool_calls],
            "usage": self.usage,
            "model": self.model,
            "latency_s": self.latency_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LLMResponse:
        return cls(
            content=d.get("content"),
            tool_calls=[ToolCallRequest(**tc) for tc in d.get("tool_calls", [])],
            usage=d.get("usage", {}),
            model=d.get("model", ""),
            latency_s=d.get("latency_s", 0.0),
        )


class LLMClient(Protocol):
    name: str

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse: ...


class OpenAICompatClient:
    """任意 OpenAI 兼容 Chat Completions 服务的最小封装。"""

    name = "openai-compat"

    def __init__(self, model: str, base_url: str, api_key: str, timeout: float = 90.0):
        from openai import OpenAI  # 延迟导入，离线环境不触发

        if not api_key:
            raise LLMError("缺少 API key：请设置 DEEPSCOUT_LLM_API_KEY 或使用 --mock 离线模式")
        self.model = model
        self._cli = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
        t0 = time.perf_counter()
        try:
            resp = self._cli.chat.completions.create(**kwargs)
        except Exception as e:  # openai 抛的异常类型较多，统一收口
            raise LLMError(f"LLM 调用失败: {e}") from e
        latency = time.perf_counter() - t0

        msg = resp.choices[0].message
        tool_calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCallRequest(name=tc.function.name, arguments=args, id=tc.id or ""))
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens or 0,
                "completion_tokens": resp.usage.completion_tokens or 0,
            }
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            usage=usage,
            model=self.model,
            latency_s=latency,
        )


class DiskCacheClient:
    """按 (模型, 消息, 工具, 温度) 哈希缓存响应，开发期重复请求不再计费。"""

    name = "cached"

    def __init__(self, inner: LLMClient, cache_dir: Path, enabled: bool = True):
        self.inner = inner
        self.model = getattr(inner, "model", "")
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, messages: list[dict], tools, temperature: float) -> Path:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "tools": tools, "temperature": temperature},
            ensure_ascii=False,
            sort_keys=True,
        )
        return self.cache_dir / (hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32] + ".json")

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        if not self.enabled:
            return self.inner.complete(messages, tools, temperature)
        key = self._key(messages, tools, temperature)
        if key.exists():
            return LLMResponse.from_dict(json.loads(key.read_text("utf-8")))
        resp = self.inner.complete(messages, tools, temperature)
        key.write_text(json.dumps(resp.to_dict(), ensure_ascii=False, indent=2), "utf-8")
        return resp


class FakeLLM:
    """确定性回放：按提示词中的「角色标记」匹配剧本并按序消费，最后一个响应循环复用。"""

    name = "fake"
    model = "fake-model"

    def __init__(self, script: dict[str, list[LLMResponse]]):
        self._script = {k: list(v) for k, v in script.items()}
        self._cursor: dict[str, int] = {}

    def _match_marker(self, messages: list[dict]) -> str | None:
        text = " ".join(str(m.get("content") or "") for m in messages)
        for marker in self._script:
            if marker in text:
                return marker
        return None

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        marker = self._match_marker(messages)
        if marker is None:
            head = " ".join(str(m.get("content") or "")[:60] for m in messages)
            raise LLMError(f"FakeLLM 没有匹配到剧本标记，请检查 prompt 是否包含已知角色标记: {head}")
        i = self._cursor.get(marker, 0)
        self._cursor[marker] = i + 1
        resp = self._script[marker][min(i, len(self._script[marker]) - 1)]
        return replace(copy.deepcopy(resp), latency_s=0.01)


def make_llm(settings: Settings, mock: bool = False) -> LLMClient:
    """根据配置组装客户端：mock -> FakeLLM；真实 -> OpenAI 兼容 + 可选磁盘缓存。"""
    if mock:
        from .fake import default_fake_script

        return FakeLLM(default_fake_script())
    inner = OpenAICompatClient(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    return DiskCacheClient(inner, Path(".cache/llm"), enabled=settings.llm_cache)

"""从 LLM 输出中稳健地抽取 JSON。

大模型经常在 JSON 前后包裹说明文字或 ``` 代码块，直接 json.loads 会频繁失败。
策略：先剥代码围栏，再从第一个 '{' 或 '[' 开始尝试严格解码，失败则向后扫描。
"""

from __future__ import annotations

import json
import re

_FENCE_HEAD = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?")
_FENCE_TAIL = re.compile(r"\n?```\s*$")


def extract_json(text: str | None):
    """从任意 LLM 文本中抽取第一个合法 JSON 对象/数组；失败返回 None。"""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = _FENCE_HEAD.sub("", t)
        t = _FENCE_TAIL.sub("", t).strip()

    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(t[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None

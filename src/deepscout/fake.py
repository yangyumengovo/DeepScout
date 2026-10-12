"""离线剧本：FakeLLM 的确定性响应，配合 mock 搜索与内置演示语料，
让整条流水线在无网络、无 API key 的环境下端到端跑通（CI 与本地演示的基础）。
"""

from __future__ import annotations

import json

from .llm import LLMResponse, ToolCallRequest
from .prompts import MARKER_OUTLINE, MARKER_PLAN, MARKER_SEARCH, MARKER_VERIFY, MARKER_WRITE


def _usage(p: int = 12, c: int = 30) -> dict:
    return {"prompt_tokens": p, "completion_tokens": c}


def default_fake_script() -> dict[str, list[LLMResponse]]:
    plan = LLMResponse(
        content=json.dumps(
            {
                "subqueries": [
                    "大模型 长上下文 扩展 技术",
                    "RoPE 位置编码 外推",
                    "长文本 评测 基准",
                ]
            },
            ensure_ascii=False,
        ),
        model="fake",
        usage=_usage(),
    )

    # 检索员剧本：第一步发起两次工具调用（真实 Function Calling 路径），
    # 第二步输出调研笔记；后续子查询直接复用最后一步（笔记）。
    search = [
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    name="fetch_page", arguments={"url": "demo://llm-context/rope"}, id="call_1"
                ),
                ToolCallRequest(
                    name="fetch_page", arguments={"url": "demo://llm-context/eval"}, id="call_2"
                ),
            ],
            model="fake",
            usage=_usage(),
        ),
        LLMResponse(
            content=(
                "- 旋转位置编码 RoPE 通过外推支持大模型扩展上下文窗口 [SRC: demo://llm-context/rope]\n"
                "- 长上下文能力需要专门基准衡量，防止「名义窗口大、有效能力弱」 [SRC: demo://llm-context/eval]\n"
                "- 检索到的其余条目相关度一般，未采用"
            ),
            model="fake",
            usage=_usage(),
        ),
    ]

    outline = LLMResponse(
        content=json.dumps(
            {
                "title": "大模型长上下文技术调研",
                "sections": ["背景与挑战", "主流扩展方法", "评测基准与结论"],
            },
            ensure_ascii=False,
        ),
        model="fake",
        usage=_usage(),
    )

    write = LLMResponse(
        content=(
            "大模型正把上下文窗口从数千 token 扩展到百万级，核心挑战在于位置编码外推与"
            "注意力计算开销的增长 [E1]。位置编码方面，RoPE 等旋转位置编码方案通过外推使模型"
            "能够适应训练之外的更长序列 [E2]。为区分「名义窗口」与「有效能力」，长文本任务"
            "需要专门基准进行评测 [E3]。"
        ),
        model="fake",
        usage=_usage(20, 80),
    )

    verify = LLMResponse(
        content=json.dumps(
            {
                "results": [
                    {"evidence_id": "E1", "verdict": "supported", "reason": "语料明确提到长上下文的两大挑战"},
                    {"evidence_id": "E2", "verdict": "supported", "reason": "RoPE 外推语义与证据一致"},
                    {"evidence_id": "E3", "verdict": "partial", "reason": "证据支持需要专门评测，但未点名具体基准"},
                ]
            },
            ensure_ascii=False,
        ),
        model="fake",
        usage=_usage(),
    )

    return {
        MARKER_PLAN: [plan],
        MARKER_SEARCH: search,
        MARKER_OUTLINE: [outline],
        MARKER_WRITE: [write],
        MARKER_VERIFY: [verify],
    }

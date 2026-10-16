from deepscout.evidence import (
    EvidenceIndex,
    build_evidence,
    chunk_text,
    filter_by_anchors,
    get_reranker,
    question_anchors,
)
from deepscout.settings import Settings


def test_chunk_respects_size():
    paras = [f"段落{i}" + "字" * 200 for i in range(8)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, size=500, overlap=80)
    assert len(chunks) >= 3
    assert all(len(c) <= 500 for c in chunks)


def test_build_evidence_dedup_and_ids():
    settings = Settings(_env_file=None)
    pages = [
        {"url": "u1", "title": "t1", "text": "RoPE 旋转位置编码支持上下文外推。"},
        {"url": "u1", "title": "t1-dup", "text": "重复页面应被去重"},
    ]
    chunks = build_evidence(pages, settings)
    assert len(chunks) == 1
    assert chunks[0].eid == "E1"


def test_index_search_ranks_relevant_chunk_first():
    settings = Settings(_env_file=None)
    pages = [
        {"url": "u1", "title": "rope", "text": "RoPE 旋转位置编码通过外推扩展上下文窗口。"},
        {"url": "u2", "title": "kv", "text": "KV Cache 量化与分页管理降低推理显存占用。"},
    ]
    index = EvidenceIndex(build_evidence(pages, settings))
    hits = index.search("RoPE 位置编码")
    assert hits
    assert hits[0][0].url == "u1"


def test_empty_index_returns_empty():
    index = EvidenceIndex([])
    assert index.search("任何查询") == []


def test_get_reranker_degrades_to_none():
    """provider=none 时必须返回 None（主流程不依赖可选组件）。"""
    assert get_reranker(Settings(_env_file=None)) is None


def test_question_anchors_latin_tokens():
    anchors = question_anchors("请对比 DeepScout-Pro 与 ResearchBot Ultra 两款产品")
    assert any("DeepScout-Pro" in a for a in anchors)
    assert any("ResearchBot" in a for a in anchors)


def test_question_anchors_quoted_span():
    anchors = question_anchors("介绍论文《Multi-Agent Consistency Voting》的核心方法")
    assert any("Consistency" in a for a in anchors)


def test_question_anchors_plain_chinese_has_none():
    assert question_anchors("大模型长上下文技术如何扩展？") == []


def test_question_anchors_ignores_generic_words():
    anchors = question_anchors("how to build an agent for the team")
    assert anchors == []


def test_filter_by_anchors():
    settings = Settings(_env_file=None)
    pages = [
        {"url": "u1", "title": "vllm", "text": "vLLM 使用 PagedAttention 管理显存。"},
        {"url": "u2", "title": "generic", "text": "推理引擎可以提升服务吞吐量。"},
    ]
    chunks = build_evidence(pages, settings)
    kept = filter_by_anchors(chunks, ["vLLM"])
    assert [c.url for c in kept] == ["u1"]
    assert filter_by_anchors(chunks, []) == chunks  # 无锚点不过滤
    assert filter_by_anchors(chunks, ["不存在的东西"]) == []  # 全不匹配 -> 空（触发拒答路径）

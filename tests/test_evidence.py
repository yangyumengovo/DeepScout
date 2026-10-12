from deepscout.evidence import EvidenceIndex, build_evidence, chunk_text
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

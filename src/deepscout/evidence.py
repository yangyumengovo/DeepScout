"""证据库：对一次调研收集到的网页正文做分块与混合检索（RAG 的检索层）。

- 分块：按段落边界切分，中文按字符数控制块长（报告类文档天然分段，比分 token 更稳）；
- 检索：BM25（jieba 分词）为基线，可选叠加 bge-small-zh 向量余弦（fastembed，本地免费），
  两路分数归一化后加权融合——即「混合检索」。

为什么自实现 BM25 而不用 rank_bm25：实测发现其 IDF 公式 log((N-df+0.5)/(df+0.5))
在词项恰好出现在一半文档时等于 log(1)=0，小语料（一次调研的证据库通常只有几十块）
上大量词项得分被清零、检索结果为空。改用 Lucene 风格 idf=log(1+(N-df+0.5)/(df+0.5))，
恒为正，兼顾大小语料。单元测试 test_evidence.py 覆盖了这个回归场景。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .settings import Settings


@dataclass
class EvidenceChunk:
    eid: str
    url: str
    title: str
    text: str


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """按段落聚合切分：优先在 \\n\\n 边界断开，块长不超过 size 字符。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            while len(p) > size:  # 单段超长：硬切 + 重叠
                chunks.append(p[:size])
                p = p[size - overlap :]
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def build_evidence(pages: list[dict], settings: Settings) -> list[EvidenceChunk]:
    """pages: [{url, title, text}] -> 去重后分块并编号为 E1..En。"""
    chunks: list[EvidenceChunk] = []
    seen: set[str] = set()
    for page in pages:
        if page.get("url") in seen or not page.get("text"):
            continue
        seen.add(page["url"])
        for piece in chunk_text(page["text"], settings.chunk_size, settings.chunk_overlap):
            chunks.append(
                EvidenceChunk(
                    eid=f"E{len(chunks) + 1}",
                    url=page["url"],
                    title=page.get("title", ""),
                    text=piece,
                )
            )
    return chunks


def _tokenize(text: str) -> list[str]:
    import jieba

    return [w for w in jieba.lcut(text) if w.strip() and w.strip() not in {"。", "，", "、", "；"}]


class BM25:
    """Lucene 风格 BM25：idf = log(1 + (N-df+0.5)/(df+0.5))，恒为正。"""

    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(tokenized_docs)
        self.doc_len = [len(d) for d in tokenized_docs]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf = [Counter(d) for d in tokenized_docs]
        self.df: Counter = Counter()
        for doc in tokenized_docs:
            for w in set(doc):
                self.df[w] += 1

    def _idf(self, w: str) -> float:
        df = self.df.get(w, 0)
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for w in query:
            idf = self._idf(w)
            for i in range(self.N):
                f = self.tf[i].get(w, 0)
                if not f:
                    continue
                norm = self.k1 * (1 - self.b + self.b * (self.doc_len[i] / self.avgdl if self.avgdl else 0))
                scores[i] += idf * f * (self.k1 + 1) / (f + norm)
        return scores


class EvidenceIndex:
    """BM25 + 可选向量的混合检索索引。"""

    def __init__(self, chunks: list[EvidenceChunk], embedder=None):
        self.chunks = chunks
        self.embedder = embedder
        self._bm25 = None
        self._vecs = None
        if chunks:
            self._bm25 = BM25([_tokenize(c.text) for c in chunks])
            if embedder is not None:
                import numpy as np

                self._vecs = np.array(embedder.embed([c.text for c in chunks]))

    def search(self, query: str, k: int = 6) -> list[tuple[EvidenceChunk, float]]:
        if not self.chunks or self._bm25 is None:
            return []
        # BM25 路径用纯 Python：证据库是单次调研的小语料（几十块），
        # 不值得为此引入 numpy 依赖；向量分支启用时 fastembed 自带 numpy。
        bm = self._bm25.get_scores(_tokenize(query))
        bm_max = max(bm) if bm else 0.0
        scores = [s / bm_max if bm_max > 0 else s for s in bm]

        if self._vecs is not None and self.embedder is not None:
            import numpy as np

            qv = np.array(self.embedder.embed([query])[0])
            cos = self._vecs @ qv / (
                np.linalg.norm(self._vecs, axis=1) * np.linalg.norm(qv) + 1e-9
            )
            scores = (0.5 * np.array(scores) + 0.5 * cos).tolist()

        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def get_embedder(settings: Settings):
    """返回向量编码器；未配置或依赖缺失时返回 None（自动退化为纯 BM25）。"""
    if settings.embed_provider != "fastembed":
        return None
    try:
        from fastembed import TextEmbedding

        model = TextEmbedding("BAAI/bge-small-zh-v1.5")

        class _Embedder:
            name = "fastembed/bge-small-zh-v1.5"

            def embed(self, texts: list[str]) -> list[list[float]]:
                return [list(v) for v in model.embed(texts)]

        return _Embedder()
    except Exception:
        return None

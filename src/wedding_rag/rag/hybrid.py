# hybrid.py
from __future__ import annotations

import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi

from wedding_rag.rag import rag_baseline as rb

TAG_RE = re.compile(r"\[c:[a-z0-9]+\]|\(score -?\d+\)")   # drop markup so it doesn't become tokens
TOKEN_RE = re.compile(r"[a-z0-9$]+")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # small; compare bge-reranker-base later

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(TAG_RE.sub(" ", text.lower()))


class Bm25Index:
    def __init__(self, con):
        path = rb.INDEX_DIR / "bm25.pkl"
        if path.exists():
            self.ids, self.bm25 = pickle.loads(path.read_bytes())
        else:
            rows = con.execute(f"SELECT chunk_id, text FROM {rb.CHUNKS_TABLE} ORDER BY chunk_id").fetchall()
            self.ids = [r[0] for r in rows]
            self.bm25 = BM25Okapi([tokenize(r[1]) for r in rows])
            path.write_bytes(pickle.dumps((self.ids, self.bm25)))

    def search(self, query: str, n: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:n]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]


class MultiRetriever:
    """mode: dense | bm25 | hybrid (reciprocal rank fusion)."""

    def __init__(self, mode: str = "hybrid", pool: int = 50, rrf_k: int = 60):
        self.rerank_n = 30
        if mode == "rerank":
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL, max_length=512)
        self.mode, self.pool, self.rrf_k = mode, pool, rrf_k
        self.dense = rb.Retriever()
        self.con = self.dense.con
        self.bm25 = Bm25Index(self.con) if mode != "dense" else None

    def _fetch(self, chunk_id: str, score: float) -> dict:
        tid, text = self.con.execute(
            f"SELECT thread_id, text FROM {rb.CHUNKS_TABLE} WHERE chunk_id = ?", [chunk_id]).fetchone()
        return {"chunk_id": chunk_id, "thread_id": tid, "score": score, "text": text}

    def __call__(self, question: str, k: int = 5) -> list[dict]:
        if self.mode == "dense":
            return self.dense(question, k)
        sparse = self.bm25.search(question, max(self.pool, k))
        if self.mode == "bm25":
            return [self._fetch(c, s) for c, s in sparse[:k]]
        dense = self.dense(question, max(self.pool, k))
        fused: dict[str, float] = {}
        for rank, r in enumerate(dense):
            fused[r["chunk_id"]] = fused.get(r["chunk_id"], 0.0) + 1 / (self.rrf_k + rank + 1)
        for rank, (c, _) in enumerate(sparse):
            fused[c] = fused.get(c, 0.0) + 1 / (self.rrf_k + rank + 1)
        n = self.rerank_n if self.mode == "rerank" else k
        cands = [self._fetch(c, s) for c, s in sorted(fused.items(), key=lambda kv: -kv[1])[:n]]
        if self.mode == "rerank":
            pairs = [(question, TAG_RE.sub(" ", c["text"])) for c in cands]   # strip the [c:id] markup
            for c, s in zip(cands, self.reranker.predict(pairs)):
                c["score"] = float(s)
            cands.sort(key=lambda c: -c["score"])
        return cands[:k]


def make_retriever(mode: str):
    return rb.Retriever() if mode == "dense" else MultiRetriever(mode)
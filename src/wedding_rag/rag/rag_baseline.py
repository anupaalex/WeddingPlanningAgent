# rag_baseline.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import duckdb
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import os, re
from wedding_rag.settings import CFG

MODEL_NAME = CFG.embedding_model
QUERY_PREFIX = CFG.query_prefix
DOC_PREFIX = CFG.doc_prefix
MODEL_MAX_TOKENS = CFG.max_input_tokens
CHUNKER = CFG.chunker
CHUNK_TOKENS = CFG.chunk_tokens
OVERLAP_TOKENS = CFG.overlap_tokens
INDEX_MIN_BODY_CHARS = CFG.index_min_body_chars
THREADS_PATH = CFG.threads_path
DB_PATH = CFG.db_path
INDEX_DIR = CFG.index_dir
CHUNKS_TABLE = CFG.chunks_table
TITLE_PREFIX = CFG.title_prefix
SLUG = re.sub(r"\W+", "_", f"{MODEL_NAME.split('/')[-1]}_{CHUNKER}_c{CHUNK_TOKENS}_o{OVERLAP_TOKENS}")




# ───────── chunker ─────────
def chunk_text(text: str, tokenizer, size: int | None = None, overlap: int | None = None) -> list[str]:
    size = size or CHUNK_TOKENS
    overlap = OVERLAP_TOKENS if overlap is None else overlap
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    if n <= size:
        return [text]
    chunks, start, step = [], 0, size - overlap
    while start < n:
        end = min(start + size, n)
        chunks.append(text[offsets[start][0]: offsets[end - 1][1]])
        if end == n:
            break
        start += step
    return chunks

def n_tok(text: str, tok) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def render_subtree(root: dict) -> str:
    lines, stack = [], [(root, 0)]
    while stack:
        n, d = stack.pop()
        pad = "  " * d
        body = n["body"].replace("\n", f"\n{pad}  ")
        lines.append(f"{pad}- [c:{n['comment_id']}] (score {n['score']}) {body}")
        stack.extend((c, d + 1) for c in reversed(n["replies"]))
    return "\n".join(lines)


def chunk_comment_aware(t: dict, tok) -> list[str]:
    """Pack whole comment subtrees into chunks; the title prefixes every chunk."""
    prefix = f"Thread: {t['title']}\n" if TITLE_PREFIX else ""
    budget = CHUNK_TOKENS - n_tok(prefix, tok) - 10        # margin for joins
    units = ([t["body"]] if t["body"] else []) + \
            [render_subtree(r) for r in t["comments"] + t["orphan_comments"]]
    chunks, cur, cur_n = [], [], 0

    def flush():
        nonlocal cur, cur_n
        if cur:
            chunks.append(prefix + "\n".join(cur))
            cur, cur_n = [], 0

    for u in units:
        n = n_tok(u, tok)
        if n > budget:                                      # one giant subtree: window-split it
            flush()
            chunks.extend(prefix + p for p in chunk_text(u, tok, budget, OVERLAP_TOKENS))
        elif cur_n + n > budget:
            flush()
            cur, cur_n = [u], n
        else:
            cur.append(u)
            cur_n += n
    flush()
    
    return chunks or [t["title"]] 
def build_chunks(model: SentenceTransformer) -> list[dict]:
    tok = model.tokenizer
    tok.model_max_length = 10**9  # silence the "longer than max length" warning; we chunk ourselves
    out = []
    for line in THREADS_PATH.open(encoding="utf-8"):
        t = json.loads(line)
        if t["num_comments_kept"] == 0 and len(t["body"]) < INDEX_MIN_BODY_CHARS:
            continue
        pieces = chunk_comment_aware(t, tok) if CHUNKER == "comment" else chunk_text(t["text"], tok)
        for i, text in enumerate(pieces):
            n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
            assert n_tok + 2 <= MODEL_MAX_TOKENS, f"chunk exceeds model limit ({n_tok})"  # no silent truncation
            out.append({"chunk_id": f"{t['thread_id']}:{i}", "thread_id": t["thread_id"],
                        "year": t["year"], "flair": t["flair"], "text": text, "n_tokens": n_tok})
    return out


# ───────── vector store interface + FAISS flat ─────────
class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...


class FaissFlatStore:
    """Exact cosine search (vectors are normalised, so inner product = cosine)."""
    def __init__(self, dim: int):
        self.index, self.ids = faiss.IndexFlatIP(dim), []

    def add(self, ids, vectors):
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.ids.extend(ids)

    def search(self, vector, k):
        scores, idx = self.index.search(np.ascontiguousarray(vector[None, :], dtype="float32"), k)
        return [(self.ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        (d / "ids.json").write_text(json.dumps(self.ids))

    @classmethod
    def load(cls, d: Path):
        s = cls.__new__(cls)
        s.index = faiss.read_index(str(d / "index.faiss"))
        s.ids = json.loads((d / "ids.json").read_text())
        return s


# ───────── index + retrieve ─────────
def build_index() -> None:
    model = SentenceTransformer(MODEL_NAME)
    chunks = build_chunks(model)
    print(f"{len(chunks)} chunks; max tokens {max(c['n_tokens'] for c in chunks)}")
    vecs = model.encode([DOC_PREFIX + c["text"] for c in chunks], batch_size=64,
                        normalize_embeddings=True, show_progress_bar=True)
    store = FaissFlatStore(vecs.shape[1])
    store.add([c["chunk_id"] for c in chunks], vecs)
    store.save(INDEX_DIR)
    # chunk metadata lives in DuckDB, keyed by chunk_id
    con = duckdb.connect(DB_PATH)
    con.execute(f"DROP TABLE IF EXISTS {CHUNKS_TABLE}")
    con.execute(f"""CREATE TABLE {CHUNKS_TABLE}(chunk_id VARCHAR PRIMARY KEY, thread_id VARCHAR, year INTEGER,
                   flair VARCHAR, text VARCHAR, n_tokens INTEGER)""")
    con.executemany(f"INSERT INTO {CHUNKS_TABLE} VALUES (?,?,?,?,?,?)",
                    [(c["chunk_id"], c["thread_id"], c["year"], c["flair"], c["text"], c["n_tokens"])
                     for c in chunks])
    con.close()
    (INDEX_DIR / "meta.json").write_text(json.dumps({  # never mix indexes built with different settings
        "model": MODEL_NAME, "chunk_tokens": CHUNK_TOKENS, "overlap": OVERLAP_TOKENS,
        "n_chunks": len(chunks)}))


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.store = FaissFlatStore.load(INDEX_DIR)
        self.con = duckdb.connect(DB_PATH, read_only=True)

    def __call__(self, question: str, k: int = 5) -> list[dict]:
        qv = self.model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
        hits = self.store.search(qv, k)
        out = []
        for chunk_id, score in hits:
            tid, text = self.con.execute(
    f"SELECT thread_id, text FROM {CHUNKS_TABLE} WHERE chunk_id = ?", [chunk_id]).fetchone()
            out.append({"chunk_id": chunk_id, "thread_id": tid, "score": score, "text": text})
        return out


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["index"]:
        build_index()
    else:
        from wedding_rag.evaluation.golden import evaluate
        evaluate(Retriever(), k=5, split=None)  # split=None while the set is tiny
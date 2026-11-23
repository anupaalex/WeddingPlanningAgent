# screen_corpus.py
from collections import Counter

import duckdb

from wedding_rag.guardrails import defences as D
from wedding_rag.rag import rag_baseline as rb

con = duckdb.connect(rb.DB_PATH, read_only=True)
rows = con.execute(f"SELECT chunk_id, text FROM {rb.CHUNKS_TABLE}").fetchall()
counts, samples = Counter(), []
for cid, text in rows:
    hits = D.explain(text)
    for rule, _ in hits:
        counts[rule] += 1
    if hits and len(samples) < 15:
        samples.append((cid, hits))
print(f"{len(rows)} chunks; {sum(counts.values())} rule hits: {dict(counts)}")
for cid, hits in samples:
    print(cid, [(r, s[:60]) for r, s in hits])
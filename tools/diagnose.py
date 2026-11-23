# diagnose.py
import json
import re
from pathlib import Path

import duckdb

from wedding_rag.settings import load_config

print("--- wiring in rag_baseline.py ---")
for i, line in enumerate(Path("rag_baseline.py").read_text(encoding="utf-8").splitlines(), 1):
    if re.search(r"CHUNKER|TITLE_PREFIX|chunk_comment_aware|CHUNKS_TABLE|FROM chunks", line):
        print(f"{i:>4}: {line.strip()}")

print("\n--- indexes ---")
for name in ["bge_small_fixed", "bge_small_comment", "bge_small_comment_notitle"]:
    cfg = load_config(name)
    meta_path = cfg.index_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    con = duckdb.connect(cfg.db_path, read_only=True)
    try:
        n, avg_tok, titled = con.execute(
            f"SELECT count(*), avg(n_tokens), "
            f"avg(CASE WHEN text LIKE 'Thread: %' THEN 1 ELSE 0 END) FROM {cfg.chunks_table}"
        ).fetchone()
    except Exception as e:
        n = avg_tok = titled = f"missing ({type(e).__name__})"
    con.close()
    print(f"{name:<28} slug={cfg.slug}")
    print(f"   meta n_chunks={meta and meta['n_chunks']}  table rows={n}  "
          f"avg tokens={avg_tok}  share starting 'Thread:'={titled}")
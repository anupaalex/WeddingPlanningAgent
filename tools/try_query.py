# try_query.py
import json
import textwrap
import time
from pathlib import Path

import duckdb

from wedding_rag.rag.rag_baseline import Retriever, DB_PATH

SNIPPET_CHARS = 500
CANDIDATES = Path("data/golden_candidates.jsonl")


def show(results: list[dict], con, full: bool) -> None:
    for rank, r in enumerate(results, 1):
        title, permalink, ts = con.execute(
            "SELECT title, permalink, created_utc FROM threads WHERE thread_id = ?",
            [r["thread_id"]],
        ).fetchone()
        text = r["text"]
        if not full and len(text) > SNIPPET_CHARS:
            text = text[:SNIPPET_CHARS] + "…"
        text = text if full else text.replace("\n", " ⏎ ")
        print(f"\n#{rank}  score={r['score']:.3f}  {time.gmtime(ts).tm_year}  "
              f"thread={r['thread_id']}  chunk={r['chunk_id']}")
        print(f"    {title}")
        if permalink:
            print(f"    https://www.reddit.com{permalink}")
        print(textwrap.indent(text, "    "))


def main() -> None:
    retriever = Retriever()
    con = duckdb.connect(DB_PATH, read_only=True)
    k, full, last = 5, False, None
    print("type a question | :k 8 | :full (toggle) | :save 1,3 (mark relevant ranks) | :quit")
    while True:
        line = input("\n> ").strip()
        if not line:
            continue
        if line == ":quit":
            break
        if line.startswith(":k "):
            k = int(line[3:])
            continue
        if line == ":full":
            full = not full
            print("full text:", full)
            continue
        if line.startswith(":save"):
            if not last:
                print("run a query first")
                continue
            question, results = last
            ranks = [int(x) for x in line[5:].replace(" ", "").split(",") if x]
            picked = [results[i - 1] for i in ranks if 1 <= i <= len(results)]
            rec = {"question": question, "type": "unknown", "source": "manual", "verified": False,
                   "gold_thread_ids": sorted({p["thread_id"] for p in picked}),
                   "gold_chunk_ids": [p["chunk_id"] for p in picked]}
            with CANDIDATES.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print("saved to", CANDIDATES)
            continue
        results = retriever(line, k)
        last = (line, results)
        show(results, con, full)


if __name__ == "__main__":
    main()
# inspect_corpus.py
import json, random, statistics
from collections import Counter
from pathlib import Path
from wedding_rag.ingestion.arctic_ingest import INJECTION_RE

recs = [json.loads(l) for l in Path("data/threads.jsonl").open(encoding="utf-8")]

print("total approx tokens:", sum(r["approx_tokens"] for r in recs))
print("threads per year:", sorted(Counter(r["year"] for r in recs).items()))

rep = sum(r["num_comments_reported"] for r in recs)
kept = sum(r["num_comments_kept"] for r in recs)
print(f"comments kept / reported: {kept}/{rep} = {kept/rep:.0%}")  # very low → check crawl completeness

print("\n21 biggest threads:")
for r in sorted(recs, key=lambda r: -r["approx_tokens"])[:21]:
    print(f'{r["approx_tokens"]:>6} tok {r["num_comments_kept"]:>4} comments  {r["title"][:80]}')

print("\nflagged threads:")
for r in recs:
    if r["flagged_any"]:
        m = INJECTION_RE.search(r["text"])
        snippet = r["text"][max(0, m.start() - 80): m.end() + 80] if m else "(no match in text)"
        print(r["thread_id"], r["title"][:60], "→", repr(snippet))

zero = [r for r in recs if r["num_comments_kept"] == 0]
print(f"\n{len(zero)} zero-comment threads; body length quartiles:",
      statistics.quantiles([len(r["body"]) for r in zero], n=4))
for r in random.Random(0).sample(zero, 10):
    print(f'{len(r["body"]):>5} chars  {r["title"][:80]}')
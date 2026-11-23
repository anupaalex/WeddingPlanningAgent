# make_golden.py
import argparse
import random

from wedding_rag.evaluation.golden import GOLDEN_PATH, GoldenQ, load_corpus, load_golden

BUCKETS = {"short": (150, 300), "medium": (300, 1500), "long": (1500, 10**9)}
TYPES = {"f": "factoid", "s": "synthesis", "l": "long_thread"}


def next_id() -> str:
    nums = [int(q.id[1:]) for q in load_golden() if q.id[1:].isdigit()]
    return f"q{max(nums, default=0) + 1:03d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", choices=BUCKETS, default="medium")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pool", action="store_true", help="also review other retrieved threads")
    a = ap.parse_args()

    corpus = load_corpus()
    used = {t for q in load_golden() for t in q.gold_thread_ids} if GOLDEN_PATH.exists() else set()
    lo, hi = BUCKETS[a.bucket]
    candidates = [r for r in corpus.values()
                  if r["thread_id"] not in used and r["num_comments_kept"] >= 3
                  and lo <= r["approx_tokens"] < hi and not r["flagged_any"]]
    random.Random(a.seed).shuffle(candidates)

    retriever = None
    if a.pool:
        from wedding_rag.rag.rag_baseline import Retriever
        retriever = Retriever()

    for r in candidates[: a.n]:
        print("\n" + "=" * 80)
        print(f"{r['thread_id']}  {r['created']}  ~{r['approx_tokens']} tokens")
        print(r["text"][:4000])
        q = input("\nQuestion a user might ask that this thread answers (blank = skip): ").strip()
        if not q:
            continue
        qtype = TYPES.get(input("type: [f]actoid / [s]ynthesis / [l]ong_thread (default f): ").strip() or "f", "factoid")
        ref = input("short reference answer, in your own words: ").strip()
        cids = [c.strip() for c in input("comment ids holding the answer (optional, comma-separated): ").split(",") if c.strip()]
        gold = [r["thread_id"]]

        if retriever:
            seen = []
            for hit in retriever(q, 20):
                if hit["thread_id"] not in seen and hit["thread_id"] not in gold:
                    seen.append(hit["thread_id"])
            seen = seen[:5]
            print("\nOther threads the retriever returned. Do any also answer the question?")
            for i, tid in enumerate(seen, 1):
                print(f"  {i}. {tid}  {corpus[tid]['title'][:80]}")
            extra = input("ranks that also answer it (e.g. 1,3; blank = none): ").replace(" ", "")
            gold += [seen[int(x) - 1] for x in extra.split(",") if x.isdigit() and 1 <= int(x) <= len(seen)]

        gq = GoldenQ(id=next_id(), question=q, type=qtype, gold_thread_ids=gold,
                     gold_comment_ids=cids, reference_answer=ref, source="manual")
        with GOLDEN_PATH.open("a", encoding="utf-8") as f:
            f.write(gq.model_dump_json() + "\n")
        print("saved", gq.id)


if __name__ == "__main__":
    main()
# python tools/pool_review.py --profile bge_small_comment_notitle
import json
import shutil
import textwrap

from wedding_rag.evaluation.golden import GOLDEN_PATH, load_corpus, load_golden
from wedding_rag.paths import DATA_DIR
from wedding_rag.rag.hybrid import make_retriever

REVIEWS = DATA_DIR / "pool_reviews.json"           # remembers your answers so you're never asked twice
K_CANDIDATES = 15


def save(qs) -> None:
    GOLDEN_PATH.write_text("".join(q.model_dump_json() + "\n" for q in qs), encoding="utf-8")


def main() -> None:
    shutil.copy(GOLDEN_PATH, GOLDEN_PATH.with_suffix(".bak"))
    corpus = load_corpus()
    reviews = json.loads(REVIEWS.read_text()) if REVIEWS.exists() else {}
    retriever = make_retriever("hybrid")
    qs = load_golden()

    for q in qs:
        if q.type == "unanswerable":
            continue
        seen = reviews.setdefault(q.id, {})
        cands: dict[str, dict] = {}
        for r in retriever(q.question, K_CANDIDATES * 3):
            tid = r["thread_id"]
            if tid not in q.gold_thread_ids and tid not in seen and tid not in cands:
                cands[tid] = r
            if len(cands) >= K_CANDIDATES:
                break
        if not cands:
            continue

        print("\n" + "=" * 80)
        print(f"{q.id} [{q.type}]  {q.question}")
        print(f"reference: {q.reference_answer}")
        print(f"current gold: {q.gold_thread_ids}")
        for tid, r in cands.items():
            while True:
                print(f"\n  thread {tid} ({corpus[tid]['year']})  {corpus[tid]['title'][:90]}")
                print(textwrap.indent(r["text"][:600].replace("\n", " ⏎ "), "    "))
                ans = input("  does this thread answer the question? [y/n/f=full thread/s=next question/q=quit] ").strip().lower()
                if ans == "f":
                    print(textwrap.indent(corpus[tid]["text"][:6000], "    | "))
                    continue
                break
            if ans == "q":
                REVIEWS.write_text(json.dumps(reviews))
                save(qs)
                return
            if ans == "s":
                break
            if ans in ("y", "n"):
                seen[tid] = ans == "y"
                if ans == "y":
                    q.gold_thread_ids.append(tid)
        REVIEWS.write_text(json.dumps(reviews))
        save(qs)                                       # saved after every question


if __name__ == "__main__":
    main()
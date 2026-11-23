# python tools/calibrate_gate.py --profile bge_small_comment_notitle
from wedding_rag.evaluation.golden import load_golden
from wedding_rag.rag.hybrid import make_retriever

retriever = make_retriever("dense")                  # cosine scores are comparable across questions
ans, unans = [], []
for q in load_golden():
    s = retriever(q.question, 1)[0]["score"]
    (unans if q.type == "unanswerable" else ans).append(s)

print(f"answerable n={len(ans)}  unanswerable n={len(unans)}")
if len(unans) < 10:
    print("warning: fewer than 10 unanswerable questions; treat any threshold as a rough guess")
print(" thr   answerable_kept  unanswerable_refused")
for i in range(55, 80):
    t = i / 100
    kept = sum(s >= t for s in ans) / len(ans)
    refused = sum(s < t for s in unans) / len(unans) if unans else float("nan")
    print(f"{t:.2f}   {kept:>14.2f}  {refused:>20.2f}")
# scores.py
from wedding_rag.evaluation.golden import load_golden
from wedding_rag.rag import hybrid

r = hybrid.make_retriever("dense")                  # cosine scores, comparable across questions
rows = [(r(q.question, 1)[0]["score"], q.type, q.id, q.notes, q.question[:60]) for q in load_golden()]
ans = sorted(s for s, t, *_ in rows if t != "unanswerable")
print(f"answerable (n={len(ans)}): min {ans[0]:.3f}  p10 {ans[len(ans) // 10]:.3f}  median {ans[len(ans) // 2]:.3f}\n")
for s, t, i, n, text in sorted(rows):
    if t == "unanswerable":
        print(f"{s:.3f}  {i}  {n:<14} {text}")
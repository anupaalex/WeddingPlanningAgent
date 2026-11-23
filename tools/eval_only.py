# eval_only.py
from wedding_rag.rag import rag_baseline as rb
from wedding_rag.evaluation.golden import evaluate

evaluate(rb.Retriever(), k=5, split=None)
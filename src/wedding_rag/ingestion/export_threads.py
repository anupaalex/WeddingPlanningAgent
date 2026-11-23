# export_threads.py
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import duckdb

DB_PATH = "data/wedding.duckdb"
OUT_PATH = Path("data/threads.jsonl")
LONG_THREAD_TOKENS = 8000  # flag threads that won't fit comfortably in one chunk


def iso_date(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def rows(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def build_tree(comments: list[dict]) -> tuple[list[dict], list[dict]]:
    """comments must be in chronological order. Returns (roots, orphans)."""
    nodes = {c["comment_id"]: {**c, "replies": []} for c in comments}
    roots, orphans = [], []
    for c in comments:
        node = nodes[c["comment_id"]]
        if c["depth"] == 0:                       # reply to the post itself
            roots.append(node)
        elif c["parent_id"] in nodes:
            nodes[c["parent_id"]]["replies"].append(node)
        else:                                     # parent missing or filtered out
            orphans.append(node)
    return roots, orphans


def render(thread: dict, roots: list[dict], orphans: list[dict]) -> str:
    lines = [
        f"# {thread['title']}",
        f"[thread:{thread['thread_id']}] {iso_date(thread['created_utc'])} · score {thread['score']}"
        + (f" · flair: {thread['flair']}" if thread["flair"] else ""),
        "",
    ]
    if thread["body"]:
        lines += [thread["body"], ""]

    def walk(nodes: list[dict]) -> None:
        stack = [(n, 0) for n in reversed(nodes)]  # iterative DFS, keeps order
        while stack:
            n, d = stack.pop()
            pad = "  " * d
            body = n["body"].replace("\n", f"\n{pad}  ")
            lines.append(f"{pad}- [c:{n['comment_id']}] (score {n['score']}) {body}")
            stack.extend((child, d + 1) for child in reversed(n["replies"]))

    walk(roots)
    if orphans:
        lines += ["", "(replies to comments that are no longer available)"]
        walk(orphans)
    return "\n".join(lines)


def max_depth(nodes: list[dict]) -> int:
    best, stack = -1, [(n, 0) for n in nodes]
    while stack:
        n, d = stack.pop()
        best = max(best, d)
        stack.extend((c, d + 1) for c in n["replies"])
    return best


def export() -> None:
    con = duckdb.connect(DB_PATH, read_only=True)
    threads = rows(con, "SELECT * FROM threads ORDER BY created_utc")
    by_thread: dict[str, list[dict]] = {}
    for c in rows(con, "SELECT * FROM comments ORDER BY thread_id, created_utc"):
        by_thread.setdefault(c["thread_id"], []).append(c)
    con.close()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sizes, zero, orphan_total, flagged_threads, long_threads = [], 0, 0, 0, 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for t in threads:
            comments = by_thread.get(t["thread_id"], [])
            roots, orphans = build_tree(comments)
            text = render(t, roots, orphans)
            tokens = len(text) // 4  # rough estimate; swap in a real tokenizer later
            flagged = t["flagged"] or any(c["flagged"] for c in comments)
            record = {
                "thread_id": t["thread_id"],
                "title": t["title"],
                "body": t["body"],
                "permalink": t["permalink"],
                "created": iso_date(t["created_utc"]),
                "year": int(iso_date(t["created_utc"])[:4]),
                "score": t["score"],
                "flair": t["flair"],
                "num_comments_reported": t["num_comments"],
                "num_comments_kept": len(comments),
                "max_depth": max(max_depth(roots), max_depth(orphans)),
                "flagged_any": flagged,
                "comments": roots,
                "orphan_comments": orphans,
                "text": text,
                "approx_tokens": tokens,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            sizes.append(tokens)
            zero += not comments
            orphan_total += len(orphans)
            flagged_threads += flagged
            long_threads += tokens > LONG_THREAD_TOKENS

    print(f"exported {len(sizes)} threads → {OUT_PATH}")
    print(f"threads with zero kept comments: {zero}")
    print(f"approx tokens per thread: median {statistics.median(sizes):.0f}, "
          f"p90 {statistics.quantiles(sizes, n=10)[-1]:.0f}, max {max(sizes)}")
    print(f"threads over {LONG_THREAD_TOKENS} tokens: {long_threads}")
    print(f"orphan comments: {orphan_total}")
    print(f"threads with injection-flagged content: {flagged_threads}")


if __name__ == "__main__":
    export()
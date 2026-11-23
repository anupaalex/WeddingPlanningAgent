# arctic_ingest.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import duckdb
import requests
from pydantic import BaseModel

BASE_URL = "https://arctic-shift.photon-reddit.com"
SUBREDDIT = "AusWeddingPlanning"
RAW_DIR = Path("data/raw")
DB_PATH = "data/wedding.duckdb"
USER_AGENT = "wedding-planner-portfolio/0.1 (personal, non-commercial research)"
PAGE_SIZE = 100
MIN_COMMENT_CHARS = 20
SALT = os.environ.get("AUTHOR_HASH_SALT", "change-me")  # set this in your env

ENDPOINTS = {"posts": "/api/posts/search", "comments": "/api/comments/search"}


# ───────────────────────── request budget (same pattern as the LLM guard) ─────────────────────────
class RequestBudgetExceeded(Exception): ...


class RequestBudget:
    """Hard cap on calls plus a minimum gap between them. Fails closed."""

    def __init__(self, max_requests: int, min_interval_s: float = 1.0):
        self.max_requests = max_requests
        self.min_interval_s = min_interval_s
        self.used = 0
        self._last = 0.0

    def acquire(self) -> None:
        if self.used >= self.max_requests:
            raise RequestBudgetExceeded(f"request cap {self.max_requests} reached")
        wait = self._last + self.min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        self.used += 1  # retries count too


# ───────────────────────── client ─────────────────────────
class ArcticClient:
    def __init__(self, budget: RequestBudget, timeout: int = 30, max_retries: int = 5):
        self.budget, self.timeout, self.max_retries = budget, timeout, max_retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def get(self, path: str, params: dict):
        for attempt in range(self.max_retries):
            self.budget.acquire()
            try:
                r = self.session.get(BASE_URL + path, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()["data"]
                if r.status_code not in (429, 500, 502, 503, 504):
                    r.raise_for_status()  # semantic error: don't retry
            except (requests.ConnectionError, requests.Timeout):
                pass  # transient: fall through to backoff
            time.sleep(min(60, 2**attempt) + random.random())  # backoff with jitter
        raise RuntimeError(f"gave up on {path} after {self.max_retries} attempts")


# ───────────────────────── crawl (raw, resumable) ─────────────────────────
def crawl(client: ArcticClient, kind: str, start: str | None = None, end: str | None = None) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{kind}.jsonl"
    ckpt_path = RAW_DIR / f"{kind}.ckpt.json"
    after = json.loads(ckpt_path.read_text())["after"] if ckpt_path.exists() else start
    total = 0
    with out_path.open("a", encoding="utf-8") as f:
        while True:
            params = {"subreddit": SUBREDDIT, "limit": PAGE_SIZE, "sort": "asc"}
            if after is not None:
                params["after"] = after
            if end:
                params["before"] = end
            rows = client.get(ENDPOINTS[kind], params)
            if not rows:
                break
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            total += len(rows)
            last_ts = int(rows[-1]["created_utc"])
            after = last_ts if last_ts != after else last_ts + 1  # guard against a stuck page
            ckpt_path.write_text(json.dumps({"after": after}))
            print(f"{kind}: {total} rows, up to {time.strftime('%Y-%m-%d', time.gmtime(last_ts))}, "
                  f"{client.budget.used} requests used")
    return total  # boundary rows may repeat; build step dedupes by id


# ───────────────────────── schema ─────────────────────────
class Thread(BaseModel):
    thread_id: str
    title: str
    body: str
    score: int
    num_comments: int
    created_utc: int
    flair: str | None
    permalink: str
    flagged: bool


class Comment(BaseModel):
    comment_id: str
    thread_id: str
    parent_id: str
    depth: int | None  # None = orphan (an ancestor is missing)
    body: str
    score: int
    created_utc: int
    author_hash: str | None
    permalink: str | None
    flagged: bool


INJECTION_RE = re.compile(
    r"ignore (all |any )?(previous|prior|above) instructions|disregard .{0,40}instructions"
    r"|system prompt|you are now|reveal your (prompt|instructions)",
    re.I,
)
BOT_AUTHORS = {"AutoModerator"}
DELETED = {"[deleted]", "[removed]"}


def read_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    out[d["id"]] = d  # dedupe by id, keep the latest copy
    return out


def hash_author(a: str | None) -> str | None:
    if not a or a in DELETED:
        return None
    return hashlib.sha256((SALT + a).encode()).hexdigest()[:12]


# ───────────────────────── normalise ─────────────────────────
def normalise_threads(raw: dict[str, dict]) -> list[Thread]:
    out = []
    for p in raw.values():
        if p.get("removed_by_category"):
            continue
        body = (p.get("selftext") or "").strip()
        body = "" if body in DELETED else body
        title = (p.get("title") or "").strip()
        out.append(Thread(
            thread_id=p["id"], title=title, body=body,
            score=int(p.get("score") or 0), num_comments=int(p.get("num_comments") or 0),
            created_utc=int(p["created_utc"]), flair=p.get("link_flair_text"),
            permalink=p.get("permalink") or "",
            flagged=bool(INJECTION_RE.search(f"{title}\n{body}")),
        ))
    return out


def compute_depths(raw: dict[str, dict]) -> dict[str, int | None]:
    """Top-level comment = depth 0. Comments with a missing ancestor get None."""
    depths: dict[str, int | None] = {}
    for start in raw:
        chain, cur, seen = [], start, set()
        base: int | None = None
        while True:
            if cur in depths:
                base = depths[cur]
                break
            if cur in seen or cur not in raw:  # cycle or missing parent
                base = None
                break
            seen.add(cur)
            chain.append(cur)
            pid = raw[cur].get("parent_id", "")
            if pid.startswith("t3_"):  # parent is the post itself
                base = -1
                break
            cur = pid.removeprefix("t1_")
        for cid in reversed(chain):
            base = None if base is None else base + 1
            depths[cid] = base
    return depths


def normalise_comments(raw: dict[str, dict], thread_ids: set[str]) -> list[Comment]:
    depths = compute_depths(raw)
    out = []
    for c in raw.values():
        body = (c.get("body") or "").strip()
        author = c.get("author")
        if body in DELETED or len(body) < MIN_COMMENT_CHARS or author in BOT_AUTHORS:
            continue
        thread_id = (c.get("link_id") or "").removeprefix("t3_")
        if thread_id not in thread_ids:
            continue
        out.append(Comment(
            comment_id=c["id"], thread_id=thread_id,
            parent_id=(c.get("parent_id") or "").split("_", 1)[-1],
            depth=depths.get(c["id"]), body=body,
            score=int(c.get("score") or 0), created_utc=int(c["created_utc"]),
            author_hash=hash_author(author), permalink=c.get("permalink"),
            flagged=bool(INJECTION_RE.search(body)),
        ))
    return out


# ───────────────────────── DuckDB ─────────────────────────
def load_duckdb(threads: list[Thread], comments: list[Comment]) -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS threads; DROP TABLE IF EXISTS comments;")
    con.execute("""CREATE TABLE threads(thread_id VARCHAR PRIMARY KEY, title VARCHAR, body VARCHAR,
        score INTEGER, num_comments INTEGER, created_utc BIGINT, flair VARCHAR,
        permalink VARCHAR, flagged BOOLEAN)""")
    con.execute("""CREATE TABLE comments(comment_id VARCHAR PRIMARY KEY, thread_id VARCHAR,
        parent_id VARCHAR, depth INTEGER, body VARCHAR, score INTEGER, created_utc BIGINT,
        author_hash VARCHAR, permalink VARCHAR, flagged BOOLEAN)""")
    con.executemany("INSERT INTO threads VALUES (?,?,?,?,?,?,?,?,?)",
                    [tuple(t.model_dump().values()) for t in threads])
    con.executemany("INSERT INTO comments VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [tuple(c.model_dump().values()) for c in comments])
    con.close()


def stats_report() -> None:
    con = duckdb.connect(DB_PATH, read_only=True)
    q = lambda sql: con.execute(sql).fetchall()
    print("threads, comments:", q("SELECT (SELECT count(*) FROM threads), (SELECT count(*) FROM comments)")[0])
    print("date range:", q("""SELECT to_timestamp(min(created_utc))::DATE, to_timestamp(max(created_utc))::DATE
                              FROM threads""")[0])
    print("threads per year:", q("""SELECT year(to_timestamp(created_utc)) y, count(*) FROM threads
                                    GROUP BY y ORDER BY y"""))
    print("comments/thread (median, p90):", q("""SELECT median(n), quantile_cont(n, 0.9) FROM
                                    (SELECT count(*) n FROM comments GROUP BY thread_id)""")[0])
    print("depth distribution:", q("SELECT depth, count(*) FROM comments GROUP BY depth ORDER BY depth LIMIT 8"))
    print("orphan comments (depth NULL):", q("SELECT count(*) FROM comments WHERE depth IS NULL")[0][0])
    print("injection-flagged: threads",
          q("SELECT count(*) FROM threads WHERE flagged")[0][0],
          "comments", q("SELECT count(*) FROM comments WHERE flagged")[0][0])
    con.close()


def build() -> None:
    raw_threads = read_jsonl(RAW_DIR / "posts.jsonl")
    raw_comments = read_jsonl(RAW_DIR / "comments.jsonl")
    threads = normalise_threads(raw_threads)
    comments = normalise_comments(raw_comments, {t.thread_id for t in threads})
    load_duckdb(threads, comments)
    stats_report()


# ───────────────────────── smoke test + volume check ─────────────────────────
def smoke(client: ArcticClient) -> None:
    posts = client.get(ENDPOINTS["posts"], {"subreddit": SUBREDDIT, "limit": 3, "sort": "asc"})
    print("posts returned:", len(posts))
    if posts:
        print("post fields:", sorted(posts[0].keys()))
        nxt = client.get(ENDPOINTS["posts"], {"subreddit": SUBREDDIT, "limit": 3, "sort": "asc",
                                              "after": posts[-1]["created_utc"]})
        print("paging with epoch `after` worked:", len(nxt), "rows")
    comments = client.get(ENDPOINTS["comments"], {"subreddit": SUBREDDIT, "limit": 3, "sort": "asc"})
    print("comments returned:", len(comments))
    if comments:
        print("comment fields:", sorted(comments[0].keys()))


def volume(client: ArcticClient) -> None:
    for kind, path in (("posts", "/api/posts/search/aggregate"),
                       ("comments", "/api/comments/search/aggregate")):
        rows = client.get(path, {"subreddit": SUBREDDIT, "aggregate": "created_utc", "frequency": "year"})
        print(kind, {r["key"]: r["doc_count"] for r in rows})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "volume", "crawl", "build"])
    ap.add_argument("--kind", choices=["posts", "comments"])
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--max-requests", type=int, default=2000)
    ap.add_argument("--min-interval", type=float, default=1.0)
    a = ap.parse_args()

    if a.cmd == "build":
        build()
    else:
        client = ArcticClient(RequestBudget(a.max_requests, a.min_interval))
        if a.cmd == "smoke":
            smoke(client)
        elif a.cmd == "volume":
            volume(client)
        else:
            assert a.kind, "--kind posts|comments is required for crawl"
            crawl(client, a.kind, a.start, a.end)
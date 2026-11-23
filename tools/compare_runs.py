# compare_runs.py
import re
import sys

pat = re.compile(r"^(q\d+) \[(\w+)\] recall@\d+=([\d.]+) first_hit_rank=(\w+)")


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        m = pat.match(line.strip())
        if m:
            out[m[1]] = (m[2], float(m[3]), m[4])
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
common = [q for q in a if q in b]
better = [q for q in common if b[q][1] > a[q][1]]
worse = [q for q in common if b[q][1] < a[q][1]]
print(f"B better on {len(better)}, worse on {len(worse)}, same on {len(common) - len(better) - len(worse)}")
for q in worse:
    print("worse :", q, a[q], "→", b[q])
for q in better:
    print("better:", q, a[q], "→", b[q])
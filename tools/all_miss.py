# python tools/all_miss.py a.txt b.txt c.txt
import re
import sys

pat = re.compile(r"^(q\d+) \[(\w+)\] recall@\d+=([\d.]+)")


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        m = pat.match(line.strip())
        if m:
            out[m[1]] = (m[2], float(m[3]))
    return out


runs = [load(p) for p in sys.argv[1:]]
qs = sorted(set.intersection(*[set(r) for r in runs]))
best = {q: max(r[q][1] for r in runs) for q in qs}
allmiss = [q for q in qs if best[q] == 0]
print(f"{len(qs)} questions")
for p, r in zip(sys.argv[1:], runs):
    print(f"  {p}: mean recall {sum(r[q][1] for q in qs) / len(qs):.2f}")
print(f"oracle (best of all variants per question): {sum(best.values()) / len(qs):.2f}")
print(f"missed by every variant: {len(allmiss)} -> {allmiss}")
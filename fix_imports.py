# fix_imports.py   python fix_imports.py [--old data_extraction]          dry run: shows every change
#                  python fix_imports.py --apply [--old data_extraction]  rewrites in place (commit with git first)
import argparse
import re
from pathlib import Path

ROOT = Path(".")
PKG_DIR = ROOT / "src" / "wedding_rag"
SCAN = [PKG_DIR, ROOT / "tools", ROOT / "tests"]
SETTINGS = "wedding_rag.settings"
SETTINGS_ALIASES = {"config", "app_secret", "app_secrets"}
RENAMED = {"guard": "budget", "mcp_server": "server"}        # old module name -> its new file name
IMPORT = re.compile(r"^(\s*)import (\w+)(?: as (\w+))?\s*$")
FROM = re.compile(r"^(\s*)from (\w+) import (.+)$")
SUSPECT = re.compile(r"""["'](?:data|logs|config|golden)[/\\]|["'](?:config\.yaml|redteam_v1\.jsonl|ledger\.db)["']""")


def build_map() -> dict:
    modmap = {}
    for p in PKG_DIR.rglob("*.py"):
        if p.name != "__init__.py":
            modmap.setdefault(p.stem, ".".join(p.relative_to(ROOT / "src").with_suffix("").parts))
    for old, new in RENAMED.items():
        if old not in modmap and new in modmap:
            modmap[old] = modmap[new]
    for alias in SETTINGS_ALIASES:
        modmap[alias] = SETTINGS
    return modmap


def rewrite_line(line: str, modmap: dict) -> str:
    m = IMPORT.match(line)
    if m:
        ind, name, alias = m.groups()
        if name in modmap:
            pkg, leaf = modmap[name].rsplit(".", 1)
            alias = alias or name
            return f"{ind}from {pkg} import {leaf}" + (f" as {alias}" if alias != leaf else "")
        return line
    m = FROM.match(line)
    if m:
        ind, name, rest = m.groups()
        if name == "secrets" and re.search(r"load_secrets|Secrets", rest):     # your own secrets module, not the stdlib one
            return f"{ind}from {SETTINGS} import {rest}"
        if name in modmap:
            return f"{ind}from {modmap[name]} import {rest}"
    return line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--old", help="the old flat folder, to spot imports that have no new home")
    a = ap.parse_args()
    modmap = build_map()
    server = modmap.get("mcp_server")
    old_names = {p.stem for p in Path(a.old).glob("*.py")} if a.old else set()
    files = [p for base in SCAN if base.exists() for p in sorted(base.rglob("*.py"))]
    unresolved = {}
    for p in files:
        text = p.read_text(encoding="utf-8")
        lines = text.split("\n")
        new_lines = [rewrite_line(l, modmap) for l in lines]
        new = "\n".join(new_lines)
        if server:
            new = new.replace('["mcp_server.py"]', f'["-m", "{server}"]')
        if new != text:
            print(f"{'rewrote' if a.apply else 'would rewrite'}: {p}")
            for o, n in zip(lines, new.split("\n")):
                if o != n:
                    print(f"    - {o.strip()}\n    + {n.strip()}")
            if a.apply:
                p.write_text(new, encoding="utf-8")
        for l in new.split("\n"):
            m = IMPORT.match(l) or FROM.match(l)
            if m and m.group(2) in old_names and m.group(2) not in modmap:
                unresolved.setdefault(m.group(2), set()).add(str(p))
    if a.apply:
        for d in [PKG_DIR, *[x for x in PKG_DIR.rglob("*") if x.is_dir() and any(x.glob("*.py"))]]:
            (d / "__init__.py").touch()
    if unresolved:
        print("\nimports of old modules that have no new home (move the file, or fix by hand):")
        for name, where in unresolved.items():
            print(f"  {name}: {', '.join(sorted(where))}")
    print("\nlines that may still use relative paths (see the table below):")
    for p in files:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if SUSPECT.search(line):
                print(f"  {p}:{i}: {line.strip()[:110]}")


main()
# show_profiles.py
from wedding_rag.settings import load_config

for name in ["bge_small_fixed", "bge_small_comment", "bge_small_comment_notitle"]:
    c = load_config(name)
    print(f"{name:<28} chunker={c.chunker:<8} title_prefix={c.title_prefix!s:<5} slug={c.slug}")
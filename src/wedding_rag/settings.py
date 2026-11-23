from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator

from wedding_rag.paths import CONFIG_DIR, ROOT

CONFIG_PATH = CONFIG_DIR / "config.yaml"
LOCAL_SECRETS_PATH = CONFIG_DIR / "config.local.yaml"


def _abs(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    profile: str
    embedding_model: str
    query_prefix: str = ""
    doc_prefix: str = ""
    max_input_tokens: int
    chunker: Literal["fixed", "comment"] = "fixed"
    title_prefix: bool = True
    retriever_mode: Literal["dense", "bm25", "hybrid", "rerank"] = "hybrid"
    chunk_tokens: int
    overlap_tokens: int
    index_min_body_chars: int = 200
    threads_path: Path = Path("data/threads.jsonl")
    db_path: str = "data/wedding.duckdb"
    index_root: Path = Path("data/index")

    @field_validator("threads_path", "index_root")
    @classmethod
    def _abs_paths(cls, v: Path) -> Path:
        return _abs(v)

    @field_validator("db_path")
    @classmethod
    def _abs_db(cls, v: str) -> str:
        return str(_abs(Path(v)))

    @model_validator(mode="after")
    def _check(self):
        if self.chunk_tokens + 2 > self.max_input_tokens:
            raise ValueError(f"chunk_tokens {self.chunk_tokens} won't fit in "
                             f"max_input_tokens {self.max_input_tokens}: it would be silently truncated")
        if not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError("overlap_tokens must be >= 0 and < chunk_tokens")
        return self

    @property
    def slug(self) -> str:
        tp = "_notitle" if (self.chunker == "comment" and not self.title_prefix) else ""
        return re.sub(r"\W+", "_", f"{self.embedding_model.split('/')[-1]}_{self.chunker}{tp}"
                                   f"_c{self.chunk_tokens}_o{self.overlap_tokens}")

    @property
    def index_dir(self) -> Path:
        return self.index_root / self.slug

    @property
    def chunks_table(self) -> str:
        return f"chunks_{self.slug}"


def load_config(profile: str | None = None, path: Path = CONFIG_PATH) -> RagConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = profile or raw["active"]
    if name not in raw["profiles"]:
        raise KeyError(f"unknown profile {name!r}; available: {sorted(raw['profiles'])}")
    merged = {**raw["base"], **(raw["profiles"][name] or {})}
    return RagConfig(profile=name, **merged)


def _profile_from_argv() -> str | None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--profile")
    ns, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return ns.profile


# ───────── secrets ─────────
class Secrets(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gemini_api_key: SecretStr


def _check_ignored() -> None:
    if LOCAL_SECRETS_PATH.exists() and (ROOT / ".git").exists():
        gi = ROOT / ".gitignore"
        lines = {l.strip() for l in gi.read_text(encoding="utf-8").splitlines()} if gi.exists() else set()
        if not lines & {"config.local.yaml", "config/config.local.yaml", "*.local.yaml", "config/*.local.yaml"}:
            raise RuntimeError(f"{LOCAL_SECRETS_PATH.name} holds a secret but is not in .gitignore; add it first")


def load_secrets() -> Secrets:
    _check_ignored()
    data = (yaml.safe_load(LOCAL_SECRETS_PATH.read_text(encoding="utf-8"))
            if LOCAL_SECRETS_PATH.exists() else None) or {}
    if "gemini_api_key" not in data and os.environ.get("GEMINI_API_KEY"):
        data["gemini_api_key"] = os.environ["GEMINI_API_KEY"]
    if "gemini_api_key" not in data:
        raise RuntimeError("No Gemini key: add gemini_api_key to config/config.local.yaml "
                           "(or set the GEMINI_API_KEY environment variable)")
    return Secrets(**data)


CFG = load_config(_profile_from_argv())
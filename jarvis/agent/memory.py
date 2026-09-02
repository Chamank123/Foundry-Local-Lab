"""memory.py — the only module that writes anything, anywhere.

One remembered fact, one dated markdown file, under memory/. It cannot write
outside that directory: every path is resolved and re-checked before it is
opened. The vault is read-only and stays that way.

Nothing here writes silently. Every write returns exactly what was written so
the caller can say it out loud.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from . import data
except ImportError:  # pragma: no cover
    import data  # type: ignore

MEMORY_DIR = data.JARVIS_ROOT / "memory"
MAX_FACT = 600


@dataclass
class Written:
    path: Path
    fact: str
    when: str

    @property
    def rel(self) -> str:
        return f"memory/{self.path.name}"


def _slug(text: str, limit: int = 48) -> str:
    keep = [c if c.isalnum() else "-" for c in text.lower()]
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")[:limit] or "note"


def remember(fact: str, source: str = "asked directly") -> Written:
    """Write one fact. Returns what was written; the caller must say it aloud."""
    fact = " ".join(fact.split())[:MAX_FACT]
    if not fact:
        raise ValueError("nothing to remember")

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    when = data.today().isoformat()
    stem = f"{when}-{_slug(fact)}"
    path = (MEMORY_DIR / f"{stem}.md").resolve()

    # Belt and braces: never write outside memory/, whatever the fact contains.
    if path.parent != MEMORY_DIR.resolve():
        raise ValueError("refusing to write outside memory/")

    n = 2
    while path.exists():
        path = (MEMORY_DIR / f"{stem}-{n}.md").resolve()
        n += 1

    path.write_text(
        f"---\ndate: {when}\nsource: {source}\n---\n\n{fact}\n", encoding="utf-8"
    )
    return Written(path=path, fact=fact, when=when)


def recall(limit: int = 20) -> list[dict]:
    """Everything remembered so far, newest first."""
    if not MEMORY_DIR.exists():
        return []
    out = []
    for path in sorted(MEMORY_DIR.glob("*.md"), reverse=True)[:limit]:
        text = path.read_text(encoding="utf-8", errors="replace")
        body = text.split("---", 2)[-1].strip()
        date = re.search(r"^date:\s*(.+)$", text, re.M)
        out.append({
            "file": f"memory/{path.name}",
            "date": date.group(1).strip() if date else "",
            "fact": body,
        })
    return out

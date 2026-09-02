"""vault.py — folders become a searchable graph.

Read-only, always. Every file here is opened 'rb' and never written. The
indexer walks the roots data.py hands it, parses markdown/text/PDF, and turns
[[wikilinks]] into edges.

Failures are collected, never swallowed: a file that cannot be read becomes a
warning the UI can say out loud. Silence is the one outcome not allowed.

    python3 -m agent.vault      index and print what was found
"""

from __future__ import annotations

import math
import re
import time
import zlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

try:  # works both as `python3 -m agent.vault` and as a plain import
    from . import data
except ImportError:  # pragma: no cover
    import data  # type: ignore

MAX_FILE_BYTES = 2 * 1024 * 1024
TEXT_EXT = {".md", ".markdown", ".txt", ".text"}
PDF_EXT = {".pdf"}
SKIP_DIRS = {
    "node_modules", ".git", ".obsidian", ".trash", ".stfolder", "__pycache__",
    ".venv", "venv", "env", "dist", "build", ".next", ".cache", ".idea",
    ".vscode", "site-packages", ".DS_Store",
}

WIKILINK = re.compile(r"\[\[([^\[\]|#]+)(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]")
TAG = re.compile(r"(?:^|\s)#([A-Za-z][\w/-]{1,40})")
WORD = re.compile(r"[a-z0-9][a-z0-9'’_-]{1,}")

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "was", "were", "are",
    "not", "but", "you", "your", "has", "have", "had", "its", "it's", "they",
    "them", "their", "our", "out", "about", "into", "than", "then", "there",
    "what", "when", "which", "who", "will", "would", "been", "being", "does",
    "did", "how", "all", "any", "can", "one", "two", "per", "via", "off",
}

# Folder name -> type, when the file does not declare one itself.
FOLDER_TYPES = {
    "clients": "client", "client": "client", "people": "person",
    "person": "person", "contacts": "person", "projects": "project",
    "project": "project", "concepts": "concept", "sops": "sop",
    "invoices": "invoice", "proposals": "proposal", "calls": "call",
    "meetings": "call", "notes": "note", "briefs": "brief",
    "campaigns": "campaign",
}


@dataclass
class Node:
    id: str
    title: str
    type: str
    path: str
    rel: str
    ext: str
    size: int
    mtime: float
    text: str = ""
    front: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)  # raw wikilink targets
    warning: str = ""
    degree: int = 0

    def excerpt(self, limit: int = 240) -> str:
        body = " ".join(self.text.split())
        return body[:limit] + ("…" if len(body) > limit else "")

    def card(self) -> dict:
        return {
            "id": self.id, "title": self.title, "type": self.type,
            "rel": self.rel, "degree": self.degree, "front": self.front,
            "tags": self.tags, "excerpt": self.excerpt(), "warning": self.warning,
        }


@dataclass
class Vault:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    adj: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    warnings: list[str] = field(default_factory=list)
    mode: str = "demo"
    roots: list[str] = field(default_factory=list)
    unresolved: Counter = field(default_factory=Counter)
    skipped: int = 0
    built_at: float = 0.0
    _df: Counter = field(default_factory=Counter)
    _tf: dict[str, Counter] = field(default_factory=dict)

    # -- shape -------------------------------------------------------------
    def counts_by_type(self) -> list[tuple[str, int]]:
        c = Counter(n.type for n in self.nodes.values())
        return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))

    def hubs(self, limit: int = 10) -> list[Node]:
        return sorted(
            self.nodes.values(), key=lambda n: (-n.degree, n.title)
        )[:limit]

    def get(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    # -- search ------------------------------------------------------------
    def search(self, query: str, limit: int = 8) -> list[tuple[Node, float]]:
        """Plain tf-idf over titles and bodies. No model needed — this is also
        what the router falls back to when no LLM is reachable."""
        terms = [t for t in tokenize(query) if t not in STOPWORDS]
        if not terms or not self.nodes:
            return []
        n_docs = len(self.nodes)
        scored: list[tuple[Node, float]] = []
        for node_id, tf in self._tf.items():
            node = self.nodes[node_id]
            score = 0.0
            for term in terms:
                if term not in tf:
                    continue
                idf = math.log(1 + n_docs / (1 + self._df[term]))
                score += (1 + math.log(tf[term])) * idf
            if score <= 0:
                continue
            title_words = set(tokenize(node.title))
            score *= 1 + 0.6 * len(title_words & set(terms))
            score *= 1 + 0.02 * min(node.degree, 20)  # hubs break ties
            scored.append((node, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].title))
        return scored[:limit]

    def relevance(self, query: str) -> float:
        """0..1 — how much this vault has to say about a question. The router
        uses it to tell a question about the files from a bit of conversation."""
        hits = self.search(query, limit=3)
        if not hits:
            return 0.0
        top = hits[0][1]
        return max(0.0, min(1.0, top / 12.0))

    # -- graph -------------------------------------------------------------
    def shortest_path(self, a: str, b: str) -> list[str]:
        if a not in self.nodes or b not in self.nodes:
            return []
        if a == b:
            return [a]
        prev: dict[str, str] = {a: a}
        queue = deque([a])
        while queue:
            cur = queue.popleft()
            for nxt in sorted(self.adj[cur]):
                if nxt in prev:
                    continue
                prev[nxt] = cur
                if nxt == b:
                    path = [b]
                    while path[-1] != a:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                queue.append(nxt)
        return []

    def to_graph(self) -> dict:
        return {
            "mode": self.mode,
            "roots": self.roots,
            "warnings": self.warnings,
            "counts": [{"type": t, "count": c} for t, c in self.counts_by_type()],
            "nodes": [
                {"id": n.id, "title": n.title, "type": n.type,
                 "degree": n.degree, "rel": n.rel}
                for n in self.nodes.values()
            ],
            "edges": [{"s": s, "t": t} for s, t in self.edges],
            "hubs": [{"id": n.id, "title": n.title, "type": n.type,
                      "degree": n.degree} for n in self.hubs(10)],
            "unresolved": self.unresolved.most_common(10),
            "skipped": self.skipped,
        }


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.lower())


# ------------------------------------------------------------------ parsing --
def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Minimal YAML-ish frontmatter: flat key: value pairs only. Anything
    fancier is left in the body rather than guessed at."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end]
    rest = text[end + 4:].lstrip("\n")
    front: dict = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip('"').strip("'")
        if value.isdigit():
            front[key.strip()] = int(value)
        else:
            front[key.strip()] = value
    return front, rest


def pdf_text(raw: bytes) -> tuple[str, str]:
    """Pull text out of a PDF with the standard library alone.

    Handles uncompressed and FlateDecode streams. Each literal string is
    decoded on its own terms: a leading UTF-16 BOM means UTF-16BE, otherwise
    latin-1 — decoding the whole stream one way turns em-dashes into mojibake
    and silently breaks the [[links]] inside.

    Returns (text, warning). A PDF this cannot read comes back as a warning,
    not as silence, so the file still appears in the graph, flagged.
    """
    literal = re.compile(rb"\((?:\\.|[^\\()])*\)", re.S)
    unescape = re.compile(rb"\\([()\\])")
    chunks: list[str] = []
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
        blob = match.group(1)
        try:
            blob = zlib.decompress(blob)
        except zlib.error:
            pass  # already plain
        if b"Tj" not in blob and b"TJ" not in blob:
            continue
        for m in literal.finditer(blob):
            body = unescape.sub(rb"\1", m.group(0)[1:-1])
            if body.startswith(b"\xfe\xff"):
                chunks.append(body[2:].decode("utf-16-be", errors="replace"))
            elif body.startswith(b"\xff\xfe"):
                chunks.append(body[2:].decode("utf-16-le", errors="replace"))
            else:
                chunks.append(body.decode("latin-1", errors="replace"))
    text = "\n".join(c for c in chunks if c.strip()).strip()
    if not text:
        return "", "PDF text could not be extracted (scanned, or an encoding this reader does not handle)"
    return text, ""


def infer_type(path: Path, front: dict, title: str) -> str:
    declared = str(front.get("type", "")).strip().lower()
    if declared:
        return declared
    for part in reversed(path.parts[:-1]):
        hit = FOLDER_TYPES.get(part.lower())
        if hit:
            return hit
    low = title.lower()
    for prefix, kind in (("invoice", "invoice"), ("proposal", "proposal"),
                         ("sop", "sop"), ("call", "call"), ("brief", "brief")):
        if low.startswith(prefix):
            return kind
    return "note"


def read_file(path: Path, root: Path) -> Node | None:
    try:
        stat = path.stat()
    except OSError as exc:
        return Node(id=str(path), title=path.stem, type="note", path=str(path),
                    rel=path.name, ext=path.suffix.lower(), size=0, mtime=0.0,
                    warning=f"could not stat: {exc}")
    ext = path.suffix.lower()
    rel = str(path.relative_to(root)) if root in path.parents else path.name
    node = Node(id=rel.replace("\\", "/"), title=path.stem, type="note",
                path=str(path), rel=rel, ext=ext, size=stat.st_size,
                mtime=stat.st_mtime)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        node.warning = f"could not read: {exc}"
        return node

    if ext in PDF_EXT:
        text, warn = pdf_text(raw)
        node.text, node.warning = text, warn
    else:
        text = raw.decode("utf-8", errors="replace")
        node.front, body = parse_frontmatter(text)
        node.text = body

    title = str(node.front.get("title", "")).strip()
    if not title:
        heading = re.search(r"^#\s+(.+)$", node.text, re.M)
        if heading:
            title = heading.group(1).strip()
        elif ext in PDF_EXT and node.text:
            first = node.text.splitlines()[0].strip()
            title = first if 0 < len(first) <= 80 else path.stem
        else:
            title = path.stem
    node.title = title
    node.type = infer_type(path, node.front, title)
    node.tags = sorted(set(TAG.findall(node.text)))
    node.links = [m.strip() for m in WIKILINK.findall(node.text)]
    return node


# ------------------------------------------------------------------ indexing --
def build(source: data.Source | None = None) -> Vault:
    src = source or data.source()
    vault = Vault(mode=src.mode, roots=[str(p) for p in src.paths],
                  warnings=list(src.warnings), built_at=time.time())

    found: list[Node] = []
    for root in src.paths:
        for path in sorted(root.rglob("*")):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext not in TEXT_EXT and ext not in PDF_EXT:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    vault.skipped += 1
                    continue
            except OSError:
                vault.skipped += 1
                continue
            node = read_file(path, root)
            if node is not None:
                found.append(node)

    for node in found:
        if node.id in vault.nodes:  # same relative path under two roots
            node.id = f"{node.id}#{len(vault.nodes)}"
        vault.nodes[node.id] = node

    # -- resolve wikilinks by title, then by filename stem
    by_title: dict[str, str] = {}
    by_stem: dict[str, str] = {}
    for node in vault.nodes.values():
        by_title.setdefault(node.title.strip().lower(), node.id)
        by_stem.setdefault(Path(node.rel).stem.strip().lower(), node.id)

    seen: set[tuple[str, str]] = set()
    for node in vault.nodes.values():
        for target in node.links:
            key = target.strip().lower()
            other = by_title.get(key) or by_stem.get(key)
            if other is None:
                vault.unresolved[target.strip()] += 1
                continue
            if other == node.id:
                continue
            pair = (node.id, other) if node.id < other else (other, node.id)
            if pair in seen:
                continue
            seen.add(pair)
            vault.edges.append(pair)
            vault.adj[pair[0]].add(pair[1])
            vault.adj[pair[1]].add(pair[0])

    for node in vault.nodes.values():
        node.degree = len(vault.adj[node.id])

    # -- search index
    for node in vault.nodes.values():
        terms = Counter(t for t in tokenize(f"{node.title} {node.text}")
                        if t not in STOPWORDS)
        vault._tf[node.id] = terms
        for term in terms:
            vault._df[term] += 1

    unreadable = [n for n in vault.nodes.values() if n.warning]
    if unreadable:
        vault.warnings.append(
            f"{len(unreadable)} file(s) indexed but unreadable — "
            + ", ".join(n.rel for n in unreadable[:3])
            + ("…" if len(unreadable) > 3 else "")
        )
    if src.paths and not vault.nodes:
        vault.warnings.append(
            "Folders were readable but held no markdown, text or PDF files."
        )
    return vault


def main() -> None:
    vault = build()
    print(f"mode: {vault.mode}")
    for root in vault.roots:
        print(f"root: {root}")
    if not vault.roots:
        print("root: (none)")
    print(f"\n{len(vault.nodes)} files, {len(vault.edges)} links"
          f"{f', {vault.skipped} skipped (>2MB or unreadable)' if vault.skipped else ''}")

    print("\ncounts by type")
    for kind, count in vault.counts_by_type():
        print(f"  {kind:<10} {count:>4}")

    print("\ntop 10 hubs")
    for node in vault.hubs(10):
        print(f"  {node.degree:>3}  {node.title}  ({node.type})")

    if vault.unresolved:
        print("\nunresolved links")
        for target, count in vault.unresolved.most_common(5):
            print(f"  {count:>3}  [[{target}]]")

    if vault.warnings:
        print("\nwarnings")
        for warning in vault.warnings:
            print(f"  ! {warning}")


if __name__ == "__main__":
    main()

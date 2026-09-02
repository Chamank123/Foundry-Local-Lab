"""main.py — HTTP server + API.

Standard library only. Binds to 127.0.0.1 and nothing else: this process can
read your private notes, so it must never be reachable from the network.

    python3 agent/main.py           http://127.0.0.1:8720
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __package__ in (None, ""):  # allow `python3 agent/main.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent import data, vault  # type: ignore
else:
    from . import data, vault

UI_DIR = data.JARVIS_ROOT / "ui"
PORT = int(os.environ.get("JARVIS_PORT", "8720"))
HOST = "127.0.0.1"

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json; charset=utf-8",
}


class Index:
    """The vault, built once and reused. Rebuild is explicit."""

    def __init__(self) -> None:
        self.vault: vault.Vault | None = None
        self.built_at = 0.0
        self.error = ""

    def get(self) -> vault.Vault:
        if self.vault is None:
            self.rebuild()
        assert self.vault is not None
        return self.vault

    def rebuild(self) -> None:
        start = time.time()
        try:
            self.vault = vault.build()
            self.error = ""
        except Exception as exc:  # degrade loudly, never silently
            self.vault = vault.Vault(warnings=[f"Indexing failed: {exc}"])
            self.error = str(exc)
        self.built_at = time.time()
        v = self.vault
        print(
            f"indexed {len(v.nodes)} files, {len(v.edges)} links "
            f"({v.mode} mode, {time.time() - start:.2f}s)"
        )
        for warning in v.warnings:
            print(f"  ! {warning}")


INDEX = Index()


class Handler(BaseHTTPRequestHandler):
    server_version = "jarvis"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # quieter than the default
        if os.environ.get("JARVIS_VERBOSE"):
            super().log_message(fmt, *args)

    # -- plumbing ----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _static(self, name: str) -> None:
        """Serve from ui/ only. Resolved and re-checked so '..' cannot escape."""
        target = (UI_DIR / name).resolve()
        if not str(target).startswith(str(UI_DIR.resolve())) or not target.is_file():
            self._json({"error": "not found"}, 404)
            return
        self._send(200, target.read_bytes(),
                   MIME.get(target.suffix.lower(), "application/octet-stream"))

    # -- routes ------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)

        if route == "/":
            self._static("index.html")
        elif route.startswith("/ui/"):
            self._static(route[4:])
        elif route == "/api/graph":
            self._json(INDEX.get().to_graph())
        elif route == "/api/status":
            self._json(self._status())
        elif route == "/api/node":
            self._node(query.get("id", [""])[0])
        elif route == "/api/search":
            self._search(query.get("q", [""])[0])
        elif route == "/api/reindex":
            INDEX.rebuild()
            self._json(self._status())
        else:
            self._json({"error": "not found", "path": route}, 404)

    do_HEAD = do_GET

    def _status(self) -> dict:
        v = INDEX.get()
        return {
            **data.describe(),
            "files": len(v.nodes),
            "links": len(v.edges),
            "skipped": v.skipped,
            "warnings": v.warnings,
            "built_at": INDEX.built_at,
            "port": PORT,
        }

    def _node(self, node_id: str) -> None:
        v = INDEX.get()
        node = v.get(node_id)
        if node is None:
            self._json({"error": "no such node", "id": node_id}, 404)
            return
        neighbours = sorted(
            (v.nodes[n] for n in v.adj[node_id]),
            key=lambda n: (-n.degree, n.title),
        )
        self._json({
            **node.card(),
            "text": node.text[:4000],
            "truncated": len(node.text) > 4000,
            "neighbours": [
                {"id": n.id, "title": n.title, "type": n.type, "degree": n.degree}
                for n in neighbours
            ],
        })

    def _search(self, query: str) -> None:
        """Keyword search over the index. Not the model talking — the UI says so."""
        v = INDEX.get()
        hits = v.search(query, limit=8)
        self._json({
            "query": query,
            "engine": "keyword",
            "hits": [
                {"id": n.id, "title": n.title, "type": n.type, "degree": n.degree,
                 "rel": n.rel, "excerpt": n.excerpt(200), "score": round(s, 2)}
                for n, s in hits
            ],
        })


def main() -> None:
    INDEX.rebuild()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    v = INDEX.get()
    print(f"\n  JARVIS  http://{HOST}:{PORT}")
    print(f"  mode    {v.mode}" + ("  (demo fixtures — safe to record)"
                                   if v.mode == "demo" else "  (your real folders)"))
    print("  stop    ctrl-c\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

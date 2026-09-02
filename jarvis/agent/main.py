"""main.py — HTTP server + API.

Standard library only. Binds to 127.0.0.1 and nothing else: this process can
read your private notes, so it must never be reachable from the network.

    python3 agent/main.py           http://127.0.0.1:8720
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __package__ in (None, ""):  # allow `python3 agent/main.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent import data, memory, tools, vault  # type: ignore
else:
    from . import data, memory, tools, vault

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


PROMPT_FILE = Path(__file__).resolve().parent / "prompt.md"

# Foundry Local assigns a dynamic port, so the endpoint is discovered rather
# than assumed. 5273 is the documented default and worth trying first.
MODEL_CANDIDATES = ["http://localhost:5273/v1", "http://127.0.0.1:5273/v1"]
HTTP_TIMEOUT = float(os.environ.get("JARVIS_MODEL_TIMEOUT", "30"))


class Model:
    """A local OpenAI-compatible endpoint — Foundry Local by default.

    Never required. When it is missing everything still works, and the UI is
    told so plainly: keyword routing must never be passed off as the model
    talking.
    """

    def __init__(self) -> None:
        self.base = ""
        self.name = ""
        self.reason = "not probed yet"
        self.checked_at = 0.0

    # -- discovery ---------------------------------------------------------
    def discover(self) -> None:
        self.checked_at = time.time()
        candidates = []
        configured = os.environ.get("JARVIS_MODEL_URL", "").strip()
        if configured:
            candidates.append(configured.rstrip("/"))
        candidates += MODEL_CANDIDATES
        found = self._from_cli()
        if found:
            candidates.append(found)

        tried = []
        for base in candidates:
            names = self._models(base)
            if names:
                self.base = base
                wanted = os.environ.get("JARVIS_MODEL", "").strip()
                self.name = wanted if wanted in names else names[0]
                self.reason = ""
                return
            tried.append(base)
        self.base = self.name = ""
        self.reason = ("No local model endpoint answered. Tried: "
                       + ", ".join(tried)
                       + ". Start Foundry Local, or set JARVIS_MODEL_URL in .env.")

    def _from_cli(self) -> str:
        """`foundry service status` prints the endpoint when the port is dynamic."""
        try:
            out = subprocess.run(["foundry", "service", "status"],
                                 capture_output=True, text=True, timeout=4).stdout
        except (OSError, subprocess.SubprocessError):
            return ""
        hit = re.search(r"https?://[\w.\-]+:\d+(?:/v1)?", out or "")
        if not hit:
            return ""
        url = hit.group(0).rstrip("/")
        return url if url.endswith("/v1") else url + "/v1"

    def _models(self, base: str) -> list[str]:
        try:
            with urllib.request.urlopen(base + "/models", timeout=3) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return []
        return [m.get("id", "") for m in payload.get("data", []) if m.get("id")]

    @property
    def available(self) -> bool:
        return bool(self.base and self.name)

    def status(self) -> dict:
        return {"available": self.available, "endpoint": self.base,
                "model": self.name, "reason": self.reason}

    # -- inference ---------------------------------------------------------
    def chat(self, messages: list[dict], max_tokens: int = 220,
             temperature: float = 0.4) -> str:
        if not self.available:
            raise RuntimeError(self.reason or "no model")
        body = json.dumps({
            "model": self.name, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return (payload["choices"][0]["message"]["content"] or "").strip()


MODEL = Model()


# ----------------------------------------------------------- conversation ---
GREETING = re.compile(
    r"^\s*(hi|hey|hello|yo|morning|afternoon|evening|good (morning|afternoon|evening)"
    r"|how('?s| is) it going|how are you|you (there|awake|up)|can you hear me"
    r"|thanks|thank you|cheers|ok|okay|cool|nice|right)\b", re.I)

INTENT = [
    ("brief_me", re.compile(r"\bbrief\b|what('?s| is) (due|on|next|happening)"
                            r"|what did i miss|catch me up|where am i", re.I)),
    ("plan_day", re.compile(r"\bplan\b|what should i do|what do i do (today|now)"
                            r"|priorit|where do i start", re.I)),
    ("read_inbox", re.compile(r"\b(inbox|e-?mails?|messages?|mail)\b|who (wrote|emailed)", re.I)),
    ("remember", re.compile(r"^\s*(remember|note that|don'?t forget|keep in mind)\b", re.I)),
    ("research_web", re.compile(r"\b(look up|search the web|google it|on the web"
                                r"|latest news|current price)\b", re.I)),
]

ROUTE_PROMPT = """You route one user message. Reply with JSON only, no prose.

{"tool": null} for conversation - greetings, opinions, follow-ups, anything you
can answer by talking.
{"tool": "<name>", "args": {...}} when a tool is genuinely needed.

Tools:
""" + "\n".join(f"- {t['name']}({', '.join(t['args']) or ''}) - {t['when']}"
                 for t in tools.SPEC)


class Conversation:
    """The last ten turns, so follow-ups resolve without being restated."""

    def __init__(self, limit: int = 10) -> None:
        self.turns: deque = deque(maxlen=limit)

    def history(self) -> list[dict]:
        out = []
        for user, reply in self.turns:
            out.append({"role": "user", "content": user})
            out.append({"role": "assistant", "content": reply})
        return out

    # -- routing -----------------------------------------------------------
    def route(self, text: str) -> tuple[str | None, dict, str]:
        """Returns (tool name or None, args, how the decision was made)."""
        if MODEL.available:
            try:
                raw = MODEL.chat(
                    [{"role": "system", "content": ROUTE_PROMPT}]
                    + self.history()[-6:]
                    + [{"role": "user", "content": text}],
                    max_tokens=90, temperature=0.0)
                hit = re.search(r"\{.*\}", raw, re.S)
                if hit:
                    choice = json.loads(hit.group(0))
                    name = choice.get("tool")
                    if name in tools.TOOLS:
                        return name, choice.get("args") or {}, "model"
                    if name is None:
                        return None, {}, "model"
            except Exception:
                pass  # fall through to scoring; never fail the turn on the model

        return (*self._score(text), "keyword")

    def _score(self, text: str) -> tuple[str | None, dict]:
        """No model: decide by scoring the question against the files.

        This is keyword matching and the UI says so. It is never described to
        the user as the model talking.
        """
        if GREETING.match(text) and len(text.split()) <= 6:
            return None, {}
        for name, pattern in INTENT:
            if pattern.search(text):
                if name == "remember":
                    fact = re.sub(r"^\s*(remember|note) that\b|^\s*(remember|don'?t forget"
                                  r"|keep in mind)\b[:,]?", "", text, flags=re.I).strip()
                    return name, {"fact": fact or text}
                if name == "research_web":
                    return name, {"query": text}
                return name, {}
        vault_obj = INDEX.get()
        if vault_obj.relevance(text) >= 0.35:
            return "search_brain", {"query": text}
        return None, {}

    # -- a turn ------------------------------------------------------------
    def ask(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {"spoken": "Nothing to answer.", "card": {}, "tool": None,
                    "engine": "none", "model": MODEL.status()}

        v = INDEX.get()
        name, args, how = self.route(text)

        if name:
            fn = tools.TOOLS[name]
            try:
                result = fn(v, **args) if args else fn(v)
            except TypeError:
                result = fn(v, text) if name in ("search_brain", "research_web") else fn(v)
            payload = result.as_json()
            # The spoken line for a tool answer is composed in Python, not by
            # the model. A model that rephrases "marked 2:1 - supervision marks
            # carry no weight" can drop the second half, and that is the one
            # failure this build treats as worse than silence.
            payload.update({"engine": "tool", "routed_by": how,
                            "model": MODEL.status()})
            self.turns.append((text, result.spoken))
            return payload

        reply, engine = self._talk(text, v)
        self.turns.append((text, reply))
        return {"spoken": reply, "card": {}, "tool": None, "engine": engine,
                "routed_by": how, "model": MODEL.status(), "warnings": []}

    def _talk(self, text: str, v) -> tuple[str, str]:
        if MODEL.available:
            try:
                system = PROMPT_FILE.read_text(encoding="utf-8")
                reply = MODEL.chat(
                    [{"role": "system", "content": system}]
                    + self.history()
                    + [{"role": "user", "content": text}],
                    max_tokens=180, temperature=0.5)
                if reply:
                    return reply, "model"
            except Exception as exc:
                return (f"The model stopped answering ({type(exc).__name__}). "
                        f"Your files still work."), "degraded"
        return self._no_model_reply(text), "fallback"

    def _no_model_reply(self, text: str) -> str:
        """Honest, short, and never dressed up as the model.

        Conversation without a language model is not something to fake. Say
        what is missing and what still works.
        """
        if GREETING.match(text):
            return "Here. No model loaded, so I'm limited to your files."
        return ("No model is loaded, so I can't talk this through — only search "
                "your files. Start Foundry Local, or ask me something a file "
                "would answer.")


CONVERSATION = Conversation()


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
        elif route == "/api/memory":
            self._json({"remembered": memory.recall(30)})
        elif route == "/api/model":
            MODEL.discover()
            self._json(MODEL.status())
        else:
            self._json({"error": "not found", "path": route}, 404)

    do_HEAD = do_GET

    def do_POST(self) -> None:  # noqa: N802
        route = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length > 64_000:
            self._json({"error": "payload too large"}, 413)
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._json({"error": "bad json"}, 400)
            return

        if route == "/api/ask":
            try:
                self._json(CONVERSATION.ask(str(body.get("text", ""))))
            except Exception as exc:                 # degrade loudly
                self._json({"spoken": f"That failed: {type(exc).__name__}.",
                            "card": {"error": str(exc)}, "tool": None,
                            "engine": "error", "model": MODEL.status()}, 200)
        elif route == "/api/remember":
            fact = str(body.get("fact", ""))
            self._json(tools.remember(INDEX.get(), fact).as_json())
        else:
            self._json({"error": "not found", "path": route}, 404)

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
            "model": MODEL.status(),
            "web": os.environ.get("JARVIS_WEB", "0") in ("1", "true", "yes"),
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
    MODEL.discover()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    v = INDEX.get()
    print(f"\n  JARVIS  http://{HOST}:{PORT}")
    print(f"  mode    {v.mode}" + ("  (demo fixtures — safe to record)"
                                   if v.mode == "demo" else "  (your real folders)"))
    print(f"  model   {MODEL.name or 'none'}"
          + (f"  ({MODEL.base})" if MODEL.available
             else "  — routing by keyword, and the UI says so"))
    print("  stop    ctrl-c\n")
    if not MODEL.available:
        print(f"  ! {MODEL.reason}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

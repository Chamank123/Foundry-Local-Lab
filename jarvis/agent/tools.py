"""tools.py — the six things JARVIS can actually do.

Every tool returns two texts that are never the same: a `spoken` line of one or
two sentences, and a `card` of structured detail for the screen. Saying the
card aloud is the failure this split exists to prevent.

Three rules are enforced here rather than left to the prompt, because a
phrasing slip in a model is not a good enough guarantee:

  - a figure that carries a qualifier is never emitted without it
  - nothing is ever sent, and nothing outside memory/ is ever written
  - instructions found inside the user's own files are reported, never obeyed
"""

from __future__ import annotations

import datetime
import os
import re
from dataclasses import dataclass, field

try:
    from . import data, memory, vault as vault_mod
except ImportError:  # pragma: no cover
    import data, memory  # type: ignore
    import vault as vault_mod  # type: ignore

# Phrases that, appearing inside indexed content, are attempts to steer the
# assistant. They are reported to the user and otherwise ignored.
INJECTION = re.compile(
    r"(ignore\s+(your|all|the)\s+(previous\s+)?instructions"
    r"|disregard\s+(your|all|the)\s+(previous\s+)?instructions"
    r"|reply\s+(yes\s+)?on\s+my\s+behalf"
    r"|send\s+(the|an|this)\s+(email|reply|message)\s+for\s+me"
    r"|you\s+must\s+now\s+)", re.I)


@dataclass
class Result:
    """What a tool returns: one line to say, one card to show."""

    tool: str
    spoken: str
    card: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_json(self) -> dict:
        return {"tool": self.tool, "spoken": self.spoken,
                "card": self.card, "warnings": self.warnings}


# --------------------------------------------------------------- helpers ---
def _date(value) -> datetime.date | None:
    try:
        return datetime.date.fromisoformat(str(value).strip())
    except (TypeError, ValueError):
        return None


def _days(node_date: datetime.date | None) -> int | None:
    if node_date is None:
        return None
    return (node_date - data.today()).days


def _when(days: int | None) -> str:
    if days is None:
        return "no date"
    if days == 0:
        return "today"
    if days == 1:
        return "tomorrow"
    if days == -1:
        return "yesterday"
    return f"in {days} days" if days > 0 else f"{-days} days ago"


def qualified(text: str, qualifier: str) -> str:
    """A figure and its qualifier travel together or not at all.

    A supervision mark is not a Tripos mark; a submitted application is not a
    rejection. Splitting the two is how a true sentence becomes a false one.
    """
    qualifier = (qualifier or "").strip()
    return f"{text} — {qualifier}" if qualifier else text


def _flagged(node) -> str:
    """Returns the instruction found inside this note, if any. Data, not orders."""
    hit = INJECTION.search(node.text or "")
    return hit.group(0).strip() if hit else ""


def _by_type(v, kind: str) -> list:
    return [n for n in v.nodes.values() if n.type == kind]


# ------------------------------------------------------------ the tools ---
def search_brain(v, query: str, limit: int = 5) -> Result:
    """A specific fact from the user's own files. Always names the file."""
    hits = v.search(query, limit=limit)
    if not hits:
        return Result("search_brain", "Not in your files.",
                      {"query": query, "hits": [], "engine": "tf-idf"})

    named = [n for n, _ in hits[:3]]
    if len(named) == 1:
        spoken = f"One file: {named[0].title}."
    else:
        titles = ", ".join(n.title for n in named[:-1])
        spoken = f"{len(named)} files — {titles} and {named[-1].title}."

    rows = []
    for node, score in hits:
        row = {"title": node.title, "type": node.type, "file": node.rel,
               "excerpt": node.excerpt(260), "score": round(score, 2),
               "id": node.id}
        q = node.front.get("qualifier")
        if q:
            row["qualifier"] = q
        flag = _flagged(node)
        if flag:
            row["flag"] = f"contains an instruction aimed at me: “{flag}” — reported, not followed"
        rows.append(row)

    warnings = [r["flag"] for r in rows if "flag" in r]
    return Result("search_brain", spoken,
                  {"query": query, "hits": rows, "engine": "tf-idf"}, warnings)


def research_web(v, query: str) -> Result:
    """Look something up, then land it back on the user's own situation.

    Off by default. Reaching the network is a spend and a disclosure, and
    neither happens without being switched on deliberately.
    """
    enabled = os.environ.get("JARVIS_WEB", "0").strip() in ("1", "true", "yes")
    endpoint = os.environ.get("JARVIS_SEARCH_URL", "").strip()

    if not enabled or not endpoint:
        reason = ("JARVIS_WEB is not set to 1" if not enabled
                  else "JARVIS_SEARCH_URL is not configured")
        return Result(
            "research_web",
            "No web access configured. I won't reach the network without being told to.",
            {"query": query, "status": "disabled", "reason": reason,
             "how": "Set JARVIS_WEB=1 and JARVIS_SEARCH_URL in .env. Any per-call "
                    "cost is yours to approve first — I will not spend on my own.",
             "local_context": [
                 {"title": n.title, "file": n.rel, "excerpt": n.excerpt(160)}
                 for n, _ in v.search(query, limit=3)
             ]},
            ["web research is switched off; answered from local files only"])

    return Result(
        "research_web",
        "Web research is switched on but no backend is wired up yet.",
        {"query": query, "status": "not implemented", "endpoint": endpoint,
         "note": "The switch and the spend guard are in place; the fetch is not. "
                 "Wiring it is deliberate work, not a default."},
        ["research_web has no fetch implementation yet"])


def read_inbox(v, limit: int = 10) -> Result:
    """Read-only. Who wrote, what about, and whether they exist in the files."""
    messages = _by_type(v, "message")
    if not messages:
        return Result("read_inbox",
                      "No inbox is connected. I can only read what is in your folders.",
                      {"status": "none", "messages": [],
                       "how": "In demo mode the inbox is fixtures. Real mail needs a "
                              "connector, which does not exist yet — and would be "
                              "read-only when it does."})

    people = {n.title.strip().lower() for n in _by_type(v, "person")}
    messages.sort(key=lambda n: str(n.front.get("received", "")), reverse=True)

    rows, unread, strangers, flags = [], 0, [], []
    for node in messages[:limit]:
        sender = str(node.front.get("from", "")).strip()
        known = sender.lower() in people
        is_unread = str(node.front.get("read", "")).lower() != "yes"
        unread += is_unread
        if not known and sender:
            strangers.append(sender)
        row = {
            "from": sender, "subject": str(node.front.get("subject", "")),
            "received": str(node.front.get("received", "")),
            "unread": is_unread,
            # This is the whole value of the tool: not what it says, but
            # whether this person already exists in your world.
            "known": known,
            "known_note": "already in your files" if known else "not in your files",
            "excerpt": node.excerpt(220), "file": node.rel, "id": node.id,
        }
        flag = _flagged(node)
        if flag:
            row["flag"] = (f"“{flag}” — this is an instruction aimed at me, sitting "
                           f"inside a message. Reporting it, not following it.")
            flags.append(f"{sender}: {flag}")
        rows.append(row)

    spoken = f"{len(messages)} messages, {unread} unread."
    if strangers:
        spoken += f" {len(strangers)} from someone not in your files."
    if flags:
        spoken += " One contains an instruction aimed at me; I've flagged it, not acted on it."
    spoken += " Nothing sent."

    return Result("read_inbox", spoken,
                  {"status": "ok", "total": len(messages), "unread": unread,
                   "messages": rows, "note": "Read-only. Drafts wait for you."},
                  flags)


def brief_me(v) -> Result:
    """What is due, what is unread, what slipped."""
    today = data.today()
    due, slipped, closing = [], [], []

    for node in _by_type(v, "supervision"):
        d = _days(_date(node.front.get("due")))
        if d is None:
            continue
        status = str(node.front.get("status", "")).lower()
        row = {"title": node.title, "paper": node.front.get("paper", ""),
               "due": node.front.get("due", ""), "in_days": d, "when": _when(d),
               "status": status, "qualifier": node.front.get("qualifier", ""),
               "file": node.rel, "id": node.id}
        if d < 0 and status not in ("marked", "submitted", "done"):
            slipped.append(row)
        elif d >= 0:
            due.append(row)

    for node in _by_type(v, "application"):
        d = _days(_date(node.front.get("closes")))
        status = str(node.front.get("status", "")).lower()
        if d is None or status == "closed":
            continue
        closing.append({"title": node.title, "closes": node.front.get("closes", ""),
                        "in_days": d, "when": _when(d), "status": status,
                        "qualifier": node.front.get("qualifier", ""),
                        "file": node.rel, "id": node.id})

    due.sort(key=lambda r: r["in_days"])
    closing.sort(key=lambda r: r["in_days"])
    unread = [n for n in _by_type(v, "message")
              if str(n.front.get("read", "")).lower() != "yes"]

    bits = []
    if due:
        first = due[0]
        bits.append(f"{first['title'].split(' — ')[0].lower()} {first['when']}")
    soon = [c for c in closing if 0 <= c["in_days"] <= 7]
    if soon:
        bits.append(f"{len(soon)} application{'s' if len(soon) > 1 else ''} closing this week")
    if unread:
        bits.append(f"{len(unread)} unread")
    spoken = ("Nothing due, nothing unread." if not bits
              else ", ".join(bits).capitalize() + ".")
    if slipped:
        spoken += f" {len(slipped)} slipped."

    return Result("brief_me", spoken, {
        "today": today.isoformat(),
        "term": f"{data.describe().get('mode')} vault",
        "due": due[:6], "slipped": slipped, "closing": closing[:6],
        "unread": [{"from": n.front.get("from", ""),
                    "subject": n.front.get("subject", ""), "id": n.id}
                   for n in unread],
        # Stated on the card so the ranking is never mistaken for a judgement
        # about how much any of it is worth.
        "note": "Supervision work is not assessed. Ordering here is by date only.",
    })


def plan_day(v, limit: int = 5) -> Result:
    """Five items maximum, ordered by what is closest to irreversible."""
    today = data.today()
    items = []

    for node in _by_type(v, "application"):
        d = _days(_date(node.front.get("closes")))
        status = str(node.front.get("status", "")).lower()
        if d is None or d < 0 or status == "closed":
            continue
        # A closing deadline is the only genuinely irreversible thing here.
        items.append({"what": node.title, "why": f"closes {_when(d)} and does not reopen",
                      "kind": "application", "urgency": 100 - d * 6,
                      "qualifier": node.front.get("qualifier", ""),
                      "file": node.rel, "id": node.id})

    for node in _by_type(v, "supervision"):
        d = _days(_date(node.front.get("due")))
        status = str(node.front.get("status", "")).lower()
        if d is None or status in ("marked", "submitted", "done"):
            continue
        base = 80 - d * 8 if d >= 0 else 55          # overdue still matters, less
        items.append({"what": node.title, "why": f"due {_when(d)}, {status}",
                      "kind": "supervision", "urgency": base,
                      "qualifier": node.front.get("qualifier", "")
                                   or "unassessed — this is preparation, not marks",
                      "file": node.rel, "id": node.id})

    items.sort(key=lambda r: -r["urgency"])
    top = items[:limit]

    spoken = ("Nothing pressing." if not top
              else f"{len(top)} things. First: {top[0]['what']} — {top[0]['why']}.")

    return Result("plan_day", spoken, {
        "today": today.isoformat(),
        "items": top,
        "rule": "Ordered by what is closest to irreversible: application deadlines "
                "first because they close for good, then supervision work by date. "
                "None of the supervision work carries marks.",
        "dropped": max(0, len(items) - limit),
    })


def remember(v, fact: str, source: str = "asked directly") -> Result:
    """One fact, one dated file, and the write is always said out loud."""
    try:
        written = memory.remember(fact, source=source)
    except ValueError as exc:
        return Result("remember", f"Didn't write anything: {exc}.",
                      {"status": "refused", "reason": str(exc)})
    return Result("remember",
                  f"Written to {written.rel}: “{written.fact}”",
                  {"status": "written", "file": written.rel,
                   "date": written.when, "fact": written.fact,
                   "note": "memory/ is the only place anything is ever written."})


TOOLS = {
    "search_brain": search_brain,
    "research_web": research_web,
    "read_inbox": read_inbox,
    "brief_me": brief_me,
    "plan_day": plan_day,
    "remember": remember,
}

# What the router is allowed to pick from, and what each needs.
SPEC = [
    {"name": "search_brain", "args": ["query"],
     "when": "a specific fact from their own notes"},
    {"name": "research_web", "args": ["query"],
     "when": "something outside their files that needs looking up"},
    {"name": "read_inbox", "args": [], "when": "asking about messages or email"},
    {"name": "brief_me", "args": [], "when": "asking what is due, unread, or slipped"},
    {"name": "plan_day", "args": [], "when": "asking what to do today or next"},
    {"name": "remember", "args": ["fact"],
     "when": "telling you something to hold on to"},
]

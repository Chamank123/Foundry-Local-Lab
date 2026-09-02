#!/usr/bin/env python3
"""generate.py — build the demo vault.

Fixed seed, fixed dates: the graph is byte-identical every run, so a demo you
record today looks the same tomorrow. Nothing here reads your real folders.

    python3 data/generate.py            rebuild data/demo/

PERSONA below is the one block to change. Swap it for the real shape of the
business and re-run; every fixture, link and number re-seeds from it.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

SEED = 20260902
TODAY = (2026, 9, 2)
OUT = Path(__file__).resolve().parent / "demo"

# ---------------------------------------------------------------- persona ---
# Placeholder shape, not anybody's real business. Replace and re-run.
PERSONA = {
    "studio": "Two-person studio: internal automation and web builds for small UK service businesses.",
    "currency": "£",
    "build_range": (4000, 9000),
    "retainer": 800,
    "day_rate": 550,
}

CLIENTS = [
    ("Ashgrove Dental", "dental practice, two sites", "retainer"),
    ("Pennine Logistics", "regional courier, 40 vans", "build"),
    ("Rowan & Vale", "boutique law firm", "build"),
    ("Bellwether Fitness", "three gyms, one owner", "retainer"),
    ("Tidewater Property", "lettings agency", "build"),
    ("Marlow Accounting", "practice, 6 staff", "retainer"),
    ("Kingfisher Legal", "conveyancing only", "build"),
    ("Saltbox Coffee", "four cafes, one roastery", "build"),
    ("Verity Physio", "single-site clinic", "retainer"),
]

PEOPLE = [
    ("Alan Whitcombe", "practice manager", "Ashgrove Dental"),
    ("Ines Duarte", "ops director", "Pennine Logistics"),
    ("Tomas Berg", "partner", "Rowan & Vale"),
    ("Hannah Okafor", "owner", "Bellwether Fitness"),
    ("Ruth Calloway", "lettings manager", "Tidewater Property"),
    ("Dev Anand", "managing partner", "Marlow Accounting"),
    ("Mira Sandoval", "office manager", "Kingfisher Legal"),
    ("Owen Pryce", "founder", "Saltbox Coffee"),
    ("Lucy Brandt", "clinic lead", "Verity Physio"),
    ("Sam Ferreira", "contract developer", None),
]

PROJECTS = [
    ("Ashgrove — recall automation", "Ashgrove Dental", "live", 6200),
    ("Pennine — dispatch dashboard", "Pennine Logistics", "build", 8800),
    ("Rowan & Vale — intake forms", "Rowan & Vale", "build", 5400),
    ("Bellwether — membership sync", "Bellwether Fitness", "live", 4600),
    ("Tidewater — viewings portal", "Tidewater Property", "build", 7900),
    ("Marlow — onboarding pipeline", "Marlow Accounting", "live", 5100),
    ("Kingfisher — matter tracker", "Kingfisher Legal", "scoping", 6700),
    ("Saltbox — stock reorder bot", "Saltbox Coffee", "build", 4300),
    ("Verity — booking rebuild", "Verity Physio", "live", 4900),
    ("Ashgrove — second site rollout", "Ashgrove Dental", "scoping", 3800),
    ("Pennine — driver app pilot", "Pennine Logistics", "paused", 9000),
]

CONCEPTS = [
    "Margin model", "Handover pack", "Reusable components", "Discovery call",
    "Change orders", "Scope creep", "Retainer ladder", "Fixed-price risk",
    "Component library", "Client onboarding", "Deposit terms", "Payment triggers",
    "Async updates", "Build velocity", "Support SLA", "Referral loop",
]

SOPS = [
    "SOP — Automation build", "SOP — Proposal", "SOP — Handover",
    "SOP — Invoice run", "SOP — Discovery call", "SOP — Incident",
]

# (client, amount, status, qualifier) — the qualifier is the whole point:
# a part-paid invoice is not a discount, and JARVIS must never say it is.
INVOICES = [
    ("Ashgrove Dental", 6200, "paid", "full, settled 14 days"),
    ("Pennine Logistics", 4400, "part-paid", "stage 1 of 2 — build still running"),
    ("Rowan & Vale", 2700, "part-paid", "50% deposit — work not yet started"),
    ("Bellwether Fitness", 800, "paid", "monthly retainer"),
    ("Tidewater Property", 7900, "unpaid", "issued 3 days ago, 30-day terms"),
    ("Marlow Accounting", 800, "paid", "monthly retainer"),
    ("Kingfisher Legal", 1200, "unpaid", "discovery phase only"),
    ("Saltbox Coffee", 4300, "paid", "full, settled on delivery"),
    ("Verity Physio", 600, "part-paid", "agreed reduction — two weeks paused by client"),
]

PROPOSALS = [
    ("Ashgrove Dental", 3800), ("Pennine Logistics", 9000), ("Rowan & Vale", 5400),
    ("Tidewater Property", 7900), ("Kingfisher Legal", 6700), ("Saltbox Coffee", 4300),
    ("Verity Physio", 4900), ("Bellwether Fitness", 5200), ("Marlow Accounting", 4100),
]

NOTE_TOPICS = [
    "Where the last four builds actually lost time",
    "Why fixed price keeps beating day rate here",
    "Retainers are the only predictable line",
    "The handover pack pays for itself",
    "Discovery calls that go nowhere have a tell",
    "Component reuse across the last three builds",
    "What clients ask for in week six",
    "Pricing the second site lower was a mistake",
    "Chasing invoices earlier changed nothing",
    "The two clients who never send change orders",
    "Subcontracting maths",
    "Support load per live project",
    "What to stop quoting for",
    "Deposit terms after the Rowan delay",
    "Async updates cut meeting hours",
    "Referrals came from two clients only",
    "Rebuilding versus patching",
    "Which projects are actually finished",
    "Capacity for the rest of the quarter",
]

CALL_TOPICS = [
    "kickoff", "scope check", "weekly update", "change request", "handover walkthrough",
    "invoice question", "renewal", "bug triage", "training session", "planning",
]

CAMPAIGN = "Q4 — practice management push"
BRIEFS = [
    ("Brief — practice management push", CAMPAIGN),
    ("Brief — retainer upsell", CAMPAIGN),
]

TYPE_DIRS = {
    "client": "clients", "person": "people", "project": "projects",
    "concept": "concepts", "sop": "sops", "invoice": "invoices",
    "proposal": "proposals", "call": "calls", "note": "notes",
    "brief": "briefs", "campaign": "campaigns",
}


def slug(title: str) -> str:
    keep = []
    for ch in title.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in " -_&—":
            keep.append("-")
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def date_back(days: int) -> str:
    """Fixed arithmetic from TODAY — no clock reads, so runs are identical."""
    import datetime

    d = datetime.date(*TODAY) - datetime.timedelta(days=days)
    return d.isoformat()


def link(title: str) -> str:
    return f"[[{title}]]"


def write_md(kind: str, title: str, front: dict, body: str) -> None:
    d = OUT / TYPE_DIRS[kind]
    d.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"type: {kind}", f"title: {title}"]
    for k, v in front.items():
        lines.append(f"{k}: {v}")
    lines += ["---", "", f"# {title}", "", body.strip(), ""]
    (d / f"{slug(title)}.md").write_text("\n".join(lines), encoding="utf-8")


def write_pdf(kind: str, title: str, lines: list[str]) -> None:
    """A minimal, valid, uncompressed PDF — enough to prove the PDF path works
    end to end without pulling in a dependency."""
    d = OUT / TYPE_DIRS[kind]
    d.mkdir(parents=True, exist_ok=True)

    def pdf_string(s: str) -> bytes:
        """Literal string. Falls back to UTF-16BE with a BOM when the text
        will not fit in latin-1 — otherwise em-dashes become '?' and the
        wikilinks in this file stop resolving."""
        try:
            raw = s.encode("latin-1")
        except UnicodeEncodeError:
            raw = b"\xfe\xff" + s.encode("utf-16-be")
        out = bytearray(b"(")
        for byte in raw:
            if byte in (0x28, 0x29, 0x5C):  # ( ) \
                out += b"\\"
            out.append(byte)
        out += b")"
        return bytes(out)

    body = bytearray(b"BT\n/F1 11 Tf\n50 780 Td\n15 TL\n")
    for ln in lines:
        body += pdf_string(ln) + b" Tj T*\n"
    body += b"ET"
    stream = bytes(body)

    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 595 842]"
        b"/Resources<</Font<</F1 5 0 R>>>>/Contents 4 0 R>>",
        b"<</Length " + str(len(stream)).encode() + b">>\nstream\n" + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<</Size {len(objs) + 1}/Root 1 0 R>>\nstartxref\n{xref}\n".encode()
        + b"%%EOF\n"
    )
    (d / f"{slug(title)}.pdf").write_bytes(bytes(out))


def build() -> dict:
    rng = random.Random(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    cur = PERSONA["currency"]
    counts: dict[str, int] = {}

    def bump(kind: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1

    client_names = [c[0] for c in CLIENTS]
    project_names = [p[0] for p in PROJECTS]

    # -- concepts: linked to each other, so hubs emerge from real structure
    for i, name in enumerate(CONCEPTS):
        others = rng.sample([c for c in CONCEPTS if c != name], 2)
        body = (
            f"Working definition, kept short on purpose.\n\n"
            f"Sits next to {link(others[0])} and {link(others[1])}. "
            f"Comes up most often during {link('Discovery call')}."
        )
        write_md("concept", name, {"date": date_back(300 - i * 7)}, body)
        bump("concept")

    # -- sops
    for i, name in enumerate(SOPS):
        cs = rng.sample(CONCEPTS, 3)
        body = (
            "Steps, in order. Deviating from this is a decision, not an accident.\n\n"
            f"1. {link(cs[0])}\n2. {link(cs[1])}\n3. {link(cs[2])}\n\n"
            f"Owner: the studio. Reviewed {date_back(60 + i * 10)}."
        )
        write_md("sop", name, {"date": date_back(200 - i * 12)}, body)
        bump("sop")

    # -- clients
    for name, desc, model in CLIENTS:
        contacts = [p[0] for p in PEOPLE if p[2] == name]
        projs = [p[0] for p in PROJECTS if p[1] == name]
        parts = [f"{desc}. Engagement model: {model}."]
        if contacts:
            parts.append("Contact: " + ", ".join(link(c) for c in contacts) + ".")
        if projs:
            parts.append("Work: " + ", ".join(link(p) for p in projs) + ".")
        parts.append(f"Onboarded via {link('Client onboarding')}.")
        write_md("client", name, {"model": model}, "\n\n".join(parts))
        bump("client")

    # -- people
    for name, role, client in PEOPLE:
        parts = [f"{role.capitalize()}."]
        if client:
            parts.append(f"At {link(client)}.")
        else:
            parts.append(f"Freelance. Used on overflow — see {link('Build velocity')}.")
        parts.append(f"Prefers {link('Async updates')}.")
        write_md("person", name, {"role": role}, "\n\n".join(parts))
        bump("person")

    # -- projects
    for i, (name, client, status, value) in enumerate(PROJECTS):
        cs = rng.sample(CONCEPTS, 2)
        sop = rng.choice(SOPS)
        body = (
            f"For {link(client)}. Status: {status}. Contracted at {cur}{value:,}.\n\n"
            f"Built to {link(sop)}. Watch {link(cs[0])} and {link(cs[1])}."
        )
        write_md(
            "project", name,
            {"client": client, "status": status, "value": value, "date": date_back(180 - i * 12)},
            body,
        )
        bump("project")

    # -- invoices
    for i, (client, amount, status, qualifier) in enumerate(INVOICES):
        projs = [p[0] for p in PROJECTS if p[1] == client]
        proj = projs[0] if projs else None
        body = (
            f"{cur}{amount:,} — {status}.\n\n"
            f"**Qualifier: {qualifier}.** This is why the number looks the way it "
            f"does. Quoting the amount without this line is misleading.\n\n"
            f"Client: {link(client)}."
            + (f" Project: {link(proj)}." if proj else "")
            + f" Raised under {link('SOP — Invoice run')}, terms per {link('Deposit terms')}."
            + "".join(f" Sent to {link(p[0])}." for p in PEOPLE if p[2] == client)
        )
        write_md(
            "invoice", f"Invoice {2026100 + i} — {client}",
            {"client": client, "amount": amount, "status": status,
             "qualifier": qualifier, "date": date_back(90 - i * 8)},
            body,
        )
        bump("invoice")

    # -- proposals (one as a PDF, to exercise the PDF path)
    for i, (client, amount) in enumerate(PROPOSALS):
        cs = rng.sample(CONCEPTS, 2)
        title = f"Proposal — {client}"
        if i == 0:
            write_pdf("proposal", title, [
                title, "",
                f"Prepared for {client}.",
                f"Fixed price: {cur}{amount:,}. Payment on [[Payment triggers]].",
                "Scope follows [[SOP — Proposal]].",
                f"Assumes no change beyond [[Change orders]].",
                f"Client record: [[{client}]].",
            ])
        else:
            body = (
                f"Fixed price {cur}{amount:,}, written to {link('SOP — Proposal')}.\n\n"
                f"For {link(client)}. Priced off {link('Margin model')}; "
                f"anything past scope goes through {link(cs[0])} and {link(cs[1])}."
            )
            write_md("proposal", title,
                     {"client": client, "amount": amount, "date": date_back(120 - i * 9)}, body)
        bump("proposal")

    # -- campaign and briefs (one brief as a PDF)
    write_md("campaign", CAMPAIGN, {"date": date_back(45)},
             "Targeting practices already running two sites.\n\n"
             + "Anchors: " + ", ".join(link(c) for c in ["Ashgrove Dental", "Marlow Accounting"])
             + f". Measured against {link('Referral loop')}.")
    bump("campaign")
    for i, (title, campaign) in enumerate(BRIEFS):
        if i == 0:
            write_pdf("brief", title, [
                title, "",
                f"Part of [[{campaign}]].",
                "Audience: practice managers at two-site clinics.",
                "Proof: [[Ashgrove — recall automation]] cut no-shows.",
                "Offer ladder per [[Retainer ladder]].",
            ])
        else:
            write_md("brief", title, {"campaign": campaign, "date": date_back(30)},
                     f"Part of {link(campaign)}.\n\n"
                     f"Push existing builds up the {link('Retainer ladder')}. "
                     f"Leads with {link('Support SLA')}.")
        bump("brief")

    # -- calls: the bulk of the graph. Weighted so the busiest clients are
    # visibly the busiest, the way they are in a real week.
    weights = [8, 7, 6, 4, 4, 3, 3, 2, 2, 2]
    topic_sop = {
        "kickoff": "SOP — Discovery call", "handover walkthrough": "SOP — Handover",
        "invoice question": "SOP — Invoice run", "bug triage": "SOP — Incident",
        "change request": "SOP — Automation build", "renewal": "SOP — Proposal",
    }
    for i in range(38):
        person, role, client = rng.choices(PEOPLE, weights=weights, k=1)[0]
        topic = CALL_TOPICS[i % len(CALL_TOPICS)]
        home = client or rng.choice(client_names)
        projs = [p[0] for p in PROJECTS if p[1] == home] or project_names
        proj = rng.choice(projs)
        cs = rng.sample(CONCEPTS, 1)
        sop = topic_sop.get(topic)
        title = f"Call — {person}, {topic} ({date_back(2 + i * 3)})"
        body = (
            f"{topic.capitalize()} with {link(person)} at {link(home)}.\n\n"
            f"On {link(proj)}. Raised {link(cs[0])}."
            + (f" Ran it to {link(sop)}." if sop else "")
            + "\n\nNext step agreed, nothing sent."
        )
        write_md("call", title,
                 {"person": person, "client": home, "date": date_back(2 + i * 3)}, body)
        bump("call")

    # -- notes
    for i, topic in enumerate(NOTE_TOPICS):
        cs = rng.sample(CONCEPTS, 2)
        client = rng.choice(client_names)
        proj = rng.choice([p[0] for p in PROJECTS if p[1] == client] or project_names)
        body = (
            f"Thinking out loud, not a conclusion.\n\n"
            f"Touches {link(cs[0])} and {link(cs[1])}. "
            f"{link(client)} is the clearest example — see {link(proj)}."
        )
        write_md("note", topic, {"date": date_back(5 + i * 11)}, body)
        bump("note")

    return counts


if __name__ == "__main__":
    counts = build()
    total = sum(counts.values())
    print(f"demo vault built at {OUT}  ({total} files, seed {SEED})")
    for kind in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {kind:<10} {counts[kind]:>3}")

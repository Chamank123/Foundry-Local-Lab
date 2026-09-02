"""data.py — THE ONLY FILE THAT TOUCHES MY REAL DATA.

Every other module asks this one where the vault lives. Flip JARVIS_DEMO and
the whole system points somewhere else; nothing else needs to know which mode
it is in.

    JARVIS_DEMO=1   invented fixtures (default) — safe to screen-record
    JARVIS_DEMO=0   the real folders listed in JARVIS_ROOTS

Demo is the default on purpose: you opt in to your real life, never out of it.

Access is read-only. This module hands out paths; it never opens a file for
writing under any of them. Writes belong to memory.py and go to memory/ alone.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path

JARVIS_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = JARVIS_ROOT / "data" / "demo"
ENV_FILE = JARVIS_ROOT / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    """Read .env into os.environ. Real environment always wins."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env()


def is_demo() -> bool:
    return os.environ.get("JARVIS_DEMO", "1").strip() not in ("0", "false", "no")


@dataclass
class Source:
    """Where the vault is, and everything that went wrong finding it."""

    mode: str  # "demo" | "real"
    paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.paths)


def source() -> Source:
    """Resolve the folders to index. Never raises — problems come back as
    warnings so the UI can say them out loud instead of failing silently."""
    if is_demo():
        src = Source(mode="demo")
        if not DEMO_DIR.exists() or not any(DEMO_DIR.rglob("*.md")):
            ensure_demo(src)
        if DEMO_DIR.exists():
            src.paths = [DEMO_DIR]
        else:
            src.warnings.append(
                f"Demo fixtures missing at {DEMO_DIR} and could not be generated. "
                "Run: python3 data/generate.py"
            )
        return src

    src = Source(mode="real")
    raw = os.environ.get("JARVIS_ROOTS", "").strip()
    if not raw:
        src.warnings.append(
            "JARVIS_DEMO=0 but JARVIS_ROOTS is not set, so there is nothing to "
            "index. Set JARVIS_ROOTS in .env to a list of folders separated by "
            f"'{os.pathsep}'."
        )
        return src

    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        p = Path(part).expanduser()
        if not p.exists():
            src.warnings.append(f"Configured folder does not exist: {p}")
        elif not p.is_dir():
            src.warnings.append(f"Configured path is not a folder: {p}")
        else:
            src.paths.append(p.resolve())

    if not src.paths:
        src.warnings.append("No readable folders in JARVIS_ROOTS — nothing to index.")
    return src


def ensure_demo(src: Source | None = None) -> None:
    """Build the demo vault if it is missing, so a clean machine runs with one
    command. Deterministic: same seed, same graph, every time."""
    import importlib.util

    gen = JARVIS_ROOT / "data" / "generate.py"
    if not gen.exists():
        if src is not None:
            src.warnings.append(f"Demo generator missing: {gen}")
        return
    spec = importlib.util.spec_from_file_location("jarvis_demo_generator", gen)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build()


def today() -> datetime.date:
    """The date the assistant should reason from.

    In demo mode that is the date the fixtures were generated around, so a
    recorded demo keeps making sense a month later. In real mode it is simply
    today. Nothing else in the codebase calls date.today() directly.
    """
    if is_demo():
        anchor = DEMO_DIR / ".anchor"
        try:
            return datetime.date.fromisoformat(anchor.read_text().strip())
        except (OSError, ValueError):
            pass
    return datetime.date.today()


def describe() -> dict:
    """A summary the UI can show without knowing anything about paths."""
    src = source()
    return {
        "mode": src.mode,
        "demo": src.mode == "demo",
        "roots": [str(p) for p in src.paths],
        "warnings": list(src.warnings),
        "today": today().isoformat(),
    }

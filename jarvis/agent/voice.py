"""voice.py — speech in and out, through ElevenLabs.

The API key lives here and never leaves the machine. The browser posts text to
/api/speak and audio to /api/listen; this module holds the credential and
returns mp3 bytes or a transcript. Nothing sensitive appears in devtools, a
screen recording, or the page source.

Speech in is ElevenLabs Scribe (`/v1/speech-to-text`, `scribe_v1`), not the
browser's Web Speech API. Web Speech exists only in Chrome, ships audio to
Google, and in Brave is a stub that fails silently — you talk and nothing
happens, with no error. Recording with MediaRecorder and transcribing here
works in every browser.

Both directions are metered and cost money, so there is a per-run budget. When
it is spent, JARVIS says so rather than quietly continuing to bill.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

try:
    from . import data
except ImportError:  # pragma: no cover
    import data  # type: ignore

API = "https://api.elevenlabs.io/v1"
TIMEOUT = float(os.environ.get("JARVIS_VOICE_TIMEOUT", "45"))

# Defaults chosen for latency and cost; both overridable in .env.
DEFAULT_TTS_MODEL = "eleven_turbo_v2_5"
STT_MODEL = "scribe_v1"

# A runaway loop — it hears its own voice, answers, hears itself again — is the
# expensive failure. These caps are the backstop under the mic-goes-deaf rule.
BUDGET_CHARS = int(os.environ.get("JARVIS_TTS_BUDGET_CHARS", "40000"))
BUDGET_SECONDS = float(os.environ.get("JARVIS_STT_BUDGET_SECONDS", "1800"))
MAX_AUDIO_BYTES = 12 * 1024 * 1024


class VoiceError(RuntimeError):
    """Something failed. It is always said out loud, never swallowed."""

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


@dataclass
class Spend:
    """What this run has cost so far, in the units each service bills in."""

    chars: int = 0
    seconds: float = 0.0
    calls: int = 0
    started: float = field(default_factory=time.time)

    def as_json(self) -> dict:
        return {
            "tts_chars": self.chars, "tts_budget": BUDGET_CHARS,
            "stt_seconds": round(self.seconds, 1), "stt_budget": BUDGET_SECONDS,
            "calls": self.calls,
        }


SPEND = Spend()


def _key() -> str:
    data.load_env()
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise VoiceError(
            "No ELEVENLABS_API_KEY. Put it in .env — it stays on this machine "
            "and never reaches the browser.", fatal=True)
    return key


def _request(path: str, *, data_bytes: bytes, content_type: str,
             accept: str = "application/json") -> bytes:
    req = urllib.request.Request(
        API + path, data=data_bytes,
        headers={"xi-api-key": _key(), "Content-Type": content_type,
                 "Accept": accept})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        if exc.code in (401, 403):
            raise VoiceError("ElevenLabs rejected the key (401/403). Check "
                             "ELEVENLABS_API_KEY.", fatal=True) from exc
        if exc.code == 429:
            raise VoiceError("ElevenLabs rate-limited or out of quota.") from exc
        raise VoiceError(f"ElevenLabs returned {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise VoiceError(f"Could not reach ElevenLabs: {exc}") from exc


# ------------------------------------------------------------------ voices ---
_voice_cache: dict = {}


def voices(refresh: bool = False) -> list[dict]:
    """The account's voices. Fetched rather than assumed: a hard-coded voice id
    that does not exist on this account fails at the worst possible moment."""
    if _voice_cache.get("list") and not refresh:
        return _voice_cache["list"]
    req = urllib.request.Request(API + "/voices", headers={"xi-api-key": _key()})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise VoiceError("ElevenLabs rejected the key.", fatal=True) from exc
        raise VoiceError(f"Could not list voices ({exc.code}).") from exc
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        raise VoiceError(f"Could not list voices: {exc}") from exc

    found = [{"id": v.get("voice_id"), "name": v.get("name", "")}
             for v in payload.get("voices", []) if v.get("voice_id")]
    _voice_cache["list"] = found
    return found


def voice_id() -> str:
    configured = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
    if configured:
        return configured
    found = voices()
    if not found:
        raise VoiceError("No voices on this ElevenLabs account.", fatal=True)
    return found[0]["id"]


# --------------------------------------------------------------------- out ---
def speak(text: str) -> bytes:
    """Text to mp3 bytes. Returns audio the browser plays; the key stays here."""
    text = " ".join((text or "").split())
    if not text:
        raise VoiceError("Nothing to say.")
    if len(text) > 2500:
        text = text[:2500]

    if SPEND.chars + len(text) > BUDGET_CHARS:
        raise VoiceError(
            f"Speech budget for this run is spent ({SPEND.chars} of "
            f"{BUDGET_CHARS} characters). Raise JARVIS_TTS_BUDGET_CHARS if you "
            f"meant to keep going.", fatal=True)

    model = os.environ.get("ELEVENLABS_TTS_MODEL", DEFAULT_TTS_MODEL)
    body = json.dumps({
        "text": text,
        "model_id": model,
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.7},
    }).encode("utf-8")
    audio = _request(
        f"/text-to-speech/{voice_id()}?output_format=mp3_44100_128",
        data_bytes=body, content_type="application/json", accept="audio/mpeg")
    SPEND.chars += len(text)
    SPEND.calls += 1
    return audio


# ---------------------------------------------------------------------- in ---
def _multipart(fields: dict[str, str], filename: str, blob: bytes,
               mime: str) -> tuple[bytes, str]:
    """Build a multipart body by hand — no requests, no dependencies."""
    boundary = f"----jarvis{uuid.uuid4().hex}"
    out = bytearray()
    for name, value in fields.items():
        out += (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n").encode("utf-8")
    out += (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n").encode("utf-8")
    out += blob
    out += f"\r\n--{boundary}--\r\n".encode("utf-8")
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def listen(blob: bytes, mime: str = "audio/webm", seconds: float = 0.0) -> dict:
    """Audio to transcript, via Scribe."""
    if not blob:
        raise VoiceError("No audio arrived.")
    if len(blob) > MAX_AUDIO_BYTES:
        raise VoiceError("That recording is too long to send in one piece.")
    if SPEND.seconds + seconds > BUDGET_SECONDS:
        raise VoiceError(
            f"Transcription budget for this run is spent "
            f"({SPEND.seconds:.0f}s of {BUDGET_SECONDS:.0f}s). Raise "
            f"JARVIS_STT_BUDGET_SECONDS if you meant to keep going.", fatal=True)

    ext = "webm" if "webm" in mime else ("ogg" if "ogg" in mime else "mp4")
    body, content_type = _multipart(
        {"model_id": STT_MODEL}, f"turn.{ext}", blob, mime)
    raw = _request("/speech-to-text", data_bytes=body, content_type=content_type)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise VoiceError("Scribe returned something unreadable.") from exc

    SPEND.seconds += seconds
    SPEND.calls += 1
    return {"text": (payload.get("text") or "").strip(),
            "language": payload.get("language_code", ""),
            "spend": SPEND.as_json()}


# ------------------------------------------------------------------ status ---
def status() -> dict:
    """Whether voice can work, and if not, exactly why — on screen, not in a log."""
    out = {"tts": False, "stt": False, "reason": "", "spend": SPEND.as_json(),
           "voice": "", "voice_name": "", "model": os.environ.get(
               "ELEVENLABS_TTS_MODEL", DEFAULT_TTS_MODEL),
           "stt_model": STT_MODEL}
    try:
        _key()
    except VoiceError as exc:
        out["reason"] = str(exc)
        return out
    try:
        found = voices()
        chosen = voice_id()
        out.update({
            "tts": True, "stt": True, "voice": chosen,
            "voice_name": next((v["name"] for v in found if v["id"] == chosen), ""),
        })
    except VoiceError as exc:
        out["reason"] = str(exc)
    return out

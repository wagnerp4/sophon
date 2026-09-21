from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from utils.device.env_bootstrap import sophon_chat_audio_dir

SpeechRole = Literal["user", "assistant"]
SpeechSource = Literal["tts", "mic", "file"]


@dataclass(frozen=True)
class SpeechClip:
    clip_id: int
    role: SpeechRole
    source: SpeechSource
    path: Path
    duration_s: float
    sample_rate: int
    created_at: str
    session_id: str


def _safe_session_id(raw: str) -> str:
    token = raw.strip() or "session"
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token)
    return token[:80] or "session"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_replay_speed(speed: float) -> str:
    value = float(speed)
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}x"
    text = f"{value:g}"
    if text.endswith("."):
        text = text[:-1]
    return f"{text}x"


def parse_replay_speed(raw: str) -> float | None:
    token = raw.strip().lower()
    if token.endswith("x"):
        token = token[:-1].strip()
    if not token:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    if value < 0.25 or value > 8.0:
        return None
    return value


def default_replay_speeds() -> tuple[float, ...]:
    return (1.0, 1.5, 2.0, 2.5, 3.0)


class SessionSpeechStore:
    def __init__(self, session_id: str, root: Path) -> None:
        self.session_id = _safe_session_id(session_id)
        self.root = root
        self.clips: list[SpeechClip] = []
        self._next_id = 1

    @classmethod
    def open(cls, session_id: str) -> "SessionSpeechStore":
        safe = _safe_session_id(session_id)
        root = sophon_chat_audio_dir() / safe
        root.mkdir(parents=True, exist_ok=True)
        store = cls(session_id=safe, root=root)
        store._load_manifest()
        return store

    @property
    def manifest_path(self) -> Path:
        return self.root / "clips.json"

    def get(self, clip_id: int) -> SpeechClip | None:
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        return None

    def last(self, role: SpeechRole | None = None) -> SpeechClip | None:
        for clip in reversed(self.clips):
            if role is None or clip.role == role:
                return clip
        return None

    def save_wave(
        self,
        wave,
        sample_rate: int,
        *,
        role: SpeechRole,
        source: SpeechSource,
    ) -> SpeechClip:
        try:
            import numpy as np
            import soundfile as sf
        except ImportError as exc:
            raise ImportError("soundfile unavailable - install with: uv sync --extra tts") from exc

        audio = np.asarray(wave)
        rate = max(int(sample_rate), 1)
        if audio.size == 0:
            raise ValueError("empty audio clip")
        clip_id = self._next_id
        path = self._clip_path(clip_id, role, source)
        sf.write(str(path), audio, rate)
        duration = float(audio.shape[0]) / float(rate)
        clip = SpeechClip(
            clip_id=clip_id,
            role=role,
            source=source,
            path=path,
            duration_s=duration,
            sample_rate=rate,
            created_at=_utc_now(),
            session_id=self.session_id,
        )
        self._next_id = clip_id + 1
        self.clips.append(clip)
        self._write_manifest()
        return clip

    def ingest_file(
        self,
        src: Path,
        *,
        role: SpeechRole,
        source: SpeechSource = "file",
    ) -> SpeechClip:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError("soundfile unavailable - install with: uv sync --extra tts") from exc

        src = src.expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(str(src))
        clip_id = self._next_id
        dest = self._clip_path(clip_id, role, source)
        try:
            data, rate = sf.read(str(src), always_2d=False)
            sf.write(str(dest), data, int(rate))
            frames = int(getattr(data, "shape", [0])[0])
            duration = float(frames) / float(max(int(rate), 1))
            sample_rate = int(rate)
        except Exception:
            shutil.copy2(src, dest)
            duration = 0.0
            sample_rate = 0
            try:
                info = sf.info(str(dest))
                duration = float(info.duration)
                sample_rate = int(info.samplerate)
            except Exception:
                pass
        clip = SpeechClip(
            clip_id=clip_id,
            role=role,
            source=source,
            path=dest,
            duration_s=duration,
            sample_rate=sample_rate,
            created_at=_utc_now(),
            session_id=self.session_id,
        )
        self._next_id = clip_id + 1
        self.clips.append(clip)
        self._write_manifest()
        return clip

    def _clip_path(self, clip_id: int, role: SpeechRole, source: SpeechSource) -> Path:
        return self.root / f"{clip_id:04d}_{role}_{source}.wav"

    def _load_manifest(self) -> None:
        path = self.manifest_path
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        items = raw.get("clips") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return
        clips: list[SpeechClip] = []
        max_id = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                clip_id = int(item["clip_id"])
                role = str(item.get("role") or "assistant")
                source = str(item.get("source") or "tts")
                if role not in ("user", "assistant"):
                    continue
                if source not in ("tts", "mic", "file"):
                    continue
                filename = str(item.get("filename") or "")
                clip_path = (self.root / filename).resolve() if filename else Path(str(item.get("path") or ""))
                if not clip_path.is_file():
                    continue
                clip = SpeechClip(
                    clip_id=clip_id,
                    role=role,
                    source=source,
                    path=clip_path,
                    duration_s=float(item.get("duration_s") or 0.0),
                    sample_rate=int(item.get("sample_rate") or 0),
                    created_at=str(item.get("created_at") or ""),
                    session_id=self.session_id,
                )
            except (KeyError, TypeError, ValueError):
                continue
            clips.append(clip)
            max_id = max(max_id, clip.clip_id)
        self.clips = clips
        self._next_id = max_id + 1
        # TODO(speech): optional prune of session dirs older than N days.

    def _write_manifest(self) -> None:
        payload = {
            "session_id": self.session_id,
            "clips": [
                {
                    "clip_id": clip.clip_id,
                    "role": clip.role,
                    "source": clip.source,
                    "filename": clip.path.name,
                    "duration_s": clip.duration_s,
                    "sample_rate": clip.sample_rate,
                    "created_at": clip.created_at,
                }
                for clip in self.clips
            ],
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

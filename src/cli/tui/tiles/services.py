from __future__ import annotations

import os
import time
from typing import Any

from cli.tui.tiles.base import BaseTile, TileState, wrap_fetch


def _probe_ollama() -> dict[str, Any]:
    from backend.ollama.backend import list_model_names, ollama_base_url, ping_daemon

    base = ollama_base_url()
    try:
        ping_daemon(base, timeout_s=2.0)
        models = list_model_names(base, timeout_s=4.0)
        return {
            "name": "ollama",
            "ok": True,
            "detail": f"up  models={len(models)}",
            "models": models[:5],
            "url": base,
        }
    except Exception as exc:
        return {"name": "ollama", "ok": False, "detail": str(exc), "models": [], "url": base}


def _probe_lmstudio() -> dict[str, Any]:
    from backend.lmstudio.backend import list_model_names, lmstudio_base_url, ping_daemon

    base = lmstudio_base_url()
    try:
        up = ping_daemon(base, timeout_s=2.0)
        if not up:
            return {"name": "lmstudio", "ok": False, "detail": "unreachable", "models": [], "url": base}
        models = list_model_names(base, timeout_s=4.0)
        shown = ", ".join(models[:3]) if models else "(none loaded)"
        return {
            "name": "lmstudio",
            "ok": True,
            "detail": f"up  {shown}",
            "models": models[:5],
            "url": base,
        }
    except Exception as exc:
        return {"name": "lmstudio", "ok": False, "detail": str(exc), "models": [], "url": base}


def _probe_tts() -> dict[str, Any]:
    backend = os.environ.get("ORODRUIN_TTS_BACKEND", "pipecat").strip() or "pipecat"
    speaker = os.environ.get("ORODRUIN_TTS_SPEAKER", "").strip() or "-"
    tool = os.environ.get("ORODRUIN_TTS_TOOL", "1").strip()
    tool_on = tool.lower() not in ("", "0", "false", "no", "off")
    detail = f"backend={backend} speaker={speaker} tool={'on' if tool_on else 'off'}"
    ok = True
    if backend.lower() in ("pipecat", "kokoro"):
        try:
            from kokoro_onnx import Kokoro  # noqa: F401
        except Exception as exc:
            ok = False
            detail = f"{detail}  import fail: {exc}"
    return {"name": "tts", "ok": ok, "detail": detail, "models": [], "url": ""}


def _probe_obsidian() -> dict[str, Any]:
    from integrations.obsidian.client import (
        ObsidianClient,
        obsidian_api_url,
        obsidian_tools_enabled,
    )

    if not obsidian_tools_enabled():
        return {
            "name": "obsidian",
            "ok": False,
            "detail": "tools off",
            "models": [],
            "url": obsidian_api_url(),
        }
    try:
        ping = ObsidianClient(timeout_s=3.0).ping()
        auth = bool(ping.get("authenticated")) if isinstance(ping, dict) else False
        return {
            "name": "obsidian",
            "ok": True,
            "detail": "auth ok" if auth else "reachable",
            "models": [],
            "url": obsidian_api_url(),
        }
    except Exception as exc:
        return {
            "name": "obsidian",
            "ok": False,
            "detail": str(exc),
            "models": [],
            "url": obsidian_api_url(),
        }


def _probe_sst() -> dict[str, Any]:
    backend = os.environ.get("ORODRUIN_SST_BACKEND", "hf_qwen").strip() or "hf_qwen"
    model = os.environ.get("ORODRUIN_SST_MODEL", "").strip() or "Qwen/Qwen3-ASR-0.6B"
    tool = os.environ.get("ORODRUIN_SST_TOOL", "1").strip()
    tool_on = tool.lower() not in ("", "0", "false", "no", "off")
    detail = f"backend={backend} model={model} tool={'on' if tool_on else 'off'}"
    ok = True
    try:
        import qwen_asr  # noqa: F401
    except Exception as exc:
        ok = False
        detail = f"{detail}  import fail: {exc}"
    try:
        import sounddevice  # noqa: F401
    except Exception as exc:
        ok = False
        detail = f"{detail}  mic fail: {exc}"
    return {"name": "sst", "ok": ok, "detail": detail, "models": [model], "url": ""}


def _fetch_services() -> TileState:
    probes = [
        _probe_ollama(),
        _probe_lmstudio(),
        _probe_tts(),
        _probe_sst(),
        _probe_obsidian(),
    ]
    chat_backend = os.environ.get("ORODRUIN_CHAT_BACKEND", "auto").strip() or "auto"
    lines = ["services", f"chat_backend={chat_backend}"]
    for row in probes:
        mark = "ok" if row.get("ok") else "down"
        lines.append(f"{row['name']}: [{mark}] {row.get('detail')}")
    any_ok = any(bool(row.get("ok")) for row in probes)
    return TileState(
        ok=any_ok,
        text="\n".join(lines),
        fetched_at=time.time(),
        payload={"services": probes, "chat_backend": chat_backend},
        error="" if any_ok else "all services down",
    )


class ServicesTile(BaseTile):
    tile_id = "services"
    refresh_s = 30.0

    def fetch(self, app: Any) -> TileState:
        return wrap_fetch(self.tile_id, "services", _fetch_services)

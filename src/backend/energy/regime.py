from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from backend.providers import provider_api_key, provider_spec

LOCAL_BACKENDS = ("hf", "ollama", "lmstudio")
# TODO: add deepseek when it is a ChatBackendId
API_BACKENDS = ("openai", "anthropic", "google")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class EnergySession:
    regime: str = "local"
    daily_usd: float = 5.0
    monthly_usd: float = 40.0
    rpm: int = 60
    on_limit: str = "ask"
    ask_timeout_s: float = 60.0
    last_local_backend: str = ""
    last_local_model: str = ""
    last_api: dict[str, str] = field(default_factory=dict)
    warned_80: bool = False
    project_root: Path = field(default_factory=Path.cwd)


def is_api_backend(backend_id: str) -> bool:
    spec = provider_spec(str(backend_id or "").strip().lower())
    if spec is None:
        return False
    return not spec.is_local


def energy_file(session: EnergySession) -> Path:
    return session.project_root / ".sophon" / "energy.local.yaml"


def defaults_from_env(project_root: Path | None = None) -> EnergySession:
    regime = os.environ.get("SOPHON_ENERGY", "local").strip().lower() or "local"
    if regime not in ("local", "api"):
        regime = "local"
    on_limit = os.environ.get("SOPHON_ENERGY_ON_LIMIT", "ask").strip().lower() or "ask"
    if on_limit != "ask":
        on_limit = "ask"
    session = EnergySession(
        regime=regime,
        daily_usd=_env_float("SOPHON_ENERGY_DAILY_USD", 5.0),
        monthly_usd=_env_float("SOPHON_ENERGY_MONTHLY_USD", 40.0),
        rpm=_env_int("SOPHON_ENERGY_RPM", 60),
        on_limit=on_limit,
        ask_timeout_s=_env_float("SOPHON_ENERGY_ASK_TIMEOUT_S", 60.0),
        project_root=project_root or Path.cwd(),
    )
    return session


def _apply_mapping(session: EnergySession, data: dict) -> None:
    regime = str(data.get("regime") or "").strip().lower()
    if regime in ("local", "api"):
        session.regime = regime
    for key, cast in (
        ("daily_usd", float),
        ("monthly_usd", float),
        ("ask_timeout_s", float),
    ):
        if key in data:
            try:
                setattr(session, key, cast(data[key]))
            except (TypeError, ValueError):
                pass
    if "rpm" in data:
        try:
            session.rpm = int(data["rpm"])
        except (TypeError, ValueError):
            pass
    last_local = data.get("last_local")
    if isinstance(last_local, dict):
        session.last_local_backend = str(last_local.get("backend") or "")
        session.last_local_model = str(last_local.get("model") or "")
    last_api = data.get("last_api")
    if isinstance(last_api, dict):
        session.last_api = {str(k): str(v) for k, v in last_api.items() if v}


def load_persisted(session: EnergySession) -> None:
    path = energy_file(session)
    if not path.is_file():
        return
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return
    if isinstance(data, dict):
        _apply_mapping(session, data)


def persist_energy(session: EnergySession) -> Path:
    path = energy_file(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "regime": session.regime,
        "daily_usd": session.daily_usd,
        "monthly_usd": session.monthly_usd,
        "rpm": session.rpm,
        "on_limit": session.on_limit,
        "ask_timeout_s": session.ask_timeout_s,
        "last_local": {
            "backend": session.last_local_backend,
            "model": session.last_local_model,
        },
        "last_api": dict(session.last_api),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def ensure_energy(state: object) -> EnergySession:
    current = getattr(state, "energy", None)
    if isinstance(current, EnergySession):
        return current
    root = getattr(state, "project_root", None)
    session = defaults_from_env(root if isinstance(root, Path) else None)
    load_persisted(session)
    if session.regime == "api" and not any(provider_api_key(name) for name in API_BACKENDS):
        session.regime = "local"
    setattr(state, "energy", session)
    return session


def status_line(session: EnergySession, *, spent_today: float) -> str:
    cap = session.daily_usd
    cap_text = "off" if cap <= 0 else f"{spent_today:.2f}/{cap:.2f}"
    return f"energy={session.regime} ${cap_text}"


def api_choices() -> list[tuple[str, str, bool]]:
    rows: list[tuple[str, str, bool]] = []
    for name in API_BACKENDS:
        key = provider_api_key(name)
        if key:
            rows.append((name, name, True))
        else:
            spec = provider_spec(name)
            hint = spec.key_hint if spec is not None else "API key"
            rows.append((name, f"{name} (set {hint})", False))
    return rows

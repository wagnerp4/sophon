from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from backend.energy.ledger import append_spend, requests_last_minute, sum_month, sum_today
from backend.energy.regime import EnergySession, ensure_energy, is_api_backend

_PRICE_PATH = Path(__file__).with_name("prices.yaml")


def _rates() -> dict:
    try:
        data = yaml.safe_load(_PRICE_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


def estimate_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> tuple[float, str]:
    table = _rates()
    models = table.get("models") if isinstance(table.get("models"), dict) else {}
    providers = table.get("providers") if isinstance(table.get("providers"), dict) else {}
    key = str(model or "").strip().lower()
    rates = None
    tag = "rough"
    if key and key in models and isinstance(models[key], dict):
        rates = models[key]
        tag = "table"
    else:
        for name, row in models.items():
            if key and (key.startswith(str(name)) or str(name) in key) and isinstance(row, dict):
                rates = row
                tag = "table"
                break
    if rates is None:
        row = providers.get(provider)
        rates = row if isinstance(row, dict) else {"input": 2.0, "output": 8.0}
        tag = "rough"
    try:
        in_rate = float(rates.get("input") or 0.0)
        out_rate = float(rates.get("output") or 0.0)
    except (TypeError, ValueError):
        in_rate, out_rate, tag = 2.0, 8.0, "rough"
    usd = (max(input_tokens, 0) * in_rate + max(output_tokens, 0) * out_rate) / 1_000_000.0
    return usd, tag


def _note(text: str) -> None:
    try:
        from cli.chat import _emit

        _emit(text)
    except Exception:
        return


def _ask(title: str, timeout_s: float) -> str:
    try:
        from cli.chat import prompt_choice
    except Exception:
        return "stop"
    choice = prompt_choice(
        title,
        [
            ("continue", "1 continue this request", True),
            ("stop", "3 stop", True),
        ],
        timeout_s,
    )
    if choice == "continue":
        return "continue"
    return "stop"


def guard_completion(state: object, max_new_tokens: int, *, purpose: str = "chat") -> str | None:
    backend = str(getattr(state, "backend_id", "") or "")
    if not is_api_backend(backend):
        return None
    session = ensure_energy(state)
    model = str(getattr(state, "server_model", "") or "")
    est_in = 4000
    est_out = max(int(max_new_tokens), 1)
    usd, tag = estimate_usd(backend, model, est_in, est_out)
    today = sum_today(session)
    month = sum_month(session)
    if session.daily_usd > 0 and today >= session.daily_usd * 0.8 and not session.warned_80:
        session.warned_80 = True
        _note(f"(energy: ${today:.2f} of ${session.daily_usd:.2f} daily)")
    reason = ""
    if session.daily_usd > 0 and today + usd > session.daily_usd:
        reason = (
            f"daily cap ${session.daily_usd:.2f} "
            f"(spent ${today:.2f}, est ${usd:.2f} {tag}, {purpose})"
        )
    elif session.monthly_usd > 0 and month + usd > session.monthly_usd:
        reason = f"monthly cap ${session.monthly_usd:.2f} (spent ${month:.2f}, est ${usd:.2f})"
    elif session.rpm > 0 and requests_last_minute(session) >= session.rpm:
        reason = f"rpm cap {session.rpm}"
    if not reason:
        return None
    if _ask(f"Energy cap: {reason}", session.ask_timeout_s) == "continue":
        return None
    return f"(energy stop: {reason})"


def record_completion(state: object, result: object, *, purpose: str = "chat") -> None:
    backend = str(getattr(state, "backend_id", "") or "")
    if not is_api_backend(backend):
        return
    if str(getattr(result, "finish_reason", "") or "") == "energy":
        return
    session = ensure_energy(state)
    model = str(getattr(state, "server_model", "") or "")
    in_tok = getattr(result, "prompt_tokens", None)
    out_tok = getattr(result, "completion_tokens", None)
    input_tokens = int(in_tok) if isinstance(in_tok, int) else 0
    output_tokens = int(out_tok) if isinstance(out_tok, int) else 0
    usd, tag = estimate_usd(backend, model, input_tokens, output_tokens)
    append_spend(
        session,
        {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "provider": backend,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "usd": round(usd, 6),
            "estimate": tag,
            "purpose": purpose,
            "session_id": str(getattr(state, "memory_session", "") or ""),
        },
    )

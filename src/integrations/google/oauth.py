from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from integrations.google.paths import (
    account_token_path,
    client_secrets_path,
    google_accounts_dir,
    google_active_path,
)

SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
)


def _require_google_libs() -> None:
    try:
        import google.auth  # noqa: F401
        import google_auth_oauthlib  # noqa: F401
        import googleapiclient  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Google libs missing. Install with: uv sync --extra tui "
            "(google-api-python-client, google-auth-oauthlib, google-auth-httplib2)"
        ) from exc


def google_available() -> bool:
    if client_secrets_path() is None and not list_accounts():
        return False
    return bool(list_accounts()) or client_secrets_path() is not None


def google_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_GOOGLE_TOOLS", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return bool(list_accounts())


def list_accounts() -> list[str]:
    root = google_accounts_dir()
    emails: list[str] = []
    try:
        paths = sorted(root.glob("*.json"))
    except OSError:
        return []
    for path in paths:
        email = _email_from_token_file(path)
        if email:
            emails.append(email)
    return emails


def active_email() -> str | None:
    path = google_active_path()
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, dict):
            email = str(raw.get("email") or "").strip()
            if email and account_token_path(email).is_file():
                return email
    accounts = list_accounts()
    return accounts[0] if accounts else None


def set_active_email(email: str) -> str:
    cleaned = (email or "").strip()
    if not cleaned:
        raise ValueError("email is required")
    if cleaned not in list_accounts():
        raise ValueError(f"unknown Google account: {cleaned}")
    google_active_path().write_text(
        json.dumps({"email": cleaned}, indent=2) + "\n",
        encoding="utf-8",
    )
    return cleaned


def disconnect_account(email: str | None = None) -> str | None:
    target = (email or "").strip() or active_email()
    if not target:
        return None
    path = account_token_path(target)
    if path.is_file():
        path.unlink()
    remaining = list_accounts()
    if remaining:
        if active_email() == target or active_email() is None:
            set_active_email(remaining[0])
    elif google_active_path().is_file():
        google_active_path().unlink()
    return target


def connect_account(*, open_browser: bool = True) -> str:
    _require_google_libs()
    from google_auth_oauthlib.flow import InstalledAppFlow

    secrets = client_secrets_path()
    if secrets is None:
        raise RuntimeError(
            "Set SOPHON_GOOGLE_CLIENT_SECRETS to a Desktop OAuth client JSON, "
            "or place it at data/google/client_secret.json"
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), list(SCOPES))
    try:
        creds = flow.run_local_server(port=0, open_browser=open_browser)
    except Exception:
        creds = flow.run_console()
    email = _email_from_credentials(creds)
    if not email:
        raise RuntimeError("OAuth succeeded but email was missing from userinfo")
    token_path = account_token_path(email)
    _write_token(token_path, creds, email=email)
    set_active_email(email)
    return email


def credentials_for(email: str | None = None) -> Any:
    _require_google_libs()
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    target = (email or "").strip() or active_email()
    if not target:
        raise RuntimeError(
            "No Google account connected. Use View → Google accounts → Connect… "
            "or call connect_account()."
        )
    path = account_token_path(target)
    if not path.is_file():
        raise RuntimeError(f"token missing for {target}")
    creds = Credentials.from_authorized_user_file(str(path), list(SCOPES))
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _write_token(path, creds, email=target)
        else:
            raise RuntimeError(f"Google credentials invalid for {target}. Reconnect the account.")
    return creds


def _write_token(path: Path, creds: Any, *, email: str) -> None:
    payload = json.loads(creds.to_json())
    payload["email"] = email
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_service(api: str, version: str, *, email: str | None = None) -> Any:
    _require_google_libs()
    from googleapiclient.discovery import build

    return build(api, version, credentials=credentials_for(email), cache_discovery=False)


def _email_from_credentials(creds: Any) -> str | None:
    _require_google_libs()
    from googleapiclient.discovery import build

    service = build("oauth2", "v2", credentials=creds, cache_discovery=False)
    info = service.userinfo().get().execute()
    email = str(info.get("email") or "").strip()
    return email or None


def _email_from_token_file(path: Path) -> str | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    for key in ("email", "account", "account_email"):
        value = str(raw.get(key) or "").strip()
        if value and "@" in value:
            return value
    stem = path.stem
    if "@" in stem:
        return stem
    return None


def format_accounts_status() -> dict[str, Any]:
    return {
        "tools": google_tools_enabled(),
        "client_secrets": str(client_secrets_path()) if client_secrets_path() else None,
        "accounts": list_accounts(),
        "active": active_email(),
    }

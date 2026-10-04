from __future__ import annotations

from dataclasses import dataclass

from utils.device.env_bootstrap import upsert_dotenv_key


@dataclass(frozen=True)
class KeyPrompt:
    env_key: str
    title: str
    steps: tuple[str, ...]


KEY_PROMPTS: dict[str, KeyPrompt] = {
    "github": KeyPrompt(
        "SOPHON_GITHUB_TOKEN",
        "GitHub",
        (
            "Open https://github.com/settings/personal-access-tokens",
            "Generate a fine-grained token.",
            "Repository access: the repositories sophon should read or create.",
            "Permissions: Contents (read and write). Administration (read and write) for github_create.",
            "Paste the token as your next message. It is saved and not printed back.",
        ),
    ),
    "openai": KeyPrompt(
        "SOPHON_OPENAI_API_KEY",
        "OpenAI",
        (
            "Open https://platform.openai.com/api-keys",
            "Create a secret key.",
            "Paste the key as your next message. It is saved and not printed back.",
        ),
    ),
    "anthropic": KeyPrompt(
        "SOPHON_ANTHROPIC_API_KEY",
        "Anthropic",
        (
            "Open https://console.anthropic.com/settings/keys",
            "Create a key.",
            "Paste the key as your next message. It is saved and not printed back.",
        ),
    ),
    "google": KeyPrompt(
        "SOPHON_GEMINI_API_KEY",
        "Gemini",
        (
            "Open https://aistudio.google.com/apikey",
            "Create an API key.",
            "Paste the key as your next message. It is saved and not printed back.",
        ),
    ),
}


def redact_secrets(text: str) -> str:
    import re

    patterns = (
        r"github_pat_[A-Za-z0-9_]+",
        r"ghp_[A-Za-z0-9]+",
        r"sk-[A-Za-z0-9_-]{8,}",
    )
    out = text or ""
    for pattern in patterns:
        out = re.sub(pattern, "[redacted]", out)
    return out


def normalize_secret(text: str) -> str:
    value = text.strip().strip("\"'")
    marker = "github_pat_"
    if value.count(marker) > 1:
        second = value.find(marker, len(marker))
        value = value[:second]
    return value


def contains_secret(text: str) -> bool:
    redacted = redact_secrets(text or "")
    return "[redacted]" in redacted
    spec = KEY_PROMPTS.get(key_id)
    if spec is None:
        return [f"unknown key {key_id}"]
    lines = [f"{spec.title} needs {spec.env_key}."]
    for index, step in enumerate(spec.steps, start=1):
        lines.append(f"{index}. {step}")
    lines.append("Cancel with /key cancel.")
    return lines


def begin_key_prompt(state: object, key_id: str) -> str:
    if key_id not in KEY_PROMPTS:
        return f"error: unknown key {key_id}"
    setattr(state, "pending_secret_key", key_id)
    return "\n".join(prompt_lines(key_id))


def take_pending_secret(state: object, line: str) -> str | None:
    key_id = str(getattr(state, "pending_secret_key", "") or "")
    if not key_id:
        return None
    text = line.strip()
    if text.startswith("/"):
        return None
    spec = KEY_PROMPTS.get(key_id)
    if spec is None:
        setattr(state, "pending_secret_key", "")
        return "error: unknown pending key"
    value = normalize_secret(text)
    if len(value) < 12 or any(ch.isspace() for ch in value):
        return "That does not look like a key. Paste only the token, or /key cancel."
    upsert_dotenv_key(spec.env_key, value)
    setattr(state, "pending_secret_key", "")
    if spec.env_key == "SOPHON_GITHUB_TOKEN":
        from integrations.github.client import github_request

        _code, payload, err = github_request("GET", "/user")
        if err:
            import os

            os.environ.pop(spec.env_key, None)
            setattr(state, "pending_secret_key", key_id)
            return (
                "GitHub rejected that token. "
                "Create a new one, paste a single copy, and revoke any token that was shown in the chat."
            )
        login = payload.get("login") if isinstance(payload, dict) else ""
        return f"Saved {spec.env_key}. GitHub accepted it as {login}."
    return f"Saved {spec.env_key}. It is not shown again."


def missing_key_ids() -> list[str]:
    import os

    missing = []
    for key_id, spec in KEY_PROMPTS.items():
        if not os.environ.get(spec.env_key, "").strip():
            missing.append(key_id)
    return missing

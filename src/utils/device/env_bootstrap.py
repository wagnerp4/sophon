from __future__ import annotations

import os
import sys
from pathlib import Path

DOTENV_LOAD_PATH: Path | None = None


def sophon_project_root() -> Path:
    for anc in Path(__file__).resolve().parents[:10]:
        if (anc / "pyproject.toml").is_file():
            return anc
    return Path.cwd().resolve()


def sophon_version_string() -> str:
    try:
        from importlib.metadata import version as pkg_version
    except ImportError:
        pkg_version = None
    if pkg_version is not None:
        try:
            found = str(pkg_version("sophon")).strip()
            if found:
                return found
        except Exception:
            pass
    root = sophon_project_root()
    text = (root / "pyproject.toml").read_text(encoding="utf-8") if (root / "pyproject.toml").is_file() else ""
    for line in text.splitlines():
        raw = line.strip()
        if raw.startswith("version") and "=" in raw:
            _, _, value = raw.partition("=")
            token = value.strip().strip("\"'")
            if token:
                return token
    return "0.1.0"


def sophon_agent_root() -> Path:
    home = Path.home()
    if sys.platform == "win32":
        profile = os.environ.get("USERPROFILE", "").strip()
        if profile:
            return Path(profile)
    return home


def sophon_data_dir() -> Path:
    return sophon_project_root() / "data"


def sophon_chat_logs_dir() -> Path:
    from utils.device.platform import is_wsl, is_windows_mount_path

    root = sophon_project_root()
    if is_wsl() and is_windows_mount_path(root):
        target = Path.home() / ".cache" / "sophon" / "chat_logs"
    else:
        target = sophon_data_dir() / "chat_logs"
    target.mkdir(parents=True, exist_ok=True)
    return target


def sophon_chat_audio_dir() -> Path:
    from utils.device.platform import is_wsl, is_windows_mount_path

    root = sophon_project_root()
    if is_wsl() and is_windows_mount_path(root):
        target = Path.home() / ".cache" / "sophon" / "chat_audio"
    else:
        target = sophon_data_dir() / "chat_audio"
    target.mkdir(parents=True, exist_ok=True)
    return target


def sophon_assets_dir() -> Path:
    return sophon_data_dir() / "assets"


def sophon_memory_dir(project_root: Path | None = None) -> Path:
    explicit = os.environ.get("SOPHON_MEMORY_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    root = project_root or sophon_project_root()
    return root / ".sophon" / "memory"


def _dotenv_path() -> Path | None:
    explicit = os.environ.get("SOPHON_ENV_FILE", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None

    root_env = sophon_project_root() / ".env"
    if root_env.is_file():
        return root_env.resolve()

    here = Path.cwd().resolve()
    for d in [here, *here.parents][:14]:
        for rel in (Path(".env"), Path("sophon") / ".env"):
            cand = (d / rel).resolve()
            if cand.is_file():
                return cand
    return None


def load_sophon_dotenv() -> None:
    global DOTENV_LOAD_PATH

    DOTENV_LOAD_PATH = None
    if os.environ.get("SOPHON_SKIP_DOTENV", "").strip().lower() in ("1", "true", "yes"):
        return
    from dotenv import load_dotenv

    path = _dotenv_path()
    if path is not None:
        load_dotenv(path, override=True, encoding="utf-8-sig")
        DOTENV_LOAD_PATH = path


DOTENV_SYNC_KEYS: tuple[str, ...] = (
    "SOPHON_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "SOPHON_ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY",
    "SOPHON_GEMINI_API_KEY",
    "GEMINI_API_KEY",
)


def dotenv_location_label() -> str:
    if DOTENV_LOAD_PATH is not None:
        return str(DOTENV_LOAD_PATH)
    return ".env"


def strip_env_value(raw: str) -> str:
    text = (raw or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1].strip()
    return text


def parse_dotenv_assignments(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            out[key] = value
    return out


def merge_dotenv_keys(
    source: Path,
    dest: Path,
    keys: tuple[str, ...] | None = None,
) -> list[str]:
    wanted = keys if keys is not None else DOTENV_SYNC_KEYS
    if not source.is_file():
        return []
    src_map = parse_dotenv_assignments(source.read_text(encoding="utf-8-sig"))
    dest_text = dest.read_text(encoding="utf-8-sig") if dest.is_file() else ""
    dest_map = parse_dotenv_assignments(dest_text)
    lines = dest_text.splitlines()
    changed: list[str] = []
    for key in wanted:
        if key not in src_map:
            continue
        new_val = src_map[key]
        stripped = strip_env_value(new_val)
        if not stripped:
            continue
        existing = dest_map.get(key)
        if existing is not None and strip_env_value(existing) == stripped:
            continue
        found = False
        for i, line in enumerate(lines):
            work = line.strip()
            if not work or work.startswith("#"):
                continue
            export_prefix = ""
            if work.lower().startswith("export "):
                export_prefix = "export "
                work = work[7:].strip()
            if "=" not in work:
                continue
            if work.partition("=")[0].strip() == key:
                lines[i] = f"{export_prefix}{key}={new_val}"
                found = True
                break
        if not found:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(f"{key}={new_val}")
        changed.append(key)
        dest_map[key] = new_val
    if not changed:
        return []
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines)
    if body and not body.endswith("\n"):
        body += "\n"
    dest.write_text(body, encoding="utf-8")
    return changed


def upsert_dotenv_key(key: str, value: str, path: Path | None = None) -> Path:
    target = path or DOTENV_LOAD_PATH
    if target is None:
        target = sophon_project_root() / ".env"
    text = target.read_text(encoding="utf-8-sig") if target.is_file() else ""
    lines = text.splitlines()
    stored = value.replace("\n", "").replace("\r", "")
    found = False
    for index, line in enumerate(lines):
        work = line.strip()
        if not work or work.startswith("#") or "=" not in work:
            continue
        if work.partition("=")[0].strip() == key:
            lines[index] = f"{key}={stored}"
            found = True
            break
    if not found:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{key}={stored}")
    body = "\n".join(lines)
    if body and not body.endswith("\n"):
        body += "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    os.environ[key] = stored
    return target

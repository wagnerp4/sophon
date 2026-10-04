from __future__ import annotations

import os
from pathlib import Path


def vault_root() -> Path | None:
    raw = os.environ.get("SOPHON_OBSIDIAN_VAULT", "").strip()
    if raw:
        path = Path(raw)
        if path.is_dir():
            return path
    for candidate in (
        Path("C:/Notes/Obsidian Notes"),
        Path("/mnt/c/Notes/Obsidian Notes"),
    ):
        if (candidate / "Personal KB").is_dir():
            return candidate
    return None


class VaultStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root if root is not None else vault_root()

    def read(self, rel: str) -> str:
        path = self._disk(rel)
        if path is not None and path.is_file():
            return path.read_text(encoding="utf-8")
        from integrations.obsidian.client import ObsidianClient

        return ObsidianClient().get_file(rel, limit=None)

    def write(self, rel: str, content: str) -> None:
        path = self._disk(rel)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return
        from integrations.obsidian.client import ObsidianClient

        ObsidianClient().put_file(rel, content)

    def exists(self, rel: str) -> bool:
        path = self._disk(rel)
        if path is not None:
            return path.is_file()
        try:
            self.read(rel)
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    def list_md(self, rel_dir: str) -> list[str]:
        path = self._disk(rel_dir)
        if path is not None:
            if not path.is_dir():
                return []
            names = sorted(item.name for item in path.iterdir() if item.suffix == ".md")
            return [rel_dir.rstrip("/") + "/" + name for name in names]
        from integrations.obsidian.client import ObsidianClient

        try:
            entries = ObsidianClient().list_dir(rel_dir)
        except RuntimeError as exc:
            if "401" in str(exc):
                raise RuntimeError(
                    "Obsidian rejected the Local REST API key. "
                    "Point SOPHON_OBSIDIAN_VAULT at the vault folder, or refresh the API key in Obsidian settings."
                ) from exc
            return []
        except (OSError, ValueError):
            return []
        out: list[str] = []
        prefix = rel_dir.rstrip("/")
        for entry in entries:
            name = str(entry).strip().rstrip("/")
            if not name.endswith(".md"):
                continue
            if "/" in name:
                out.append(name if name.startswith(prefix) else prefix + "/" + name.split("/")[-1])
            else:
                out.append(prefix + "/" + name)
        return sorted(set(out))

    def _disk(self, rel: str) -> Path | None:
        if self.root is None:
            return None
        return self.root / rel

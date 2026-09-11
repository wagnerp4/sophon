from __future__ import annotations

from pathlib import Path

from backend.factory import BackendId

from utils.download.hf import (
    configure_hub_verbose,
    download_hf,
    download_hf_by_repo_only,
    download_preset_snapshot,
    hub_tqdm_bridge_factory,
    hub_token_for_snapshot,
    resolve_hf_hub_pull_paths,
    snapshot_hf_files,
)

__all__ = [
    "BackendId",
    "configure_hub_verbose",
    "download_hf",
    "download_hf_by_repo_only",
    "download_model",
    "download_preset_snapshot",
    "hub_tqdm_bridge_factory",
    "hub_token_for_snapshot",
    "resolve_hf_hub_pull_paths",
    "snapshot_hf_files",
]


def download_model(
    backend: BackendId,
    *,
    preset: str | None = None,
    repo_id: str | None = None,
    local_dir: Path | str | None = None,
    revision: str | None = None,
    cwd: Path | None = None,
    tqdm_class: type | None = None,
    verbose: bool = False,
) -> Path:
    if backend == "hf":
        if preset is None:
            if repo_id is None:
                raise ValueError("backend 'hf' requires preset=... or repo_id=...")
            return download_hf_by_repo_only(
                repo_id,
                local_dir=local_dir,
                revision=revision,
                cwd=cwd,
                tqdm_class=tqdm_class,
                verbose=verbose,
            )
        return download_hf(
            preset=preset,
            repo_id=repo_id,
            local_dir=local_dir,
            revision=revision,
            cwd=cwd,
            tqdm_class=tqdm_class,
            verbose=verbose,
        )
    if backend == "ollama":
        # TODO(backend-download): Implement scripted weights pull for Ollama (CLI or HTTP) once callers need it.
        raise NotImplementedError(
            "Model download for backend 'ollama' is not implemented yet.",
        )
    raise KeyError(f"unknown backend {backend!r}")

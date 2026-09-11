from __future__ import annotations

from typing import Literal

FinetuneBackendId = Literal["auto", "hf_peft", "unsloth"]

_BACKEND_IDS: tuple[FinetuneBackendId, ...] = ("auto", "hf_peft", "unsloth")


def finetune_backend_ids() -> tuple[FinetuneBackendId, ...]:
    return _BACKEND_IDS


def _unsloth_available() -> bool:
    try:
        import unsloth  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_finetune_backend_id(requested: str) -> str:
    token = (requested or "auto").strip().lower()
    if token not in _BACKEND_IDS:
        raise ValueError(f"unknown finetune backend {requested!r}; choose from {', '.join(_BACKEND_IDS)}")
    if token != "auto":
        return token
    if _unsloth_available():
        return "unsloth"
    return "hf_peft"


def load_finetune_backend(backend_id: str):
    resolved = resolve_finetune_backend_id(backend_id)
    if resolved == "unsloth":
        from training.finetune.backends.unsloth import UnslothFinetuneBackend

        return UnslothFinetuneBackend()
    from training.finetune.backends.hf_peft import HfPeftFinetuneBackend

    return HfPeftFinetuneBackend()

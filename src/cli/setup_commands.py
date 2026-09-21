from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Any

from cli.setup_recommend import (
    SCORE_RECIPE,
    SetupCombo,
    build_setup_report,
    combo_from_dict,
)


def _emit(msg: str) -> None:
    from cli.chat import _emit as emit

    emit(msg)


def _io():
    from cli.chat import _CURRENT_IO

    return _CURRENT_IO


def _pending_combo(state: Any) -> SetupCombo | None:
    row = getattr(state, "setup_pending", None)
    if not isinstance(row, dict) or not row:
        return None
    try:
        return combo_from_dict(row)
    except TypeError:
        return None


def _clear_pending(state: Any) -> None:
    state.setup_pending = None


def _store_report(state: Any, report) -> None:
    state.setup_pending = report.pending
    state.setup_last_table = report.table
    state.setup_last_recipe = report.recipe


def _prompt_lines() -> list[str]:
    return [
        "install rank 1, or skip and leave the session unchanged.",
        "1 /setup apply",
        "2 /setup skip",
    ]


def _ensure_pip_module(module: str, pip_name: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        pass
    _emit(f"installing {pip_name} into the current interpreter ({sys.executable}) ...")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        _emit(f"dep install failed: {exc}")
        return False
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = err[-1] if err else f"exit {proc.returncode}"
        _emit(f"dep install failed: {tail}")
        return False
    try:
        importlib.import_module(module)
    except ImportError:
        _emit(f"{module} still missing after pip into {sys.executable}")
        return False
    return True


def _install_implied_deps(combo: SetupCombo) -> bool:
    for dep in combo.missing_deps:
        if dep == "bitsandbytes":
            if not _ensure_pip_module("bitsandbytes", "bitsandbytes"):
                return False
            continue
        # TODO: map additional implied deps beyond bitsandbytes
        _emit(f"unknown implied dep {dep!r}, skipped")
    return True


def apply_setup_combo(state: Any, combo: SetupCombo) -> bool:
    from cli.chat import switch_session_backend, switch_session_model

    if combo.backend == "lmstudio" and not combo.lmstudio_listed:
        _emit(
            f"LM Studio GGUF {combo.model_id!r} is not listed. "
            f"Download it in LM Studio, then /model lmstudio:{combo.model_id}"
        )
        return False
    if not _install_implied_deps(combo):
        _emit("/setup apply aborted: missing interpreter deps")
        return False
    if combo.backend == "hf":
        from backend.hf.registry import HF_MODEL_PRESETS, preset_has_weights

        if combo.model_id in HF_MODEL_PRESETS and not preset_has_weights(
            combo.model_id, getattr(state, "project_root", None)
        ):
            from cli.chat import _cmd_model_download

            _cmd_model_download(state, combo.model_id)
            if not preset_has_weights(combo.model_id, getattr(state, "project_root", None)):
                _emit("/setup apply aborted: weights not on disk")
                return False
        same = (
            getattr(state, "backend_id", None) == "hf"
            and getattr(state, "preset_key", None) == combo.model_id
        )
        if same and str(getattr(state, "quantization", "none") or "none") != combo.cli_quant:
            state.model_path = ""
            state.preset_key = None
        state.quantization = combo.cli_quant
    if combo.backend != getattr(state, "backend_id", None):
        switch_session_backend(state, combo.backend)
    try:
        switch_session_model(state, combo.switch_target())
    except ValueError as exc:
        _emit(str(exc))
        return False
    state.params.max_new_tokens = int(combo.max_new_tokens)
    os.environ["SOPHON_CHAT_BACKEND"] = combo.backend
    # TODO: persist SOPHON_CHAT_BACKEND to .env after a successful /setup apply
    _emit(
        f"now: {combo.backend} · {combo.model_id} · {combo.quant} · "
        f"ctx {combo.ctx} · max_new_tokens {combo.max_new_tokens} · "
        f"tok/s {combo.tok_s:.0f} ({combo.tok_s_source})"
    )
    _clear_pending(state)
    return True


def _decide_after_table(state: Any) -> None:
    io = _io()
    chooser = getattr(io, "on_setup_choice", None)
    for line in _prompt_lines():
        _emit(line)
    if not callable(chooser):
        return
    combo = _pending_combo(state)
    summary = "rank 1"
    if combo is not None:
        summary = (
            f"rank 1: {combo.backend} {combo.model_id} {combo.quant} "
            f"ctx {combo.ctx} tok/s {combo.tok_s:.0f}"
        )
    try:
        choice = str(chooser(summary) or "skip").strip().lower()
    except Exception:
        choice = "skip"
    if choice in ("apply", "install", "1"):
        if combo is None:
            _emit("run /setup first")
            return
        apply_setup_combo(state, combo)
        return
    _clear_pending(state)
    _emit("/setup skip: session unchanged")


def handle_setup(state: Any, arg: str) -> bool:
    raw = (arg or "").strip()
    action = raw.split(None, 1)[0].lower() if raw else ""
    if action in ("skip", "keep", "exit"):
        if _pending_combo(state) is None:
            _emit("run /setup first")
            return False
        _clear_pending(state)
        _emit("/setup skip: session unchanged")
        return False
    if action == "apply":
        combo = _pending_combo(state)
        if combo is None:
            _emit("run /setup first")
            return False
        apply_setup_combo(state, combo)
        return False
    if action == "why":
        recipe = getattr(state, "setup_last_recipe", "") or SCORE_RECIPE
        table = getattr(state, "setup_last_table", "")
        _emit(recipe)
        if table:
            _emit(table)
        else:
            _emit("run /setup first")
        return False
    if action and action not in ("apply", "skip", "keep", "exit", "why"):
        _emit("usage: /setup | /setup apply | /setup skip | /setup why")
        return False

    report = build_setup_report(state)
    _store_report(state, report)
    _emit(report.table)
    if report.pending is None:
        _emit("no scored combination. session unchanged.")
        return False
    _decide_after_table(state)
    return False

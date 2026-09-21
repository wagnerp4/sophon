from __future__ import annotations

from training.orchestrator.config import load_trainer_config
from training.orchestrator.hf_lora import HfLoraDriver
from training.orchestrator.sheet import format_trainer_sheet
from training.orchestrator.spec import (
    DRIVER_HF_LORA,
    DRIVER_SSL4SED,
    DRIVERS,
    TrainerSpec,
    merge_select,
    spec_from_session,
    spec_to_session,
)
from training.orchestrator.ssl4sed import Ssl4sedDriver


def _emit(msg: str) -> None:
    from cli.chat import _emit as emit

    emit(msg)


def _driver(state: object, spec: TrainerSpec):
    if spec.driver == DRIVER_SSL4SED:
        return Ssl4sedDriver(state, load_trainer_config(getattr(state, "project_root", None)))
    return HfLoraDriver(state)


def _parse_fields(tokens: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"expected KEY=VAL, got {token!r}")
        key, value = token.split("=", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def handle_trainer(state: object, arg: str) -> bool:
    raw = (arg or "").strip()
    spec = spec_from_session(state)
    if not raw:
        _emit(format_trainer_sheet(state))
        return False
    parts = raw.split()
    action = parts[0].lower()
    rest = parts[1:]

    if action == "use":
        if not rest or rest[0] not in DRIVERS:
            _emit("usage: /trainer use hf-lora | ssl4sed")
            return False
        spec.driver = rest[0]
        spec_to_session(state, spec)
        _emit(f"trainer driver={spec.driver}")
        return False

    if action == "select":
        try:
            fields = _parse_fields(rest)
        except ValueError as exc:
            _emit(str(exc))
            _emit("usage: /trainer select task=… data=… model=… target=…")
            return False
        spec = merge_select(spec, fields)
        spec_to_session(state, spec)
        _emit(
            f"selected driver={spec.driver} task={spec.task or '-'} "
            f"data={spec.data or '-'} model={spec.model or '-'} target={spec.target or '-'}"
        )
        return False

    if action == "data":
        if rest and rest[0].lower() == "get":
            if len(rest) < 2:
                _emit("usage: /trainer data get KEY")
                return False
            key = rest[1]
            driver = _driver(state, spec)
            driver.fetch_data(key, _emit)
            return False
        driver = _driver(state, spec)
        rows = [row for row in driver.inventory() if row.kind == "data"]
        if not rows:
            _emit("no data rows")
            return False
        for row in rows:
            size = f" {row.size_hint}" if row.size_hint else ""
            _emit(f"  {row.key}  {row.status}{size}  {row.detail}")
        return False

    if action == "models":
        driver = _driver(state, spec)
        rows = [row for row in driver.inventory() if row.kind == "model"]
        if not rows:
            _emit("no model rows")
            return False
        for row in rows:
            _emit(f"  {row.key}  {row.status}  {row.detail}")
        return False

    if action == "run":
        full = False
        extras: list[str] = []
        for token in rest:
            lower = token.lower()
            if lower == "full":
                full = True
                continue
            if lower.startswith("target="):
                spec.target = token.split("=", 1)[1].strip()
                spec.driver = DRIVER_SSL4SED
                continue
            if "=" in token:
                extras.append(token)
                continue
            _emit(f"unrecognized run token {token!r}")
            return False
        spec.fast_dev = not full
        spec.overrides = tuple(extras)
        spec_to_session(state, spec)
        driver = _driver(state, spec)
        problems = driver.preflight(spec)
        for line in problems:
            _emit(line)
        blocking = [line for line in problems if "missing" in line or "ineligible" in line or "need ~" in line or "no model" in line or "unknown" in line]
        if spec.driver == DRIVER_HF_LORA and blocking:
            _emit("run aborted")
            return False
        if spec.driver == DRIVER_SSL4SED and any("need ~" in line for line in problems):
            _emit("run aborted (disk)")
            return False
        try:
            driver.run(spec, _emit)
        except Exception as exc:
            _emit(f"(trainer run failed: {exc})")
        return False

    if action == "status":
        driver = _driver(state, spec)
        for line in driver.status():
            _emit(line)
        return False

    if action == "watch":
        if spec.driver != DRIVER_SSL4SED:
            _emit("watch is for ssl4sed. /trainer use ssl4sed")
            return False
        driver = Ssl4sedDriver(state, load_trainer_config(getattr(state, "project_root", None)))
        for line in driver.watch_lines():
            _emit(line)
        return False

    if action == "stop":
        driver = _driver(state, spec)
        for line in driver.stop():
            _emit(line)
        return False

    _emit("usage: /trainer | use | select | data | data get KEY | models | run [full] | status | watch | stop")
    return False

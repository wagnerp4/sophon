from __future__ import annotations

import json
import shlex
import threading
from typing import Any

from training.orchestrator.config import TrainerConfig
from training.orchestrator.protocol import CatalogRow, LogFn
from training.orchestrator.spec import DRIVER_SSL4SED, TrainerSpec
from training.orchestrator.wsl import WslHost, stream_popen

_INVENTORY_PY = (
    "import json; "
    "from src.utils.data.download.registry import REGISTRY; "
    "print(json.dumps(["
    "{'key': s.key, 'task': s.task, 'status': s.status, 'size_gb': s.size_gb, "
    "'root': s.default_data_root, 'name': s.display_name} "
    "for s in REGISTRY.values()]))"
)


def _fmt_bytes(n: int | None) -> str:
    if n is None or n < 0:
        return "unknown"
    for label, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= div:
            return f"{n / div:.2f} {label}"
    return f"{n} B"


class Ssl4sedDriver:
    driver_id = DRIVER_SSL4SED

    def __init__(self, state: object, config: TrainerConfig) -> None:
        self.state = state
        self.config = config
        self.host = WslHost(config.ssl4sed_root, config.wsl_distro)

    def default_target(self) -> str:
        if self.config.default_ssl4sed_target:
            return self.config.default_ssl4sed_target
        cached = str(getattr(self.state, "ssl4sed_default_target", "") or "")
        if cached:
            return cached
        captured = self.host.run_capture(
            "grep -E '^default_target:' config/runs.yaml | head -1",
            timeout=10.0,
        )
        line = captured.text.strip().splitlines()[0] if captured.text.strip() else ""
        token = "mat-sed/passt"
        if ":" in line:
            token = line.split(":", 1)[1].strip().strip("\"'") or token
        self.state.ssl4sed_default_target = token
        return token

    def registry_rows(self) -> list[dict[str, Any]]:
        cached = getattr(self.state, "ssl4sed_registry", None)
        if isinstance(cached, list) and cached:
            return cached
        cmd = "PYTHONPATH=src .venv/bin/python -c " + shlex.quote(_INVENTORY_PY)
        captured = self.host.run_capture(cmd, timeout=25.0)
        if captured.code != 0 or not captured.text.strip():
            cmd = "uv run python -c " + shlex.quote(_INVENTORY_PY)
            captured = self.host.run_capture(cmd, timeout=50.0)
        text = (captured.stdout or captured.text).strip()
        if captured.code != 0 or not text:
            return []
        line = ""
        for candidate in reversed(text.splitlines()):
            token = candidate.strip()
            if token.startswith("[") or token.startswith("{"):
                line = token
                break
        if not line:
            return []
        try:
            rows = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(rows, list):
            return []
        self.state.ssl4sed_registry = rows
        return rows

    def _present_map(self, rels: list[str]) -> dict[str, bool]:
        paths = [rel for rel in rels if rel]
        if not paths:
            return {}
        quoted = " ".join(shlex.quote(rel) for rel in paths)
        py = (
            "import json,os,sys; "
            "print(json.dumps({p: os.path.exists(p) for p in sys.argv[1:]}))"
        )
        captured = self.host.run_capture("python3 -c " + shlex.quote(py) + " " + quoted, timeout=15.0)
        try:
            data = json.loads(captured.text.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): bool(v) for k, v in data.items()}

    def inventory(self) -> list[CatalogRow]:
        out: list[CatalogRow] = [
            CatalogRow("task", "sed", "ready", "sound event detection"),
            CatalogRow("task", "at", "ready", "audio tagging"),
            CatalogRow("task", "seld", "ready", "sound event localization and detection"),
        ]
        rows = self.registry_rows()
        present_map = self._present_map([str(raw.get("root") or "") for raw in rows[:8] if isinstance(raw, dict)])
        picked = 0
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            key = str(raw.get("key") or "")
            if not key:
                continue
            rel = str(raw.get("root") or "")
            present = bool(present_map.get(rel))
            size = raw.get("size_gb")
            size_hint = f"{float(size):.0f} GiB" if isinstance(size, (int, float)) else ""
            out.append(
                CatalogRow(
                    "data",
                    key,
                    "ready" if present else "missing",
                    str(raw.get("name") or raw.get("status") or ""),
                    size_hint,
                )
            )
            picked += 1
            if picked >= 8:
                break
        target = self.default_target()
        out.append(CatalogRow("model", target, "ready", "default train.sh target"))
        return out

    def preflight(self, spec: TrainerSpec) -> list[str]:
        lines: list[str] = []
        total, avail = self.host.disk_bytes()
        lines.append(
            f"ssl4sed disk total={_fmt_bytes(total)} free={_fmt_bytes(avail)} root={self.config.ssl4sed_root}"
        )
        target = spec.target or self.default_target()
        lines.append(f"target={target} fast_dev={spec.fast_dev}")
        if spec.data:
            rows = {str(row.get("key")): row for row in self.registry_rows() if isinstance(row, dict)}
            meta = rows.get(spec.data)
            if meta is None:
                lines.append(f"data key {spec.data!r} not in ssl4sed download list")
            else:
                rel = str(meta.get("root") or "")
                present = self.host.path_exists(rel) if rel else False
                if not present:
                    size = meta.get("size_gb")
                    need = float(size) * (1024**3) if isinstance(size, (int, float)) else 0.0
                    lines.append(f"data {spec.data} missing under {rel}")
                    if need and avail is not None and need > avail:
                        lines.append(
                            f"need ~{_fmt_bytes(int(need))} but free is {_fmt_bytes(avail)}"
                        )
        return lines

    def fetch_data(self, key: str, on_log: LogFn) -> None:
        rows = {str(row.get("key")): row for row in self.registry_rows() if isinstance(row, dict)}
        meta = rows.get(key)
        total, avail = self.host.disk_bytes()
        on_log(f"ssl4sed disk free={_fmt_bytes(avail)} / {_fmt_bytes(total)}")
        if meta is not None:
            size = meta.get("size_gb")
            if isinstance(size, (int, float)):
                need = int(float(size) * (1024**3))
                on_log(f"{key} size_hint={_fmt_bytes(need)}")
                if avail is not None and need > avail:
                    on_log(f"refuse: need {_fmt_bytes(need)} free={_fmt_bytes(avail)}")
                    return
            rel = str(meta.get("root") or "")
            if rel and self.host.path_exists(rel):
                on_log(f"{key} already present at {rel}")
                return
        quoted = shlex.quote(key)
        proc = self.host.popen(f"uv run ssl4sed download get {quoted}")
        self.state.ssl4sed_proc = proc
        try:
            code = stream_popen(proc, on_log)
        finally:
            self.state.ssl4sed_proc = None
        if code != 0:
            on_log(f"(ssl4sed download get {key} exit {code})")

    def run(self, spec: TrainerSpec, on_log: LogFn) -> None:
        if bool(getattr(self.state, "ssl4sed_running", False)):
            on_log("ssl4sed already running; /trainer status or /trainer stop")
            return
        target = spec.target or self.default_target()
        flags = "--fast-dev" if spec.fast_dev else ""
        cmd = f"./exps/train.sh --target {shlex.quote(target)} {flags}".strip()
        on_log(f"ssl4sed launch: {cmd}")
        thread = threading.Thread(
            target=self._run_thread,
            args=(cmd, on_log),
            name="trainer-ssl4sed",
            daemon=True,
        )
        self.state.ssl4sed_running = True
        self.state.ssl4sed_thread = thread
        thread.start()
        on_log("ssl4sed started in background. /trainer watch  /trainer stop")

    def _run_thread(self, cmd: str, on_log: LogFn) -> None:
        proc = None
        try:
            proc = self.host.popen(cmd)
            self.state.ssl4sed_proc = proc
            self._capture_run_paths()
            code = stream_popen(proc, on_log)
            on_log(f"(ssl4sed exit {code})")
        except Exception as exc:
            on_log(f"(ssl4sed failed: {exc})")
        finally:
            self.state.ssl4sed_running = False
            self.state.ssl4sed_proc = None

    def _capture_run_paths(self) -> None:
        captured = self.host.run_capture(
            "ls -1dt features/run_logs/* 2>/dev/null | head -1",
            timeout=10.0,
        )
        folder = captured.text.strip().splitlines()[0].strip() if captured.text.strip() else ""
        if folder:
            self.state.ssl4sed_run_dir = folder
            self.state.ssl4sed_run_meta = folder.rstrip("/") + "/run_meta.json"
            self.state.ssl4sed_run_log = folder.rstrip("/") + "/run.log"

    def status(self) -> list[str]:
        lines: list[str] = []
        running = bool(getattr(self.state, "ssl4sed_running", False))
        lines.append("ssl4sed running" if running else "ssl4sed idle")
        proc = getattr(self.state, "ssl4sed_proc", None)
        pid = getattr(proc, "pid", None) if proc is not None else None
        if pid:
            lines.append(f"pid={pid}")
        meta = str(getattr(self.state, "ssl4sed_run_meta", "") or "")
        log = str(getattr(self.state, "ssl4sed_run_log", "") or "")
        if meta:
            lines.append(f"run_meta={meta}")
        if log:
            lines.append(f"run.log={log}")
        lines.extend(self.watch_lines(limit=12))
        return lines

    def watch_lines(self, limit: int = 20) -> list[str]:
        lines: list[str] = []
        meta = str(getattr(self.state, "ssl4sed_run_meta", "") or "")
        log = str(getattr(self.state, "ssl4sed_run_log", "") or "")
        if not meta and not log:
            self._capture_run_paths()
            meta = str(getattr(self.state, "ssl4sed_run_meta", "") or "")
            log = str(getattr(self.state, "ssl4sed_run_log", "") or "")
        if meta:
            captured = self.host.run_capture(f"python3 -c {shlex.quote('import json,sys; p=sys.argv[1]; print(json.dumps(json.load(open(p)), indent=2)[:2000])')} {shlex.quote(meta)}", timeout=15.0)
            if captured.text.strip():
                lines.append("run_meta.json:")
                lines.extend(captured.text.strip().splitlines()[:40])
        if log:
            captured = self.host.run_capture(f"tail -n {int(limit)} {shlex.quote(log)}", timeout=10.0)
            if captured.text.strip():
                lines.append("run.log (tail):")
                lines.extend(captured.text.strip().splitlines()[-limit:])
        if not lines:
            lines.append("no run_meta.json / run.log yet")
        return lines

    def stop(self) -> list[str]:
        proc = getattr(self.state, "ssl4sed_proc", None)
        if proc is None:
            self.state.ssl4sed_running = False
            return ["no ssl4sed process"]
        try:
            proc.terminate()
        except Exception as exc:
            return [f"terminate failed: {exc}"]
        self.state.ssl4sed_running = False
        # TODO: map run_meta.json metrics onto loop/policies/goal.yaml and propose overlay.yaml under Review
        return ["sent terminate to ssl4sed process"]

from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass, field


def _safe_cuda() -> object | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch
    except Exception:
        return None
    return None


def _host_rss_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            handle = kernel32.GetCurrentProcess()
            counters = _PMC()
            counters.cb = ctypes.sizeof(_PMC)
            try:
                fn = kernel32.K32GetProcessMemoryInfo
            except AttributeError:
                fn = ctypes.windll.psapi.GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            if not fn(handle, ctypes.byref(counters), counters.cb):
                return None
            return int(counters.WorkingSetSize)
        except Exception:
            return None
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(usage)
        return int(usage) * 1024
    except Exception:
        return None


def _fmt_mib(n: int | None) -> str:
    if n is None:
        return "n/a"
    return f"{n / (1024 * 1024):.0f}MiB"


def _fmt_pct(num: int | None, den: int | None) -> str:
    if num is None or den is None or den <= 0:
        return ""
    return f" ({100.0 * num / den:.0f}%)"


@dataclass
class TurnStat:
    input_tokens: int
    new_tokens: int
    gen_time_s: float
    tok_s: float
    stop_reason: str


@dataclass
class SessionStats:
    max_position_embeddings: int | None = None
    ema_alpha: float = 0.3
    window_size: int = 5

    turns: int = 0
    total_new_tokens: int = 0
    total_gen_time_s: float = 0.0
    peak_tok_s: float = 0.0
    min_tok_s: float | None = None
    ema_tok_s: float | None = None
    first_turn_time_s: float | None = None
    last_turn: TurnStat | None = None
    exceptions: int = 0

    last_ctx_tokens: int = 0
    max_ctx_tokens_seen: int = 0

    last_vram_used_bytes: int | None = None
    peak_vram_used_bytes: int | None = None
    last_vram_free_bytes: int | None = None
    prev_vram_used_bytes: int | None = None

    last_host_rss_bytes: int | None = None

    _window: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    _ctx_warned: bool = False

    def __post_init__(self) -> None:
        if self.window_size != self._window.maxlen:
            self._window = deque(maxlen=self.window_size)

    def record_turn(
        self,
        *,
        input_tokens: int,
        new_tokens: int,
        gen_time_s: float,
        stop_reason: str,
    ) -> TurnStat:
        tok_s = (new_tokens / gen_time_s) if gen_time_s > 0 and new_tokens > 0 else 0.0
        turn = TurnStat(
            input_tokens=input_tokens,
            new_tokens=new_tokens,
            gen_time_s=gen_time_s,
            tok_s=tok_s,
            stop_reason=stop_reason,
        )
        self.turns += 1
        self.total_new_tokens += max(0, new_tokens)
        self.total_gen_time_s += max(0.0, gen_time_s)
        if self.first_turn_time_s is None:
            self.first_turn_time_s = gen_time_s
        if tok_s > 0:
            self.peak_tok_s = max(self.peak_tok_s, tok_s)
            self.min_tok_s = tok_s if self.min_tok_s is None else min(self.min_tok_s, tok_s)
            if self.ema_tok_s is None:
                self.ema_tok_s = tok_s
            else:
                self.ema_tok_s = self.ema_alpha * tok_s + (1.0 - self.ema_alpha) * self.ema_tok_s
            self._window.append(tok_s)
        self.last_turn = turn
        self.last_ctx_tokens = input_tokens + new_tokens
        self.max_ctx_tokens_seen = max(self.max_ctx_tokens_seen, self.last_ctx_tokens)
        self._sample_memory()
        return turn

    def record_exception(self) -> None:
        self.exceptions += 1

    def reset(self) -> None:
        self.turns = 0
        self.total_new_tokens = 0
        self.total_gen_time_s = 0.0
        self.peak_tok_s = 0.0
        self.min_tok_s = None
        self.ema_tok_s = None
        self.first_turn_time_s = None
        self.last_turn = None
        self.exceptions = 0
        self.last_ctx_tokens = 0
        self.max_ctx_tokens_seen = 0
        self.last_vram_used_bytes = None
        self.peak_vram_used_bytes = None
        self.last_vram_free_bytes = None
        self.prev_vram_used_bytes = None
        self.last_host_rss_bytes = None
        self._window.clear()
        self._ctx_warned = False
        torch = _safe_cuda()
        if torch is not None:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def _sample_memory(self) -> None:
        torch = _safe_cuda()
        if torch is not None:
            try:
                self.prev_vram_used_bytes = self.last_vram_used_bytes
                self.last_vram_used_bytes = int(torch.cuda.memory_allocated())
                self.peak_vram_used_bytes = int(torch.cuda.max_memory_allocated())
                free_b, _total_b = torch.cuda.mem_get_info()
                self.last_vram_free_bytes = int(free_b)
            except Exception:
                pass
        self.last_host_rss_bytes = _host_rss_bytes()

    def mean_tok_s(self) -> float:
        if self.total_gen_time_s <= 0:
            return 0.0
        return self.total_new_tokens / self.total_gen_time_s

    def window_tok_s(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def context_fill_ratio(self) -> float | None:
        if not self.max_position_embeddings or self.max_position_embeddings <= 0:
            return None
        return self.last_ctx_tokens / self.max_position_embeddings

    def warn_if_context_high(self, threshold: float = 0.9) -> str | None:
        ratio = self.context_fill_ratio()
        if ratio is None or ratio < threshold:
            self._ctx_warned = False
            return None
        if self._ctx_warned:
            return None
        self._ctx_warned = True
        return (
            f"warning: context at {ratio * 100:.0f}% of "
            f"{self.max_position_embeddings} tokens. Consider /reset or /pop."
        )

    def format_footer(self) -> str:
        if self.last_turn is None:
            return "[dbg] no turns yet"
        t = self.last_turn
        parts = [
            f"turn={self.turns}",
            f"in={t.input_tokens}",
            f"out={t.new_tokens}",
            f"t={t.gen_time_s:.2f}s",
            f"tok/s={t.tok_s:.1f}",
        ]
        if self.ema_tok_s is not None:
            parts.append(f"ema={self.ema_tok_s:.1f}")
        ctx_str = f"ctx={self.last_ctx_tokens}"
        if self.max_position_embeddings:
            ctx_str += f"/{self.max_position_embeddings}"
        parts.append(ctx_str)
        if self.last_vram_used_bytes is not None:
            parts.append(f"vram={_fmt_mib(self.last_vram_used_bytes)}")
        if self.peak_vram_used_bytes is not None:
            parts.append(f"peak={_fmt_mib(self.peak_vram_used_bytes)}")
        parts.append(f"stop={t.stop_reason}")
        return "[dbg] " + " ".join(parts)

    def format_table(self) -> str:
        lines: list[str] = []
        lines.append(f"session : turns={self.turns} exceptions={self.exceptions}")
        lines.append(
            "        : "
            f"new_tokens={self.total_new_tokens} gen_time={self.total_gen_time_s:.2f}s "
            f"mean_tok_s={self.mean_tok_s():.2f} "
            f"ema={(self.ema_tok_s or 0.0):.2f} "
            f"window{len(self._window)}={self.window_tok_s():.2f}"
        )
        peak = f"{self.peak_tok_s:.2f}" if self.peak_tok_s > 0 else "n/a"
        mn = f"{self.min_tok_s:.2f}" if self.min_tok_s is not None else "n/a"
        ft = f"{self.first_turn_time_s:.2f}s" if self.first_turn_time_s is not None else "n/a"
        lines.append(f"        : peak_tok_s={peak} min_tok_s={mn} first_turn={ft}")
        ctx_pct = _fmt_pct(self.last_ctx_tokens, self.max_position_embeddings)
        max_pos = self.max_position_embeddings if self.max_position_embeddings else "unknown"
        lines.append(
            "context : "
            f"last={self.last_ctx_tokens}{ctx_pct} max_seen={self.max_ctx_tokens_seen} window={max_pos}"
        )
        if self.last_vram_used_bytes is not None or self.peak_vram_used_bytes is not None:
            delta = ""
            if self.prev_vram_used_bytes is not None and self.last_vram_used_bytes is not None:
                d = self.last_vram_used_bytes - self.prev_vram_used_bytes
                sign = "+" if d >= 0 else ""
                delta = f" delta={sign}{d / (1024 * 1024):.0f}MiB"
            lines.append(
                "memory  : "
                f"vram_used={_fmt_mib(self.last_vram_used_bytes)} "
                f"peak={_fmt_mib(self.peak_vram_used_bytes)} "
                f"free={_fmt_mib(self.last_vram_free_bytes)}{delta} "
                f"host_rss={_fmt_mib(self.last_host_rss_bytes)}"
            )
        else:
            lines.append(f"memory  : host_rss={_fmt_mib(self.last_host_rss_bytes)} (no cuda)")
        if self.last_turn is not None:
            t = self.last_turn
            lines.append(
                "last    : "
                f"in={t.input_tokens} out={t.new_tokens} t={t.gen_time_s:.2f}s "
                f"tok/s={t.tok_s:.1f} stop={t.stop_reason}"
            )
        return "\n".join(lines)


def initial_debug_mode_from_env() -> bool:
    raw = os.environ.get("LMWRAP_CHAT_DEBUG", "").strip().lower()
    return raw in ("1", "true", "yes", "on")

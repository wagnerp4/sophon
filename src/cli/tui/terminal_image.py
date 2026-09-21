from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Iterator

_probed = False
_graphics = False
_label = "halfblocks"
_widget_cls: Any = None


def _skip_csi_autodetect() -> bool:
    explicit = os.environ.get("SOPHON_TERMINAL_GRAPHICS", "").strip().lower()
    if explicit in ("0", "off", "none", "halfblocks", "unicode"):
        return True
    if os.environ.get("SOPHON_SKIP_TERMINAL_GRAPHICS", "").strip().lower() in ("1", "true", "yes"):
        return True
    return sys.platform == "win32"


class _NonTtyStdout:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def isatty(self) -> bool:
        return False

    def write(self, data: Any) -> Any:
        if self._inner is None:
            return 0
        return self._inner.write(data)

    def flush(self) -> None:
        if self._inner is not None:
            self._inner.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@contextmanager
def _without_csi_queries() -> Iterator[None]:
    stdout = sys.__stdout__
    stdin = sys.__stdin__
    sys.__stdout__ = _NonTtyStdout(stdout)
    if stdin is not None:
        sys.__stdin__ = _NonTtyStdout(stdin)
    try:
        yield
    finally:
        sys.__stdout__ = stdout
        sys.__stdin__ = stdin


def probe_terminal_graphics() -> None:
    global _probed, _graphics, _label, _widget_cls
    if _probed:
        return
    _probed = True
    try:
        if _skip_csi_autodetect():
            with _without_csi_queries():
                from textual_image.widget import Image as ImageWidget
        else:
            from textual_image.widget import Image as ImageWidget
            from textual_image.renderable import Image as AutoRenderable
            from textual_image.renderable.halfcell import Image as HalfcellRenderable
            from textual_image.renderable.sixel import Image as SixelRenderable
            from textual_image.renderable.tgp import Image as TGPRenderable
    except ImportError:
        return
    _widget_cls = ImageWidget
    if _skip_csi_autodetect():
        _graphics = False
        _label = "halfblocks"
        return
    if AutoRenderable is SixelRenderable:
        _graphics = True
        _label = "sixel"
        return
    if AutoRenderable is TGPRenderable:
        _graphics = True
        _label = "kitty"
        return
    if AutoRenderable is HalfcellRenderable:
        _label = "halfblocks"
        return
    _label = "unicode"


def graphics_enabled() -> bool:
    probe_terminal_graphics()
    return _graphics


def graphics_label() -> str:
    probe_terminal_graphics()
    return _label


def image_widget_class() -> Any:
    probe_terminal_graphics()
    return _widget_cls


def cell_pixel_size() -> tuple[int, int]:
    probe_terminal_graphics()
    if _skip_csi_autodetect():
        return 10, 20
    try:
        from textual_image._terminal import get_cell_size

        size = get_cell_size()
        return max(1, int(size.width)), max(1, int(size.height))
    except Exception:
        return 8, 16

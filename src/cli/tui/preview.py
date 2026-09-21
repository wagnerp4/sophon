from __future__ import annotations

import io
import math
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from cli.tui.terminal_image import (
    cell_pixel_size,
    graphics_enabled,
    graphics_label,
    image_widget_class,
)

_PATH_TOKEN_RE = re.compile(
    r"([MmLlHhVvAaZz])|([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)
_COLOR_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_BG = (18, 20, 24)
_MESH_BASE = (188, 196, 208)
_LIGHT = np.array([-0.32, 0.55, 0.77], dtype=np.float64)
_ORBIT_STEP = 0.14
_MAX_IMAGE_SIDE = 2048
# TODO: mesh decimation for large STL files
# TODO: cubic/quadratic SVG commands (C/S/Q/T)
# TODO: PDF two-page spread / zoom
# TODO: notebook cell outputs via the same sixel/kitty image widget


@dataclass
class Mesh:
    vertices: np.ndarray
    normals: np.ndarray


def image_to_halfblocks(image: Image.Image) -> Text:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width, _ = pixels.shape
    rows = height // 2
    out = Text()
    for row in range(rows):
        if row:
            out.append("\n")
        top = pixels[row * 2]
        bottom = pixels[row * 2 + 1]
        for col in range(width):
            t = top[col]
            b = bottom[col]
            out.append(
                "▀",
                style=f"#{t[0]:02x}{t[1]:02x}{t[2]:02x} on #{b[0]:02x}{b[1]:02x}{b[2]:02x}",
            )
    return out


def parse_stl_bytes(data: bytes) -> Mesh:
    if _looks_ascii_stl(data):
        try:
            return parse_stl_text(data.decode("utf-8"))
        except ValueError:
            pass
    return _parse_binary_stl(data)


def parse_stl_text(text: str) -> Mesh:
    verts: list[list[tuple[float, float, float]]] = []
    norms: list[tuple[float, float, float]] = []
    current: list[tuple[float, float, float]] = []
    normal = (0.0, 0.0, 1.0)
    for raw in text.splitlines():
        parts = raw.strip().split()
        if not parts:
            continue
        key = parts[0].lower()
        if key == "facet" and len(parts) >= 5 and parts[1].lower() == "normal":
            normal = (float(parts[2]), float(parts[3]), float(parts[4]))
            current = []
        elif key == "vertex" and len(parts) >= 4:
            current.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif key == "endfacet":
            if len(current) == 3:
                verts.append(current)
                norms.append(normal)
            current = []
    if not verts:
        raise ValueError("No triangular facets in STL text.")
    vertices = np.asarray(verts, dtype=np.float64)
    normals = np.asarray(norms, dtype=np.float64)
    return Mesh(vertices=vertices, normals=_ensure_normals(vertices, normals))


def render_mesh(
    mesh: Mesh,
    width: int,
    height: int,
    yaw: float,
    pitch: float,
    roll: float,
) -> Image.Image:
    width = max(2, int(width))
    height = max(2, int(height))
    rot = _rotation_matrix(yaw, pitch, roll)
    centered = mesh.vertices - mesh.vertices.reshape(-1, 3).mean(axis=0)
    extent = float(np.abs(centered).max()) or 1.0
    scaled = centered / extent
    world = scaled @ rot.T
    normals = mesh.normals @ rot.T
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.clip(lengths, 1e-8, None)
    normals = normals / lengths
    facing = normals[:, 2] > 0.02
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:] = _BG
    zbuf = np.full((height, width), -1.0e9, dtype=np.float64)
    margin = 0.12
    span = 2.0 * (1.0 + margin)
    scale = min(width, height) / span
    ox = (width - 1) * 0.5
    oy = (height - 1) * 0.5
    for i, visible in enumerate(facing):
        if not visible:
            continue
        pts = world[i]
        xs = ox + pts[:, 0] * scale
        ys = oy - pts[:, 1] * scale
        zs = pts[:, 2]
        shade = float(np.clip(0.22 + 0.78 * max(0.0, float(normals[i] @ _LIGHT)), 0.08, 1.0))
        color = tuple(int(c * shade) for c in _MESH_BASE)
        _fill_triangle(rgb, zbuf, xs, ys, zs, color)
        edge = tuple(min(255, int(c * 1.18 + 18)) for c in color)
        _stroke_triangle(rgb, zbuf, xs, ys, zs, edge)
    return Image.fromarray(rgb, mode="RGB")


def render_svg(markup: str, width: int, height: int) -> Image.Image:
    width = max(2, int(width))
    height = max(2, int(height))
    root = ET.fromstring(markup)
    vb = _svg_viewbox(root)
    img = Image.new("RGB", (width, height), _BG)
    sx = (width * 0.84) / vb[2]
    sy = (height * 0.84) / vb[3]
    scale = min(sx, sy)
    ox = (width - vb[2] * scale) * 0.5 - vb[0] * scale
    oy = (height - vb[3] * scale) * 0.5 - vb[1] * scale

    def tx(x: float, y: float) -> tuple[int, int]:
        return int(round(ox + x * scale)), int(round(oy + y * scale))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(img, "RGBA")
    for node in root.iter():
        tag = node.tag.split("}", 1)[-1]
        if tag != "path":
            continue
        d = node.attrib.get("d", "")
        fill = _parse_color(node.attrib.get("fill", ""), (0, 122, 204))
        for poly in _path_polygons(d):
            if len(poly) < 3:
                continue
            pts = [tx(x, y) for x, y in poly]
            draw.polygon(pts, fill=(*fill, 255))
    return img


def _looks_ascii_stl(data: bytes) -> bool:
    if len(data) < 15:
        return False
    head = data[:80].lstrip().lower()
    if not head.startswith(b"solid"):
        return False
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + count * 50
        if expected == len(data) and b"facet" not in data[:512].lower():
            return False
    return b"facet" in data.lower() or b"vertex" in data.lower()


def _parse_binary_stl(data: bytes) -> Mesh:
    if len(data) < 84:
        raise ValueError("STL file is truncated.")
    count = struct.unpack_from("<I", data, 80)[0]
    need = 84 + count * 50
    if len(data) < need:
        raise ValueError("Binary STL facet count does not match file size.")
    verts = np.empty((count, 3, 3), dtype=np.float64)
    norms = np.empty((count, 3), dtype=np.float64)
    offset = 84
    try:
        for i in range(count):
            chunk = struct.unpack_from("<12fH", data, offset)
            norms[i] = chunk[0:3]
            verts[i, 0] = chunk[3:6]
            verts[i, 1] = chunk[6:9]
            verts[i, 2] = chunk[9:12]
            offset += 50
    except struct.error as exc:
        raise ValueError(f"Binary STL is corrupt: {exc}") from exc
    if count == 0:
        raise ValueError("Binary STL has no facets.")
    return Mesh(vertices=verts, normals=_ensure_normals(verts, norms))


def _ensure_normals(vertices: np.ndarray, normals: np.ndarray) -> np.ndarray:
    e1 = vertices[:, 1] - vertices[:, 0]
    e2 = vertices[:, 2] - vertices[:, 0]
    computed = np.cross(e1, e2)
    missing = np.linalg.norm(normals, axis=1) < 1e-8
    if missing.any():
        normals = normals.copy()
        normals[missing] = computed[missing]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.clip(lengths, 1e-8, None)
    return normals / lengths


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def _fill_triangle(
    rgb: np.ndarray,
    zbuf: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    height, width, _ = rgb.shape
    min_x = max(0, int(math.floor(float(xs.min()))))
    max_x = min(width - 1, int(math.ceil(float(xs.max()))))
    min_y = max(0, int(math.floor(float(ys.min()))))
    max_y = min(height - 1, int(math.ceil(float(ys.max()))))
    if min_x > max_x or min_y > max_y:
        return
    x0, x1, x2 = (float(xs[0]), float(xs[1]), float(xs[2]))
    y0, y1, y2 = (float(ys[0]), float(ys[1]), float(ys[2]))
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-10:
        return
    z0, z1, z2 = (float(zs[0]), float(zs[1]), float(zs[2]))
    shade = np.array(color, dtype=np.uint8)
    for y in range(min_y, max_y + 1):
        py = y + 0.5
        for x in range(min_x, max_x + 1):
            px = x + 0.5
            w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
            w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
            w2 = 1.0 - w0 - w1
            if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                continue
            z = w0 * z0 + w1 * z1 + w2 * z2
            if z >= zbuf[y, x]:
                zbuf[y, x] = z
                rgb[y, x] = shade


def _stroke_triangle(
    rgb: np.ndarray,
    zbuf: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    pairs = ((0, 1), (1, 2), (2, 0))
    for a, b in pairs:
        _draw_line(
            rgb,
            zbuf,
            float(xs[a]),
            float(ys[a]),
            float(zs[a]),
            float(xs[b]),
            float(ys[b]),
            float(zs[b]),
            color,
        )


def _draw_line(
    rgb: np.ndarray,
    zbuf: np.ndarray,
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    color: tuple[int, int, int],
) -> None:
    height, width, _ = rgb.shape
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    shade = np.array(color, dtype=np.uint8)
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        z = z0 + (z1 - z0) * t + 0.002
        if z >= zbuf[y, x]:
            zbuf[y, x] = z
            rgb[y, x] = shade


def _svg_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    raw = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if raw:
        parts = [float(p) for p in re.split(r"[,\s]+", raw.strip()) if p]
        if len(parts) == 4:
            return parts[0], parts[1], parts[2], parts[3]
    width = float(re.sub(r"[^0-9.+-]", "", root.attrib.get("width", "100")) or 100)
    height = float(re.sub(r"[^0-9.+-]", "", root.attrib.get("height", "100")) or 100)
    return 0.0, 0.0, width, height


def _parse_color(value: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    text = value.strip()
    match = _COLOR_RE.match(text)
    if match is None:
        return default
    hex_value = match.group(1)
    if len(hex_value) == 3:
        hex_value = "".join(ch * 2 for ch in hex_value)
    return int(hex_value[0:2], 16), int(hex_value[2:4], 16), int(hex_value[4:6], 16)


def _path_tokens(d: str) -> Iterable[str | float]:
    for match in _PATH_TOKEN_RE.finditer(d):
        cmd, number = match.groups()
        if cmd is not None:
            yield cmd
        else:
            yield float(number)


def _path_polygons(d: str) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    cx = cy = 0.0
    start = (0.0, 0.0)
    command = "L"
    args: list[float] = []

    def flush() -> None:
        nonlocal current
        if len(current) >= 3:
            polygons.append(current)
        current = []

    def take(count: int) -> list[float] | None:
        if len(args) < count:
            return None
        values = args[:count]
        del args[:count]
        return values

    for token in _path_tokens(d):
        if isinstance(token, str):
            command = token
            args = []
            if command in "Zz":
                current.append(start)
                cx, cy = start
                flush()
            continue
        args.append(token)
        if command in "Mm" and (values := take(2)) is not None:
            x, y = values
            if command == "m":
                x += cx
                y += cy
            if current:
                flush()
            cx, cy = x, y
            start = (cx, cy)
            current = [(cx, cy)]
            command = "l" if command == "m" else "L"
        elif command in "Ll" and (values := take(2)) is not None:
            x, y = values
            if command == "l":
                x += cx
                y += cy
            cx, cy = x, y
            current.append((cx, cy))
        elif command in "Hh" and (values := take(1)) is not None:
            x = values[0]
            if command == "h":
                x += cx
            cx = x
            current.append((cx, cy))
        elif command in "Vv" and (values := take(1)) is not None:
            y = values[0]
            if command == "v":
                y += cy
            cy = y
            current.append((cx, cy))
        elif command in "Aa" and (values := take(7)) is not None:
            rx, ry, phi, large, sweep, x, y = values
            if command == "a":
                x += cx
                y += cy
            current.extend(_arc_points(cx, cy, rx, ry, phi, large, sweep, x, y))
            cx, cy = x, y
    flush()
    return polygons


def _arc_points(
    x1: float,
    y1: float,
    rx: float,
    ry: float,
    phi_deg: float,
    large: float,
    sweep: float,
    x2: float,
    y2: float,
) -> list[tuple[float, float]]:
    rx = abs(rx)
    ry = abs(ry)
    if rx < 1e-8 or ry < 1e-8:
        return [(x2, y2)]
    phi = math.radians(phi_deg)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    dx = (x1 - x2) / 2.0
    dy = (y1 - y2) / 2.0
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:
        scale = math.sqrt(lam)
        rx *= scale
        ry *= scale
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den)) if den else 0.0
    if int(large) == int(sweep):
        coef = -coef
    cxp = coef * (rx * y1p) / ry
    cyp = coef * -(ry * x1p) / rx
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

    def _angle(ux: float, uy: float, vx: float, vy: float) -> float:
        n = math.hypot(ux, uy) * math.hypot(vx, vy)
        if n == 0.0:
            return 0.0
        c = max(-1.0, min(1.0, (ux * vx + uy * vy) / n))
        ang = math.acos(c)
        if ux * vy - uy * vx < 0.0:
            ang = -ang
        return ang

    theta1 = _angle(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = _angle(
        (x1p - cxp) / rx,
        (y1p - cyp) / ry,
        (-x1p - cxp) / rx,
        (-y1p - cyp) / ry,
    )
    if sweep == 0 and dtheta > 0:
        dtheta -= 2.0 * math.pi
    elif sweep != 0 and dtheta < 0:
        dtheta += 2.0 * math.pi
    steps = max(8, int(abs(dtheta) / (math.pi / 32)))
    points: list[tuple[float, float]] = []
    for i in range(1, steps + 1):
        t = theta1 + dtheta * i / steps
        x = cx + rx * math.cos(t) * cos_phi - ry * math.sin(t) * sin_phi
        y = cy + rx * math.cos(t) * sin_phi + ry * math.sin(t) * cos_phi
        points.append((x, y))
    return points


class RasterPreview(Widget, can_focus=True, can_focus_children=False):
    DEFAULT_CSS = """
    RasterPreview {
        overflow: hidden;
        height: 1fr;
        width: 1fr;
        background: $background;
    }
    RasterPreview #raster-static {
        width: 1fr;
        height: 1fr;
    }
    RasterPreview #raster-image {
        width: 1fr;
        height: 1fr;
    }
    """
    BINDINGS = [
        Binding("left", "yaw_left", show=False),
        Binding("right", "yaw_right", show=False),
        Binding("up", "pitch_up", show=False),
        Binding("down", "pitch_down", show=False),
        Binding("pageup", "page_prev", show=False),
        Binding("pagedown", "page_next", show=False),
        Binding("end", "page_last", show=False),
        Binding("q", "roll_left", show=False),
        Binding("e", "roll_right", show=False),
        Binding("r", "reset_orbit", "Reset", show=True),
        Binding("home", "reset_orbit", show=False),
    ]

    class PdfPageChanged(Message):
        def __init__(self, index: int, count: int) -> None:
            self.index = index
            self.count = count
            super().__init__()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._kind: str | None = None
        self._mesh: Mesh | None = None
        self._svg_text: str | None = None
        self._image: Image.Image | None = None
        self._pdf_doc = None
        self._pdf_page = 0
        self._pdf_view = "text"
        self._error: str | None = None
        self._yaw = 0.65
        self._pitch = 0.45
        self._roll = 0.0
        self._drag: tuple[int, int] | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="raster-static")
        widget_cls = image_widget_class()
        if widget_cls is not None:
            image = widget_cls(id="raster-image")
            image.display = False
            yield image

    @property
    def can_orbit(self) -> bool:
        return self._kind == "mesh" and self._mesh is not None

    @property
    def can_page(self) -> bool:
        return self._kind == "pdf" and self._pdf_doc is not None

    @property
    def pdf_view(self) -> str:
        return self._pdf_view

    def _close_pdf(self) -> None:
        doc = self._pdf_doc
        self._pdf_doc = None
        self._pdf_page = 0
        if doc is None:
            return
        try:
            doc.close()
        except Exception:
            pass

    def on_unmount(self) -> None:
        self._close_pdf()

    def clear_preview(self) -> None:
        self._kind = None
        self._mesh = None
        self._svg_text = None
        self._image = None
        self._close_pdf()
        self._error = None
        self._drag = None
        self._set_fallback("")

    def show_mesh(self, mesh: Mesh, *, reset_orbit: bool = True) -> None:
        self._close_pdf()
        self._kind = "mesh"
        self._mesh = mesh
        self._svg_text = None
        self._image = None
        self._error = None
        if reset_orbit:
            self._yaw = 0.65
            self._pitch = 0.45
            self._roll = 0.0
        self._paint()

    def show_svg(self, markup: str) -> None:
        self._close_pdf()
        self._kind = "svg"
        self._svg_text = markup
        self._mesh = None
        self._image = None
        self._error = None
        self._paint()

    def show_image(self, data: bytes) -> None:
        self._close_pdf()
        self._mesh = None
        self._svg_text = None
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
            self._image = image.convert("RGB")
        except Exception as exc:
            self.show_error(f"Image preview failed: {exc}")
            return
        self._kind = "image"
        self._error = None
        self._paint()

    def show_pdf(self, data: bytes, *, reset_page: bool = True) -> None:
        from cli.tui.pdf_view import open_pdf

        self._mesh = None
        self._svg_text = None
        self._image = None
        keep_page = 0 if reset_page else self._pdf_page
        self._close_pdf()
        try:
            self._pdf_doc = open_pdf(data)
        except ValueError as exc:
            self.show_error(str(exc))
            return
        self._kind = "pdf"
        self._error = None
        last = int(self._pdf_doc.page_count) - 1
        self._pdf_page = max(0, min(last, keep_page))
        self._paint()
        self._emit_pdf_page()

    def set_pdf_view(self, mode: str) -> None:
        if mode not in {"text", "raster"}:
            return
        if mode == self._pdf_view and self._error is None:
            return
        self._pdf_view = mode
        self._paint()
        self._emit_pdf_page()

    def page_source_text(self) -> str:
        from cli.tui.pdf_view import pdf_page_source

        if self._pdf_doc is None:
            return ""
        return pdf_page_source(self._pdf_doc, self._pdf_page)

    def show_error(self, message: str) -> None:
        self._kind = None
        self._mesh = None
        self._svg_text = None
        self._image = None
        self._close_pdf()
        self._error = message
        self._set_fallback(message)

    def on_resize(self, event: events.Resize) -> None:
        self._paint()

    def on_click(self, event: events.Click) -> None:
        self.focus()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self.focus()
        if not self.can_orbit:
            return
        self._drag = (event.x, event.y)
        self.capture_mouse()
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if not self.can_page or self._pdf_view == "text":
            return
        self._pdf_delta(-1)
        event.stop()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if not self.can_page or self._pdf_view == "text":
            return
        self._pdf_delta(1)
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        self.capture_mouse(False)
        self._drag = None

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._drag is None or not self.can_orbit:
            return
        dx = event.x - self._drag[0]
        dy = event.y - self._drag[1]
        self._drag = (event.x, event.y)
        self._yaw += dx * 0.06
        self._pitch = float(np.clip(self._pitch + dy * 0.06, -1.2, 1.2))
        self._paint()
        event.stop()

    def action_yaw_left(self) -> None:
        if self.can_page:
            self._pdf_delta(-1)
            return
        self._nudge(-_ORBIT_STEP, 0.0, 0.0)

    def action_yaw_right(self) -> None:
        if self.can_page:
            self._pdf_delta(1)
            return
        self._nudge(_ORBIT_STEP, 0.0, 0.0)

    def action_pitch_up(self) -> None:
        if self.can_page:
            self._pdf_delta(-1)
            return
        self._nudge(0.0, -_ORBIT_STEP, 0.0)

    def action_pitch_down(self) -> None:
        if self.can_page:
            self._pdf_delta(1)
            return
        self._nudge(0.0, _ORBIT_STEP, 0.0)

    def action_page_prev(self) -> None:
        self._pdf_delta(-1)

    def action_page_next(self) -> None:
        self._pdf_delta(1)

    def action_page_last(self) -> None:
        if not self.can_page:
            return
        last = int(self._pdf_doc.page_count) - 1
        self._pdf_set_page(last)

    def action_roll_left(self) -> None:
        self._nudge(0.0, 0.0, -_ORBIT_STEP)

    def action_roll_right(self) -> None:
        self._nudge(0.0, 0.0, _ORBIT_STEP)

    def action_reset_orbit(self) -> None:
        if self.can_page:
            self._pdf_set_page(0)
            return
        if not self.can_orbit:
            return
        self._yaw = 0.65
        self._pitch = 0.45
        self._roll = 0.0
        self._paint()

    def _nudge(self, yaw: float, pitch: float, roll: float) -> None:
        if not self.can_orbit:
            return
        self._yaw += yaw
        self._pitch = float(np.clip(self._pitch + pitch, -1.2, 1.2))
        self._roll += roll
        self._paint()

    def _pdf_delta(self, step: int) -> None:
        if not self.can_page:
            return
        self._pdf_set_page(self._pdf_page + int(step))

    def _pdf_set_page(self, index: int) -> None:
        if not self.can_page:
            return
        last = int(self._pdf_doc.page_count) - 1
        next_index = max(0, min(last, int(index)))
        if next_index == self._pdf_page and self._error is None:
            return
        self._pdf_page = next_index
        self._paint()
        self._emit_pdf_page()

    def _emit_pdf_page(self) -> None:
        if self._pdf_doc is None:
            return
        self.post_message(self.PdfPageChanged(self._pdf_page, int(self._pdf_doc.page_count)))

    def _image_widget(self):
        try:
            return self.query_one("#raster-image")
        except NoMatches:
            return None

    def _fallback(self) -> Static:
        return self.query_one("#raster-static", Static)

    def _set_fallback(self, content) -> None:
        widget = self._image_widget()
        if widget is not None:
            widget.image = None
            widget.display = False
        static = self._fallback()
        static.display = True
        static.update(content)
        if self._kind == "pdf" and self._pdf_view == "text":
            self.styles.overflow_y = "auto"
            static.styles.height = "auto"
            static.styles.padding = (0, 1)
            return
        self.styles.overflow_y = "hidden"
        static.styles.height = "1fr"
        static.styles.padding = 0

    def _show_graphics(self, image: Image.Image) -> None:
        widget = self._image_widget()
        if widget is None:
            self._set_fallback(image_to_halfblocks(image))
            return
        static = self._fallback()
        static.display = False
        widget.display = True
        widget.image = image
        self.styles.overflow_y = "hidden"

    def _wants_graphics(self) -> bool:
        if not graphics_enabled():
            return False
        if self._kind == "image":
            return True
        if self._kind == "svg":
            return True
        return self._kind == "pdf" and self._pdf_view == "raster"

    def _target_pixels(self) -> tuple[int, int]:
        size = self.size
        cells_w = max(2, size.width)
        cells_h = max(2, size.height)
        if self._wants_graphics():
            cell_w, cell_h = cell_pixel_size()
            px_w = min(_MAX_IMAGE_SIDE, cells_w * cell_w)
            px_h = min(_MAX_IMAGE_SIDE, cells_h * cell_h)
            return max(2, px_w), max(2, px_h)
        return cells_w, cells_h * 2

    def _paint(self) -> None:
        if not self.is_mounted:
            return
        if self._error:
            self._set_fallback(self._error)
            return
        px_w, px_h = self._target_pixels()
        try:
            if self._kind == "mesh" and self._mesh is not None:
                image = render_mesh(self._mesh, px_w, px_h, self._yaw, self._pitch, self._roll)
                self._set_fallback(image_to_halfblocks(image))
                return
            if self._kind == "svg" and self._svg_text is not None:
                image = render_svg(self._svg_text, px_w, px_h)
            elif self._kind == "image" and self._image is not None:
                image = self._fit_still(self._image, px_w, px_h)
            elif self._kind == "pdf" and self._pdf_doc is not None:
                from cli.tui.pdf_view import pdf_page_body, render_pdf_page

                if self._pdf_view == "text":
                    self._set_fallback(pdf_page_body(self._pdf_doc, self._pdf_page))
                    return
                image = render_pdf_page(self._pdf_doc, self._pdf_page, px_w, px_h)
            else:
                self._set_fallback("")
                return
        except Exception as exc:
            self._set_fallback(f"Preview failed: {exc}")
            return
        if self._wants_graphics():
            self._show_graphics(image)
            return
        self._set_fallback(image_to_halfblocks(image))

    def _fit_still(self, image: Image.Image, width: int, height: int) -> Image.Image:
        if self._wants_graphics():
            fitted = image.copy()
            fitted.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            return fitted
        fitted = image.copy()
        fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), _BG)
        x = max(0, (width - fitted.width) // 2)
        y = max(0, (height - fitted.height) // 2)
        canvas.paste(fitted, (x, y))
        return canvas

    def repaint(self) -> None:
        self._paint()

    def preview_label(self) -> str:
        if self._kind == "pdf":
            if self._pdf_view == "text":
                return "text"
            if graphics_enabled():
                return graphics_label()
            return "blocks"
        if self._kind == "image":
            if graphics_enabled():
                return graphics_label()
            return "blocks"
        return "preview"

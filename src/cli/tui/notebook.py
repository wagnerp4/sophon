from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field

from PIL import Image
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Markdown, Static

from cli.tui.preview import image_to_halfblocks

_MAX_CELLS = 80
_MAX_OUTPUT_CHARS = 8000
_IMAGE_CELLS_W = 48
_IMAGE_CELLS_H = 16
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# TODO: kernel execution (jupyter_client) for live Run Cell
# TODO: widget / html / latex outputs
# TODO: collapse long outputs / cell folding


@dataclass
class NotebookOutput:
    kind: str
    text: str = ""
    image: Image.Image | None = None


@dataclass
class NotebookCell:
    cell_type: str
    source: str
    outputs: list[NotebookOutput] = field(default_factory=list)
    execution: str = ""


def parse_notebook(text: str) -> list[NotebookCell]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Notebook JSON root must be an object.")
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list):
        raise ValueError("Notebook has no cells array.")
    cells: list[NotebookCell] = []
    for raw in raw_cells[:_MAX_CELLS]:
        if not isinstance(raw, dict):
            continue
        cell_type = str(raw.get("cell_type") or "raw")
        source = _join_source(raw.get("source"))
        count = raw.get("execution_count")
        execution = str(count) if count is not None else ""
        outputs = [
            _parse_output(item)
            for item in raw.get("outputs") or []
            if isinstance(item, dict)
        ]
        cells.append(
            NotebookCell(
                cell_type=cell_type,
                source=source,
                outputs=[item for item in outputs if item.text or item.image is not None],
                execution=execution,
            )
        )
    return cells


class NotebookPreview(VerticalScroll):
    DEFAULT_CSS = """
    NotebookPreview {
        height: 1fr;
        border: none;
        background: $background;
    }

    NotebookPreview .nb-cell {
        height: auto;
        padding: 0 1 1 1;
        border-bottom: solid $secondary;
    }

    NotebookPreview .nb-label {
        color: $text-muted;
        text-style: bold;
        padding: 1 0 0 0;
    }

    NotebookPreview .nb-code {
        height: auto;
        color: $text;
    }

    NotebookPreview .nb-out {
        height: auto;
        color: $text-muted;
        padding-left: 1;
    }

    NotebookPreview .nb-err {
        height: auto;
        color: $error;
        padding-left: 1;
    }

    NotebookPreview .nb-image {
        height: auto;
        padding-left: 1;
    }
    """

    async def show_notebook(self, text: str) -> None:
        await self.remove_children()
        try:
            cells = parse_notebook(text)
        except (json.JSONDecodeError, ValueError) as exc:
            await self.mount(Static(f"Notebook parse failed: {exc}"))
            return
        if not cells:
            await self.mount(Static("Empty notebook."))
            return
        for index, cell in enumerate(cells, start=1):
            await self.mount(_NotebookCellView(index, cell))
        self.scroll_home(animate=False)

    async def clear_notebook(self) -> None:
        await self.remove_children()


class _NotebookCellView(Vertical):
    def __init__(self, index: int, cell: NotebookCell) -> None:
        super().__init__(classes="nb-cell")
        self._index = index
        self._cell = cell

    def compose(self) -> ComposeResult:
        cell = self._cell
        kind = cell.cell_type
        if kind == "code":
            label = f"In [{cell.execution or self._index}]"
        else:
            label = f"{kind} [{self._index}]"
        yield Static(label, classes="nb-label")
        if kind == "markdown":
            yield Markdown(cell.source or "_Empty markdown cell._")
        else:
            yield Static(_code_renderable(cell.source), classes="nb-code")
        for output in cell.outputs:
            if output.image is not None:
                yield Static(image_to_halfblocks(output.image), classes="nb-image")
            if output.text:
                css = "nb-err" if output.kind == "error" else "nb-out"
                yield Static(output.text, classes=css)


def _join_source(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    if value is None:
        return ""
    return str(value)


def _parse_output(raw: dict) -> NotebookOutput:
    kind = str(raw.get("output_type") or "stream")
    if kind == "stream":
        return NotebookOutput(kind="stream", text=_clip(_join_source(raw.get("text"))))
    if kind == "error":
        traceback = raw.get("traceback")
        if isinstance(traceback, list):
            body = "\n".join(str(line) for line in traceback)
        else:
            body = f"{raw.get('ename', '')}: {raw.get('evalue', '')}"
        return NotebookOutput(kind="error", text=_clip(_strip_ansi(body)))
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    image = _image_from_data(data)
    text = ""
    if "text/markdown" in data:
        text = _clip(_join_source(data.get("text/markdown")))
    elif "text/plain" in data:
        text = _clip(_join_source(data.get("text/plain")))
    elif "text/html" in data:
        text = _clip(_join_source(data.get("text/html")))
    return NotebookOutput(kind=kind, text=text, image=image)


def _image_from_data(data: dict) -> Image.Image | None:
    blob = data.get("image/png") or data.get("image/jpeg")
    if blob is None:
        return None
    raw = _join_source(blob).replace("\n", "")
    try:
        decoded = base64.b64decode(raw)
        image = Image.open(io.BytesIO(decoded)).convert("RGB")
    except Exception:
        return None
    return image.resize((_IMAGE_CELLS_W, _IMAGE_CELLS_H * 2), Image.Resampling.LANCZOS)


def _code_renderable(source: str):
    try:
        from rich.syntax import Syntax

        return Syntax(source or " ", "python", theme="ansi_dark", word_wrap=True, padding=0)
    except Exception:
        return source or ""


def _clip(text: str) -> str:
    cleaned = _strip_ansi(text)
    if len(cleaned) > _MAX_OUTPUT_CHARS:
        return cleaned[:_MAX_OUTPUT_CHARS] + "\n…"
    return cleaned


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)

from __future__ import annotations

from PIL import Image

_BG = (18, 20, 24)
_MAX_TEXT_PAGES = 20
# TODO: optional higher-dpi raster when the preview pane is large
# TODO: two-page spread


def _load_pymupdf():
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        import fitz as pymupdf
        return pymupdf


def pdf_available() -> bool:
    try:
        _load_pymupdf()
    except ImportError:
        return False
    return True


def open_pdf(data: bytes):
    try:
        pymupdf = _load_pymupdf()
    except ImportError as exc:
        raise ValueError(
            "PDF preview needs pymupdf. Install the tui extra (uv sync --extra tui)."
        ) from exc
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc
    if getattr(doc, "is_encrypted", False):
        unlocked = False
        try:
            unlocked = bool(doc.authenticate(""))
        except Exception:
            unlocked = False
        if not unlocked:
            doc.close()
            raise ValueError("Encrypted PDF is not supported.")
    if doc.page_count < 1:
        doc.close()
        raise ValueError("PDF has no pages.")
    return doc


def pdf_source_note(doc) -> str:
    count = int(doc.page_count)
    chunks = [
        f"PDF · {count} page{'s' if count != 1 else ''}",
        "Preview-only. PgUp/PgDn or left/right changes page. Saving is disabled.",
        "",
    ]
    for index in range(min(count, _MAX_TEXT_PAGES)):
        text = (doc[index].get_text("text") or "").strip()
        chunks.append(f"--- page {index + 1} ---")
        chunks.append(text or "(no extractable text)")
        chunks.append("")
    if count > _MAX_TEXT_PAGES:
        chunks.append(f"--- {count - _MAX_TEXT_PAGES} more pages not listed ---")
    return "\n".join(chunks)


def render_pdf_page(doc, page_index: int, width: int, height: int) -> Image.Image:
    pymupdf = _load_pymupdf()

    width = max(2, int(width))
    height = max(2, int(height))
    last = int(doc.page_count) - 1
    page_index = max(0, min(last, int(page_index)))
    page = doc[page_index]
    rect = page.rect
    if rect.width <= 0 or rect.height <= 0:
        return Image.new("RGB", (width, height), _BG)
    scale = min(width / float(rect.width), height / float(rect.height))
    scale = max(scale, 0.05)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    canvas = Image.new("RGB", (width, height), _BG)
    x = max(0, (width - image.width) // 2)
    y = max(0, (height - image.height) // 2)
    canvas.paste(image, (x, y))
    return canvas

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static

from cli.tui.tiles.base import BaseTile, TileState, http_get_json, http_get_text, wrap_fetch

_DEFAULT_FEEDS = (
    "https://www.tagesschau.de/xml/rss2/,"
    "https://www.br.de/nachrichten/rss"
)
_DEFAULT_ARXIV_QUERIES = (
    "all:\"sound event detection\" OR all:\"sound event detection and localization\" OR "
    "all:\"sound event localization\" OR all:SELD",
    "all:\"brain-computer interface\" OR all:\"brain computer interface\" OR "
    "(ti:BCI AND (all:EEG OR all:neural OR all:interface)) OR "
    "cat:q-bio.NC",
)
_ARXIV_PER_QUERY = 3


def _news_limit() -> int:
    raw = os.environ.get("ORODRUIN_NEWS_LIMIT", "6").strip()
    try:
        return max(1, min(int(raw or "6"), 12))
    except ValueError:
        return 6


def _papers_limit() -> int:
    raw = os.environ.get("ORODRUIN_PAPERS_LIMIT", "4").strip()
    try:
        return max(1, min(int(raw or "4"), 8))
    except ValueError:
        return 4


def _feeds() -> list[str]:
    raw = os.environ.get("ORODRUIN_NEWS_FEEDS", _DEFAULT_FEEDS).strip()
    return [part.strip() for part in raw.split(",") if part.strip()]


def _want_hn() -> bool:
    raw = os.environ.get("ORODRUIN_NEWS_HN", "1").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _want_hf() -> bool:
    raw = os.environ.get("ORODRUIN_PAPERS_HF", "1").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _arxiv_cat() -> str:
    return os.environ.get("ORODRUIN_NEWS_ARXIV_CAT", "").strip() or os.environ.get(
        "ORODRUIN_PAPERS_ARXIV_CAT", ""
    ).strip()


def _arxiv_queries() -> list[str]:
    raw = (
        os.environ.get("ORODRUIN_PAPERS_ARXIV_QUERY", "").strip()
        or os.environ.get("ORODRUIN_NEWS_ARXIV_QUERY", "").strip()
    )
    if raw:
        return [part.strip() for part in raw.split("||") if part.strip()]
    if _arxiv_cat():
        return []
    return list(_DEFAULT_ARXIV_QUERIES)


def _escape_rich(text: str) -> str:
    return text.replace("[", "\\[").replace("]", "\\]")


def _parse_rss_items(xml_text: str, *, source: str, limit: int) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    items: list[dict[str, Any]] = []
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1].lower()
        if tag != "item":
            continue
        title = ""
        link = ""
        published = ""
        for child in list(node):
            ctag = child.tag.rsplit("}", 1)[-1].lower()
            if ctag == "title" and child.text:
                title = child.text.strip()
            elif ctag == "link" and child.text:
                link = child.text.strip()
            elif ctag in ("pubdate", "published", "updated") and child.text:
                published = child.text.strip()
        if not title:
            continue
        ts = 0.0
        if published:
            try:
                ts = parsedate_to_datetime(published).timestamp()
            except (TypeError, ValueError, OverflowError):
                ts = 0.0
        items.append(
            {
                "title": title,
                "link": link,
                "source": source,
                "published_at": ts,
            }
        )
        if len(items) >= limit:
            break
    return items


def _fetch_rss(limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    per_feed = max(2, limit // max(1, len(_feeds()) or 1))
    for url in _feeds():
        try:
            body = http_get_text(url, timeout_s=10.0)
            host = url.split("/")[2] if "://" in url else url
            out.extend(_parse_rss_items(body, source=host, limit=per_feed))
        except Exception:
            continue
    return out


def _fetch_hn(limit: int) -> list[dict[str, Any]]:
    ids = http_get_json("https://hacker-news.firebaseio.com/v0/topstories.json", timeout_s=8.0)
    if not isinstance(ids, list):
        return []
    items: list[dict[str, Any]] = []
    for story_id in ids[: max(limit * 3, 15)]:
        try:
            data = http_get_json(
                f"https://hacker-news.firebaseio.com/v0/item/{int(story_id)}.json",
                timeout_s=6.0,
            )
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        title = str(data.get("title") or "").strip()
        if not title:
            continue
        items.append(
            {
                "title": title,
                "link": str(data.get("url") or f"https://news.ycombinator.com/item?id={story_id}"),
                "source": "hn",
                "published_at": float(data.get("time") or 0),
                "score": data.get("score"),
            }
        )
        if len(items) >= limit:
            break
    return items


def _build_arxiv_searches() -> list[str]:
    queries = _arxiv_queries()
    cat = _arxiv_cat()
    if queries:
        if cat:
            return [f"({query}) AND cat:{cat}" for query in queries]
        return queries
    if cat:
        return [f"cat:{cat}"]
    return []


def _fetch_arxiv_query(search: str, limit: int) -> list[dict[str, Any]]:
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query={quote(search)}&sortBy=submittedDate&sortOrder=descending"
        f"&start=0&max_results={max(1, limit)}"
    )
    body = http_get_text(url, timeout_s=12.0)
    root = ET.fromstring(body)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        title_node = entry.find("atom:title", ns)
        id_node = entry.find("atom:id", ns)
        updated = entry.find("atom:updated", ns)
        published = entry.find("atom:published", ns)
        title = (title_node.text or "").strip().replace("\n", " ")
        link = (id_node.text or "").strip() if id_node is not None else ""
        if link.startswith("http://"):
            link = "https://" + link[len("http://") :]
        ts = 0.0
        stamp_node = published if published is not None and published.text else updated
        if stamp_node is not None and stamp_node.text:
            raw_ts = stamp_node.text.strip().replace("Z", "+00:00")
            try:
                ts = datetime.fromisoformat(raw_ts).timestamp()
            except (TypeError, ValueError, OverflowError):
                try:
                    ts = parsedate_to_datetime(stamp_node.text.strip()).timestamp()
                except (TypeError, ValueError, OverflowError):
                    ts = 0.0
        if not title:
            continue
        items.append(
            {
                "title": title,
                "link": link,
                "source": "arxiv",
                "published_at": ts,
                "section": "arxiv",
            }
        )
    return items


def _fetch_arxiv(limit: int) -> list[dict[str, Any]]:
    searches = _build_arxiv_searches()
    if not searches:
        return []
    per = max(1, min(_ARXIV_PER_QUERY, limit))
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for search in searches:
        for row in _fetch_arxiv_query(search, per):
            link = str(row.get("link") or "")
            if link and link in seen:
                continue
            if link:
                seen.add(link)
            items.append(row)
            if len(items) >= limit:
                return items
    return items


def _parse_hf_ts(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _fetch_hf_daily(limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for offset in range(0, 4):
        day = date.today() - timedelta(days=offset)
        url = (
            "https://huggingface.co/api/daily_papers"
            f"?date={day.isoformat()}&limit={max(limit, 5)}&sort=trending"
        )
        try:
            data = http_get_json(url, timeout_s=12.0)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            paper = entry.get("paper") if isinstance(entry.get("paper"), dict) else {}
            paper_id = str(paper.get("id") or entry.get("id") or "").strip()
            title = str(paper.get("title") or entry.get("title") or "").strip()
            if not title:
                continue
            link = f"https://huggingface.co/papers/{paper_id}" if paper_id else ""
            if not link:
                continue
            key = paper_id or link or title
            if key in seen:
                continue
            seen.add(key)
            upvotes = paper.get("upvotes")
            items.append(
                {
                    "title": title,
                    "link": link,
                    "source": "hf",
                    "published_at": _parse_hf_ts(
                        paper.get("publishedAt") or entry.get("publishedAt")
                    ),
                    "section": "hf",
                    "upvotes": upvotes,
                }
            )
            if len(items) >= limit:
                return items
        if items:
            break
    return items[:limit]


def _short_title(title: str, width: int = 52) -> str:
    clean = " ".join(title.split())
    if len(clean) <= width:
        return clean
    return clean[: width - 1] + "…"


def _short_url(url: str, width: int = 56) -> str:
    clean = " ".join(str(url or "").split())
    if len(clean) <= width:
        return clean
    return clean[: width - 1] + "…"


def _rel_time(ts: float, *, now: float | None = None) -> str:
    if not ts:
        return "?"
    base = now if now is not None else time.time()
    age = max(0, int(base - ts))
    if age < 60:
        return f"{age}s"
    if age < 3600:
        return f"{age // 60}m"
    if age < 86400:
        return f"{age // 3600}h"
    if age < 86400 * 7:
        return f"{age // 86400}d"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _format_item_markup(row: dict[str, Any], *, now: float, title_width: int = 52) -> str:
    src = _escape_rich(str(row.get("source") or "?"))
    when = _escape_rich(_rel_time(float(row.get("published_at") or 0), now=now))
    title = _escape_rich(_short_title(str(row.get("title") or ""), width=title_width))
    link = str(row.get("link") or "").strip()
    head = f"[[{src}]] {when} · {title}"
    if not link:
        return head
    shown = _escape_rich(_short_url(link))
    href = link.replace("\\", "\\\\").replace('"', "%22")
    return f'{head}\n  [u][link="{href}"]{shown}[/link][/u]'


def _fetch_general_news(limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        rows.extend(_fetch_rss(limit))
    except Exception as exc:
        errors.append(f"rss: {exc}")
    if _want_hn():
        try:
            rows.extend(_fetch_hn(min(5, limit)))
        except Exception as exc:
            errors.append(f"hn: {exc}")
    rows.sort(key=lambda row: float(row.get("published_at") or 0), reverse=True)
    return rows[:limit], errors


def _fetch_papers(limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    arxiv_rows: list[dict[str, Any]] = []
    hf_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_arxiv = pool.submit(_fetch_arxiv, limit) if _build_arxiv_searches() else None
        fut_hf = pool.submit(_fetch_hf_daily, limit) if _want_hf() else None
        if fut_arxiv is not None:
            try:
                arxiv_rows = fut_arxiv.result()
            except Exception as exc:
                errors.append(f"arxiv: {exc}")
        if fut_hf is not None:
            try:
                hf_rows = fut_hf.result()
            except Exception as exc:
                errors.append(f"hf: {exc}")
    return arxiv_rows[:limit], hf_rows[:limit], errors


def _fetch_news() -> TileState:
    news_limit = _news_limit()
    papers_limit = _papers_limit()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_news = pool.submit(_fetch_general_news, news_limit)
        fut_papers = pool.submit(_fetch_papers, papers_limit)
        news_rows, news_errors = fut_news.result()
        arxiv_rows, hf_rows, paper_errors = fut_papers.result()

    errors = [*news_errors, *paper_errors]
    if not news_rows and not arxiv_rows and not hf_rows:
        raise ValueError("; ".join(errors) if errors else "no news or papers")

    now = time.time()
    lines = ["news | papers"]
    lines.append("[news]")
    if news_rows:
        for row in news_rows:
            lines.append(_format_item_markup(row, now=now, title_width=56).replace("\n", "\n  "))
    else:
        lines.append("  —")
    lines.append("[papers · arxiv]")
    if arxiv_rows:
        for row in arxiv_rows:
            lines.append(_format_item_markup(row, now=now, title_width=48).replace("\n", "\n  "))
    else:
        lines.append("  —")
    lines.append("[papers · hf]")
    if hf_rows:
        for row in hf_rows:
            lines.append(_format_item_markup(row, now=now, title_width=48).replace("\n", "\n  "))
    else:
        lines.append("  —")

    return TileState(
        ok=True,
        text="\n".join(lines),
        fetched_at=now,
        error="; ".join(errors),
        payload={
            "news": news_rows,
            "arxiv": arxiv_rows,
            "hf": hf_rows,
        },
    )


class NewsPapersPanel(Vertical):
    DEFAULT_CSS = """
    NewsPapersPanel {
        border: solid $primary;
        padding: 0 1;
        height: 1fr;
        min-height: 12;
    }
    NewsPapersPanel.tile-span-2 {
        column-span: 2;
    }
    #news-papers-body {
        height: 1fr;
        layout: horizontal;
    }
    .news-papers-col {
        width: 1fr;
        height: 1fr;
        padding: 0 1 0 0;
    }
    .news-papers-col-title {
        height: 1;
        text-style: bold;
    }
    .news-papers-section {
        height: 1;
        color: $text-muted;
        text-style: bold;
        margin-top: 1;
    }
    .news-papers-line {
        height: auto;
        color: $text;
    }
    #news-col-scroll, #papers-col-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="news-papers-body"):
            with Vertical(id="news-col", classes="news-papers-col"):
                yield Static("news", classes="news-papers-col-title")
                with VerticalScroll(id="news-col-scroll"):
                    yield Static("", id="news-col-body")
            with Vertical(id="papers-col", classes="news-papers-col"):
                yield Static("papers", classes="news-papers-col-title")
                with VerticalScroll(id="papers-col-scroll"):
                    yield Static("", id="papers-col-body")

    def _render_items(self, rows: list[dict[str, Any]], *, now: float, title_width: int) -> str:
        if not rows:
            return "—"
        blocks = [_format_item_markup(row, now=now, title_width=title_width) for row in rows]
        return "\n".join(blocks)

    def apply(self, state: TileState) -> None:
        payload = state.payload or {}
        now = float(state.fetched_at or time.time())
        news_rows = list(payload.get("news") or []) if isinstance(payload, dict) else []
        arxiv_rows = list(payload.get("arxiv") or []) if isinstance(payload, dict) else []
        hf_rows = list(payload.get("hf") or []) if isinstance(payload, dict) else []

        news_body = self.query_one("#news-col-body", Static)
        papers_body = self.query_one("#papers-col-body", Static)

        if not state.ok and not news_rows and not arxiv_rows and not hf_rows:
            news_body.update(_escape_rich(state.error or "news n/a"))
            papers_body.update("—")
            return

        news_body.update(self._render_items(news_rows, now=now, title_width=56))

        paper_parts: list[str] = []
        paper_parts.append("[bold]arxiv top picks[/bold]")
        paper_parts.append(self._render_items(arxiv_rows, now=now, title_width=46))
        paper_parts.append("")
        paper_parts.append("[bold]hf daily · trending[/bold]")
        paper_parts.append(self._render_items(hf_rows, now=now, title_width=46))
        if state.error and not arxiv_rows and not hf_rows:
            paper_parts.append(_escape_rich(state.error))
        papers_body.update("\n".join(paper_parts))


class NewsTile(BaseTile):
    tile_id = "news"
    refresh_s = 600.0
    span_cols = 2

    def panel(self) -> Any:
        panel = NewsPapersPanel(id=f"dash-{self.tile_id}")
        panel.add_class("tile-span-2")
        return panel

    def fetch(self, app: Any) -> TileState:
        return wrap_fetch(self.tile_id, "news", _fetch_news)

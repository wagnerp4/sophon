from __future__ import annotations

import json
import os
import sqlite3
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DEFAULT_URL = "http://127.0.0.1:23119/api"
_ITEM_LIMIT = 250
_SKIP_ITEM_TYPES = {"attachment", "note", "annotation"}
_MAX_BODY_CHARS = 12_000


def zotero_api_url() -> str:
    raw = os.environ.get("SOPHON_ZOTERO_API_URL", "").strip()
    if raw:
        return raw.rstrip("/")
    return _DEFAULT_URL


def zotero_db_path() -> Path | None:
    for key in ("SOPHON_ZOTERO_DB_PATH", "ZOTERO_DB_PATH"):
        raw = os.environ.get(key, "").strip()
        if raw:
            path = Path(raw).expanduser()
            return path if path.is_file() else None
    for candidate in _guess_db_paths():
        if candidate.is_file():
            return candidate
    return None


def _guess_db_paths() -> list[Path]:
    out: list[Path] = [Path.home() / "Zotero" / "zotero.sqlite"]
    profile = os.environ.get("USERPROFILE", "").strip()
    if profile:
        out.append(Path(profile) / "Zotero" / "zotero.sqlite")
    users = Path("/mnt/c/Users")
    if users.is_dir():
        out.extend(sorted(users.glob("*/Zotero/zotero.sqlite")))
    return out


def zotero_available() -> bool:
    if zotero_db_path() is not None:
        return True
    try:
        _request("/users/0/collections", query={"limit": "1"}, timeout_s=2.0)
        return True
    except Exception:
        return False


def zotero_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_ZOTERO_TOOLS", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return zotero_available()


def _request(
    path: str,
    *,
    query: dict[str, str] | None = None,
    timeout_s: float = 30.0,
    expect_json: bool = True,
) -> Any:
    base = zotero_api_url()
    url = f"{base}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    headers = {
        "User-Agent": "sophon-zotero/0.1",
        "Accept": "application/json" if expect_json else "*/*",
        "Zotero-API-Version": "3",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Zotero HTTP {exc.code} on {path}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Zotero unreachable at {base} ({exc.reason}). "
            "Start Zotero with local API enabled."
        ) from exc
    if not raw.strip():
        return [] if expect_json else ""
    if expect_json:
        return json.loads(raw)
    return raw


def _parent_key(value: object) -> str | None:
    if value is None or value is False:
        return None
    text = str(value).strip()
    if not text or text.lower() == "false":
        return None
    return text


def _item_year(date: object) -> str:
    text = str(date or "").strip()
    if len(text) >= 4 and text[:4].isdigit():
        return text[:4]
    return ""


def _item_creators(creators: object) -> str:
    if not isinstance(creators, list):
        return ""
    names: list[str] = []
    for row in creators:
        if not isinstance(row, dict):
            continue
        if row.get("name"):
            names.append(str(row["name"]).strip())
            continue
        last = str(row.get("lastName") or "").strip()
        first = str(row.get("firstName") or "").strip()
        label = " ".join(part for part in (first, last) if part)
        if label:
            names.append(label)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return f"{names[0]} et al."


def _path_from_file_url(raw: str) -> Path | None:
    text = raw.strip()
    if not text:
        return None
    if text.startswith("file:"):
        parsed = urllib.parse.urlparse(text)
        path = urllib.parse.unquote(parsed.path)
        if sys.platform == "win32" and path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path.lstrip("/")
        return Path(path)
    return Path(text)


def _storage_pdf_path(data_dir: Path, attachment_key: str, stored: str) -> Path | None:
    if not stored:
        return None
    if stored.startswith("storage:"):
        name = stored.split(":", 1)[1]
        path = data_dir / "storage" / attachment_key / name
        return path if path.is_file() else None
    candidate = Path(stored)
    if candidate.is_file():
        return candidate
    nested = data_dir / "storage" / attachment_key / candidate.name
    return nested if nested.is_file() else None


@dataclass(frozen=True)
class ZoteroCollection:
    key: str
    name: str
    parent_key: str | None
    n_items: int = 0
    n_collections: int = 0


@dataclass(frozen=True)
class ZoteroItem:
    key: str
    title: str
    item_type: str
    creators: str = ""
    year: str = ""


class ZoteroClient:
    def ping(self) -> dict[str, Any]:
        try:
            _request("/users/0/collections", query={"limit": "1"}, timeout_s=2.0)
            return {"ok": True, "backend": "http", "url": zotero_api_url()}
        except Exception as exc:
            db = zotero_db_path()
            if db is not None:
                return {"ok": True, "backend": "sqlite", "url": str(db)}
            return {"ok": False, "backend": "none", "url": zotero_api_url(), "error": str(exc)}

    def list_collections(self) -> list[ZoteroCollection]:
        # TODO: include group libraries as extra roots
        try:
            return self._http_collections()
        except Exception:
            return self._sqlite_collections()

    def list_collection_items(self, key: str, *, limit: int = _ITEM_LIMIT) -> list[ZoteroItem]:
        # TODO: page collections larger than _ITEM_LIMIT
        try:
            return self._http_items(key, limit=limit)
        except Exception:
            return self._sqlite_items(key, limit=limit)

    def item_pdf_path(self, key: str) -> Path | None:
        try:
            path = self._http_pdf_path(key)
            if path is not None:
                return path
        except Exception:
            pass
        return self._sqlite_pdf_path(key)

    def item_markdown(self, key: str) -> str:
        try:
            return self._http_markdown(key)
        except Exception:
            return self._sqlite_markdown(key)

    def resolve_collection(self, spec: str) -> ZoteroCollection | None:
        text = spec.strip()
        if not text:
            return None
        cols = self.list_collections()
        by_key = {col.key: col for col in cols}
        if text in by_key:
            return by_key[text]
        grouped: dict[str | None, list[ZoteroCollection]] = {}
        for col in cols:
            grouped.setdefault(col.parent_key, []).append(col)
        if "/" in text:
            current: ZoteroCollection | None = None
            parent: str | None = None
            for part in [bit.strip() for bit in text.split("/") if bit.strip()]:
                kids = grouped.get(parent, [])
                match = next((kid for kid in kids if kid.name.lower() == part.lower()), None)
                if match is None:
                    return None
                current = match
                parent = match.key
            return current
        exact = [col for col in cols if col.name.lower() == text.lower()]
        if len(exact) == 1:
            return exact[0]
        if exact:
            return exact[0]
        contains = [col for col in cols if text.lower() in col.name.lower()]
        if len(contains) == 1:
            return contains[0]
        return None

    def format_tree(self, *, collection: str = "", depth: int = 2) -> dict[str, Any]:
        cols = self.list_collections()
        grouped: dict[str | None, list[ZoteroCollection]] = {}
        for col in cols:
            grouped.setdefault(col.parent_key, []).append(col)
        for rows in grouped.values():
            rows.sort(key=lambda item: item.name.lower())
        root_key: str | None = None
        root_name = "Zotero"
        if collection.strip():
            found = self.resolve_collection(collection)
            if found is None:
                return {"error": f"collection not found: {collection}"}
            root_key = found.key
            root_name = found.name
        levels = max(1, min(int(depth), 8))
        lines: list[str] = [f"{root_name}  collections={len(cols)}"]
        omitted = 0

        def walk(parent: str | None, level: int, remaining: int) -> None:
            nonlocal omitted
            kids = grouped.get(parent, [])
            if remaining <= 0:
                if kids:
                    omitted += len(kids)
                return
            for col in kids:
                lines.append(f"{'  ' * level}- {col.name}  [{col.key}]")
                walk(col.key, level + 1, remaining - 1)

        walk(root_key, 0, levels)
        return {
            "root": root_name,
            "depth": levels,
            "collections": len(cols),
            "omitted_branches": omitted,
            "tree": "\n".join(lines),
        }

    def search_items(
        self,
        query: str,
        *,
        collection: str = "",
        limit: int = 20,
    ) -> dict[str, Any]:
        q = query.strip()
        if not q:
            return {"error": "query is required"}
        n = max(1, min(int(limit), 50))
        col_key = ""
        col_name = ""
        if collection.strip():
            found = self.resolve_collection(collection)
            if found is None:
                return {"error": f"collection not found: {collection}"}
            col_key = found.key
            col_name = found.name
        if zotero_db_path() is not None:
            hits = self._sqlite_search(q, collection_key=col_key or None, limit=n)
            backend = "sqlite"
        else:
            hits = self._http_search(q, collection_key=col_key or None, limit=n)
            backend = "http"
        return {
            "query": q,
            "collection": col_name or None,
            "backend": backend,
            "count": len(hits),
            "items": hits,
        }

    def library_metrics(self) -> dict[str, Any]:
        if zotero_db_path() is None:
            cols = self.list_collections()
            return {
                "backend": "http",
                "collections": len(cols),
                "note": "Start with sqlite (SOPHON_ZOTERO_DB_PATH) for citation coverage and year/type breakdowns.",
            }
        return self._sqlite_metrics()

    def item_record(self, spec: str, *, include_pdf_text: bool = False) -> str:
        key = spec.strip()
        if not key:
            return "error: key or title is required"
        if not (len(key) == 8 and key.isalnum()):
            found = self.search_items(key, limit=5)
            items = found.get("items") if isinstance(found, dict) else None
            if isinstance(items, list) and items:
                first = items[0]
                if isinstance(first, dict) and first.get("key"):
                    key = str(first["key"])
        body = self.item_markdown(key)
        if not include_pdf_text:
            return body
        excerpt = self.pdf_excerpt(key)
        if not excerpt:
            return body + "\n\n(no local PDF text indexed or readable)\n"
        return body + "\n\n## PDF excerpt\n\n" + excerpt

    def pdf_excerpt(self, key: str, *, max_chars: int = 6000) -> str:
        path = self.item_pdf_path(key)
        if path is None or not path.is_file():
            return ""
        try:
            import fitz
        except ImportError:
            return ""
        # TODO: page-range reads and OCR fallback for image-only PDFs
        try:
            doc = fitz.open(path)
        except Exception:
            return ""
        chunks: list[str] = []
        total = 0
        try:
            for page in doc:
                text = page.get_text("text") or ""
                if not text.strip():
                    continue
                chunks.append(text.strip())
                total += len(text)
                if total >= max_chars:
                    break
        finally:
            doc.close()
        blob = "\n\n".join(chunks).strip()
        if len(blob) > max_chars:
            return blob[: max_chars - 1] + "…"
        return blob

    def _http_collections(self) -> list[ZoteroCollection]:
        data = _request("/users/0/collections", query={"limit": "5000"})
        if not isinstance(data, list):
            return []
        out: list[ZoteroCollection] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            payload = row.get("data") if isinstance(row.get("data"), dict) else {}
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            key = str(row.get("key") or payload.get("key") or "").strip()
            name = str(payload.get("name") or "").strip()
            if not key or not name:
                continue
            n_items = meta.get("numItems")
            n_cols = meta.get("numCollections")
            out.append(
                ZoteroCollection(
                    key=key,
                    name=name,
                    parent_key=_parent_key(payload.get("parentCollection")),
                    n_items=int(n_items) if isinstance(n_items, int) else 0,
                    n_collections=int(n_cols) if isinstance(n_cols, int) else 0,
                )
            )
        return out

    def _http_items(self, key: str, *, limit: int) -> list[ZoteroItem]:
        data = _request(
            f"/users/0/collections/{urllib.parse.quote(key)}/items/top",
            query={"limit": str(max(1, min(limit, 500)))},
        )
        if not isinstance(data, list):
            return []
        out: list[ZoteroItem] = []
        for row in data:
            item = self._parse_item(row)
            if item is not None:
                out.append(item)
        return out

    def _http_pdf_path(self, key: str) -> Path | None:
        children = _request(f"/users/0/items/{urllib.parse.quote(key)}/children")
        if not isinstance(children, list):
            return None
        for row in children:
            if not isinstance(row, dict):
                continue
            payload = row.get("data") if isinstance(row.get("data"), dict) else {}
            if str(payload.get("contentType") or "") != "application/pdf":
                continue
            child_key = str(row.get("key") or payload.get("key") or "").strip()
            if not child_key:
                continue
            raw = _request(
                f"/users/0/items/{urllib.parse.quote(child_key)}/file/view/url",
                expect_json=False,
            )
            path = _path_from_file_url(str(raw))
            if path is not None and path.is_file():
                return path
        return None

    def _http_markdown(self, key: str) -> str:
        row = _request(f"/users/0/items/{urllib.parse.quote(key)}")
        if isinstance(row, list) and row:
            row = row[0]
        if not isinstance(row, dict):
            return f"Zotero item {key} was not found."
        return _markdown_from_api_item(row)

    def _parse_item(self, row: object) -> ZoteroItem | None:
        if not isinstance(row, dict):
            return None
        payload = row.get("data") if isinstance(row.get("data"), dict) else {}
        item_type = str(payload.get("itemType") or "").strip()
        if item_type in _SKIP_ITEM_TYPES:
            return None
        key = str(row.get("key") or payload.get("key") or "").strip()
        title = str(payload.get("title") or payload.get("name") or "Untitled").strip()
        if not key:
            return None
        return ZoteroItem(
            key=key,
            title=title or "Untitled",
            item_type=item_type,
            creators=_item_creators(payload.get("creators")),
            year=_item_year(payload.get("date")),
        )

    def _connect_sqlite(self) -> sqlite3.Connection:
        db = zotero_db_path()
        if db is None:
            raise RuntimeError("Zotero sqlite not found. Set SOPHON_ZOTERO_DB_PATH.")
        uri = f"file:{db.resolve().as_posix()}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _sqlite_collections(self) -> list[ZoteroCollection]:
        conn = self._connect_sqlite()
        try:
            deleted = {int(row[0]) for row in conn.execute("SELECT collectionID FROM deletedCollections")}
            rows = conn.execute(
                """
                SELECT c.collectionID, c.collectionName, c.parentCollectionID, c.key, p.key AS parentKey
                FROM collections c
                LEFT JOIN collections p ON p.collectionID = c.parentCollectionID
                WHERE c.libraryID = (SELECT libraryID FROM libraries WHERE type = 'user' LIMIT 1)
                """
            ).fetchall()
        finally:
            conn.close()
        out: list[ZoteroCollection] = []
        for row in rows:
            cid = int(row["collectionID"])
            if cid in deleted:
                continue
            name = str(row["collectionName"] or "").strip()
            key = str(row["key"] or "").strip()
            if not name or not key:
                continue
            parent = row["parentKey"]
            out.append(
                ZoteroCollection(
                    key=key,
                    name=name,
                    parent_key=str(parent) if parent else None,
                )
            )
        return out

    def _sqlite_items(self, key: str, *, limit: int) -> list[ZoteroItem]:
        conn = self._connect_sqlite()
        try:
            rows = conn.execute(
                """
                SELECT i.key AS itemKey, it.typeName, title.value AS title, datev.value AS dateVal,
                       c.lastName, c.firstName, ic.orderIndex
                FROM collections col
                JOIN collectionItems ci ON ci.collectionID = col.collectionID
                JOIN items i ON i.itemID = ci.itemID
                JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
                LEFT JOIN itemData title_id ON title_id.itemID = i.itemID AND title_id.fieldID = (
                    SELECT fieldID FROM fields WHERE fieldName = 'title' LIMIT 1
                )
                LEFT JOIN itemDataValues title ON title.valueID = title_id.valueID
                LEFT JOIN itemData date_id ON date_id.itemID = i.itemID AND date_id.fieldID = (
                    SELECT fieldID FROM fields WHERE fieldName = 'date' LIMIT 1
                )
                LEFT JOIN itemDataValues datev ON datev.valueID = date_id.valueID
                LEFT JOIN itemCreators ic ON ic.itemID = i.itemID AND ic.orderIndex = 0
                LEFT JOIN creators c ON c.creatorID = ic.creatorID
                WHERE col.key = ?
                  AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
                  AND it.typeName NOT IN ('attachment', 'note', 'annotation')
                ORDER BY title.value COLLATE NOCASE
                LIMIT ?
                """,
                (key, max(1, min(limit, 500))),
            ).fetchall()
        finally:
            conn.close()
        items: list[ZoteroItem] = []
        seen: set[str] = set()
        for row in rows:
            item_key = str(row["itemKey"] or "").strip()
            if not item_key or item_key in seen:
                continue
            seen.add(item_key)
            last = str(row["lastName"] or "").strip()
            first = str(row["firstName"] or "").strip()
            creators = " ".join(part for part in (first, last) if part)
            items.append(
                ZoteroItem(
                    key=item_key,
                    title=str(row["title"] or "Untitled").strip() or "Untitled",
                    item_type=str(row["typeName"] or "").strip(),
                    creators=creators,
                    year=_item_year(row["dateVal"]),
                )
            )
        return items

    def _sqlite_pdf_path(self, key: str) -> Path | None:
        db = zotero_db_path()
        if db is None:
            return None
        data_dir = db.parent
        conn = self._connect_sqlite()
        try:
            rows = conn.execute(
                """
                SELECT child.key AS attachKey, ia.path, ia.contentType
                FROM items parent
                JOIN itemAttachments ia ON ia.parentItemID = parent.itemID
                JOIN items child ON child.itemID = ia.itemID
                WHERE parent.key = ?
                  AND child.itemID NOT IN (SELECT itemID FROM deletedItems)
                """,
                (key,),
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            if str(row["contentType"] or "") != "application/pdf":
                continue
            stored = str(row["path"] or "")
            attach_key = str(row["attachKey"] or "")
            path = _storage_pdf_path(data_dir, attach_key, stored)
            if path is not None:
                return path
        return None

    def _http_search(self, query: str, *, collection_key: str | None, limit: int) -> list[dict[str, Any]]:
        path = "/users/0/items"
        if collection_key:
            path = f"/users/0/collections/{urllib.parse.quote(collection_key)}/items"
        data = _request(
            path,
            query={"q": query, "qmode": "everything", "limit": str(limit)},
        )
        if not isinstance(data, list):
            return []
        out: list[dict[str, Any]] = []
        for row in data:
            item = self._parse_item(row)
            if item is None:
                continue
            out.append(_item_dict(item, source="http"))
        return out

    def _sqlite_search(self, query: str, *, collection_key: str | None, limit: int) -> list[dict[str, Any]]:
        tokens = [tok.lower() for tok in _search_tokens(query)]
        if not tokens:
            tokens = [query.strip().lower()]
        conn = self._connect_sqlite()
        hits: dict[str, dict[str, Any]] = {}
        try:
            like = f"%{query.strip()}%"
            col_filter = ""
            meta_params: list[object] = [like, like, like, like]
            if collection_key:
                col_filter = """
                  AND i.itemID IN (
                    SELECT ci.itemID FROM collectionItems ci
                    JOIN collections col ON col.collectionID = ci.collectionID
                    WHERE col.key = ?
                  )
                """
                meta_params.append(collection_key)
            meta_params.append(limit)
            meta_sql = f"""
                SELECT i.key AS itemKey, it.typeName, title.value AS title, datev.value AS dateVal,
                       c.lastName, c.firstName
                FROM items i
                JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
                LEFT JOIN itemData title_id ON title_id.itemID = i.itemID AND title_id.fieldID = (
                    SELECT fieldID FROM fields WHERE fieldName = 'title' LIMIT 1
                )
                LEFT JOIN itemDataValues title ON title.valueID = title_id.valueID
                LEFT JOIN itemData date_id ON date_id.itemID = i.itemID AND date_id.fieldID = (
                    SELECT fieldID FROM fields WHERE fieldName = 'date' LIMIT 1
                )
                LEFT JOIN itemDataValues datev ON datev.valueID = date_id.valueID
                LEFT JOIN itemData extra_id ON extra_id.itemID = i.itemID AND extra_id.fieldID = (
                    SELECT fieldID FROM fields WHERE fieldName = 'extra' LIMIT 1
                )
                LEFT JOIN itemDataValues extra ON extra.valueID = extra_id.valueID
                LEFT JOIN itemCreators ic ON ic.itemID = i.itemID AND ic.orderIndex = 0
                LEFT JOIN creators c ON c.creatorID = ic.creatorID
                WHERE i.itemID NOT IN (SELECT itemID FROM deletedItems)
                  AND it.typeName NOT IN ('attachment', 'note', 'annotation')
                  AND i.libraryID = (SELECT libraryID FROM libraries WHERE type = 'user' LIMIT 1)
                  AND (
                    title.value LIKE ? COLLATE NOCASE
                    OR extra.value LIKE ? COLLATE NOCASE
                    OR c.lastName LIKE ? COLLATE NOCASE
                    OR c.firstName LIKE ? COLLATE NOCASE
                  )
                  {col_filter}
                LIMIT ?
            """
            for row in conn.execute(meta_sql, meta_params):
                item_key = str(row["itemKey"] or "").strip()
                if not item_key or item_key in hits:
                    continue
                hits[item_key] = _row_hit(row, source="metadata")
            remaining = max(0, limit - len(hits))
            if remaining and tokens:
                placeholders = ",".join("?" for _ in tokens)
                ft_sql = f"""
                    SELECT parent.key AS itemKey, it.typeName, title.value AS title, datev.value AS dateVal,
                           c.lastName, c.firstName, COUNT(DISTINCT w.word) AS wordHits
                    FROM fulltextWords w
                    JOIN fulltextItemWords fw ON fw.wordID = w.wordID
                    JOIN itemAttachments ia ON ia.itemID = fw.itemID
                    JOIN items parent ON parent.itemID = ia.parentItemID
                    JOIN itemTypes it ON it.itemTypeID = parent.itemTypeID
                    LEFT JOIN itemData title_id ON title_id.itemID = parent.itemID AND title_id.fieldID = (
                        SELECT fieldID FROM fields WHERE fieldName = 'title' LIMIT 1
                    )
                    LEFT JOIN itemDataValues title ON title.valueID = title_id.valueID
                    LEFT JOIN itemData date_id ON date_id.itemID = parent.itemID AND date_id.fieldID = (
                        SELECT fieldID FROM fields WHERE fieldName = 'date' LIMIT 1
                    )
                    LEFT JOIN itemDataValues datev ON datev.valueID = date_id.valueID
                    LEFT JOIN itemCreators ic ON ic.itemID = parent.itemID AND ic.orderIndex = 0
                    LEFT JOIN creators c ON c.creatorID = ic.creatorID
                    WHERE lower(w.word) IN ({placeholders})
                      AND parent.itemID NOT IN (SELECT itemID FROM deletedItems)
                      AND it.typeName NOT IN ('attachment', 'note', 'annotation')
                """
                ft_params: list[object] = list(tokens)
                if collection_key:
                    ft_sql += """
                      AND parent.itemID IN (
                        SELECT ci.itemID FROM collectionItems ci
                        JOIN collections col ON col.collectionID = ci.collectionID
                        WHERE col.key = ?
                      )
                    """
                    ft_params.append(collection_key)
                ft_sql += """
                    GROUP BY parent.key
                    HAVING COUNT(DISTINCT w.word) >= ?
                    ORDER BY wordHits DESC
                    LIMIT ?
                """
                ft_params.extend([len(tokens), remaining])
                for row in conn.execute(ft_sql, ft_params):
                    item_key = str(row["itemKey"] or "").strip()
                    if not item_key or item_key in hits:
                        continue
                    hits[item_key] = _row_hit(row, source="fulltext")
        finally:
            conn.close()
        return list(hits.values())[:limit]

    def _sqlite_metrics(self) -> dict[str, Any]:
        conn = self._connect_sqlite()
        try:
            items = conn.execute(
                """
                SELECT COUNT(*) FROM items i
                JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
                WHERE i.itemID NOT IN (SELECT itemID FROM deletedItems)
                  AND it.typeName NOT IN ('attachment', 'note', 'annotation')
                  AND i.libraryID = (SELECT libraryID FROM libraries WHERE type = 'user' LIMIT 1)
                """
            ).fetchone()[0]
            collections = conn.execute(
                """
                SELECT COUNT(*) FROM collections c
                WHERE c.libraryID = (SELECT libraryID FROM libraries WHERE type = 'user' LIMIT 1)
                  AND c.collectionID NOT IN (SELECT collectionID FROM deletedCollections)
                """
            ).fetchone()[0]
            pdfs = conn.execute(
                """
                SELECT COUNT(*) FROM itemAttachments ia
                JOIN items child ON child.itemID = ia.itemID
                WHERE ia.contentType = 'application/pdf'
                  AND child.itemID NOT IN (SELECT itemID FROM deletedItems)
                """
            ).fetchone()[0]
            indexed = conn.execute("SELECT COUNT(*) FROM fulltextItems").fetchone()[0]
            dois = conn.execute(
                """
                SELECT COUNT(*) FROM itemData d
                JOIN items i ON i.itemID = d.itemID
                JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
                WHERE d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'DOI' LIMIT 1)
                  AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
                  AND it.typeName NOT IN ('attachment', 'note', 'annotation')
                  AND i.libraryID = (SELECT libraryID FROM libraries WHERE type = 'user' LIMIT 1)
                """
            ).fetchone()[0]
            citekeys = conn.execute(
                """
                SELECT COUNT(*) FROM itemData d
                JOIN itemDataValues v ON v.valueID = d.valueID
                JOIN items i ON i.itemID = d.itemID
                WHERE d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'extra' LIMIT 1)
                  AND v.value LIKE '%Citation Key:%'
                  AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
                  AND i.libraryID = (SELECT libraryID FROM libraries WHERE type = 'user' LIMIT 1)
                """
            ).fetchone()[0]
            by_type = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    """
                    SELECT it.typeName, COUNT(*)
                    FROM items i
                    JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
                    WHERE i.itemID NOT IN (SELECT itemID FROM deletedItems)
                      AND it.typeName NOT IN ('attachment', 'note', 'annotation')
                      AND i.libraryID = (SELECT libraryID FROM libraries WHERE type = 'user' LIMIT 1)
                    GROUP BY it.typeName
                    ORDER BY COUNT(*) DESC
                    """
                )
            }
            by_year = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    """
                    SELECT substr(v.value, 1, 4) AS year, COUNT(*)
                    FROM itemData d
                    JOIN itemDataValues v ON v.valueID = d.valueID
                    JOIN items i ON i.itemID = d.itemID
                    JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
                    WHERE d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'date' LIMIT 1)
                      AND substr(v.value, 1, 4) GLOB '[0-9][0-9][0-9][0-9]'
                      AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
                      AND it.typeName NOT IN ('attachment', 'note', 'annotation')
                    GROUP BY year
                    ORDER BY year DESC
                    LIMIT 20
                    """
                )
            }
            top_collections = [
                {"key": str(row[0]), "name": str(row[1]), "items": int(row[2])}
                for row in conn.execute(
                    """
                    SELECT col.key, col.collectionName, COUNT(*)
                    FROM collections col
                    JOIN collectionItems ci ON ci.collectionID = col.collectionID
                    JOIN items i ON i.itemID = ci.itemID
                    JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
                    WHERE col.libraryID = (SELECT libraryID FROM libraries WHERE type = 'user' LIMIT 1)
                      AND col.collectionID NOT IN (SELECT collectionID FROM deletedCollections)
                      AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
                      AND it.typeName NOT IN ('attachment', 'note', 'annotation')
                    GROUP BY col.key
                    ORDER BY COUNT(*) DESC
                    LIMIT 12
                    """
                )
            ]
        finally:
            conn.close()
        return {
            "backend": "sqlite",
            "items": int(items),
            "collections": int(collections),
            "pdfs": int(pdfs),
            "fulltext_indexed_attachments": int(indexed),
            "with_doi": int(dois),
            "with_citation_key": int(citekeys),
            "by_type": by_type,
            "by_year": by_year,
            "top_collections": top_collections,
            "note": "Counts are local library coverage (items, DOIs, Better BibTeX keys, indexed PDFs). Live Crossref/Google Scholar citation counts are not stored in Zotero.",
        }

    def _sqlite_markdown(self, key: str) -> str:
        conn = self._connect_sqlite()
        try:
            row = conn.execute(
                """
                SELECT i.key, it.typeName, title.value AS title, datev.value AS dateVal,
                       abs.value AS abstract, doi.value AS doi, url.value AS url
                FROM items i
                JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
                LEFT JOIN itemData title_id ON title_id.itemID = i.itemID AND title_id.fieldID = (
                    SELECT fieldID FROM fields WHERE fieldName = 'title' LIMIT 1
                )
                LEFT JOIN itemDataValues title ON title.valueID = title_id.valueID
                LEFT JOIN itemData date_id ON date_id.itemID = i.itemID AND date_id.fieldID = (
                    SELECT fieldID FROM fields WHERE fieldName = 'date' LIMIT 1
                )
                LEFT JOIN itemDataValues datev ON datev.valueID = date_id.valueID
                LEFT JOIN itemData abs_id ON abs_id.itemID = i.itemID AND abs_id.fieldID = (
                    SELECT fieldID FROM fields WHERE fieldName = 'abstractNote' LIMIT 1
                )
                LEFT JOIN itemDataValues abs ON abs.valueID = abs_id.valueID
                LEFT JOIN itemData doi_id ON doi_id.itemID = i.itemID AND doi_id.fieldID = (
                    SELECT fieldID FROM fields WHERE fieldName = 'DOI' LIMIT 1
                )
                LEFT JOIN itemDataValues doi ON doi.valueID = doi_id.valueID
                LEFT JOIN itemData url_id ON url_id.itemID = i.itemID AND url_id.fieldID = (
                    SELECT fieldID FROM fields WHERE fieldName = 'url' LIMIT 1
                )
                LEFT JOIN itemDataValues url ON url.valueID = url_id.valueID
                WHERE i.key = ?
                LIMIT 1
                """,
                (key,),
            ).fetchone()
            creators = conn.execute(
                """
                SELECT c.lastName, c.firstName
                FROM items i
                JOIN itemCreators ic ON ic.itemID = i.itemID
                JOIN creators c ON c.creatorID = ic.creatorID
                WHERE i.key = ?
                ORDER BY ic.orderIndex
                """,
                (key,),
            ).fetchall()
        finally:
            conn.close()
        if row is None:
            return f"Zotero item {key} was not found."
        names: list[str] = []
        for creator in creators:
            last = str(creator["lastName"] or "").strip()
            first = str(creator["firstName"] or "").strip()
            label = " ".join(part for part in (first, last) if part)
            if label:
                names.append(label)
        fake = {
            "key": key,
            "data": {
                "title": row["title"] or "Untitled",
                "itemType": row["typeName"] or "",
                "date": row["dateVal"] or "",
                "DOI": row["doi"] or "",
                "url": row["url"] or "",
                "abstractNote": row["abstract"] or "",
                "creators": [{"firstName": n.split(" ", 1)[0], "lastName": n.split(" ", 1)[-1]} for n in names],
            },
        }
        if names:
            fake["data"]["creators"] = [{"name": name} for name in names]
        return _markdown_from_api_item(fake)


def _markdown_from_api_item(row: dict[str, Any]) -> str:
    payload = row.get("data") if isinstance(row.get("data"), dict) else {}
    title = str(payload.get("title") or payload.get("name") or "Untitled").strip() or "Untitled"
    lines = [f"# {title}", ""]
    creators = _item_creators(payload.get("creators"))
    if creators:
        if isinstance(payload.get("creators"), list) and len(payload["creators"]) > 1:
            names: list[str] = []
            for entry in payload["creators"]:
                if not isinstance(entry, dict):
                    continue
                if entry.get("name"):
                    names.append(str(entry["name"]))
                    continue
                last = str(entry.get("lastName") or "").strip()
                first = str(entry.get("firstName") or "").strip()
                label = " ".join(part for part in (first, last) if part)
                if label:
                    names.append(label)
            if names:
                lines.append(f"- Authors: {', '.join(names)}")
        else:
            lines.append(f"- Authors: {creators}")
    if payload.get("date"):
        lines.append(f"- Date: {payload['date']}")
    if payload.get("publicationTitle"):
        lines.append(f"- Publication: {payload['publicationTitle']}")
    if payload.get("itemType"):
        lines.append(f"- Type: {payload['itemType']}")
    if payload.get("DOI"):
        lines.append(f"- DOI: {payload['DOI']}")
    if payload.get("url"):
        lines.append(f"- URL: {payload['url']}")
    key = str(row.get("key") or payload.get("key") or "").strip()
    if key:
        lines.append(f"- Key: {key}")
    abstract = str(payload.get("abstractNote") or "").strip()
    if abstract:
        lines.extend(["", "## Abstract", "", abstract])
    return "\n".join(lines) + "\n"


def _search_tokens(query: str) -> list[str]:
    out: list[str] = []
    current = []
    for ch in query.lower():
        if ch.isalnum():
            current.append(ch)
            continue
        if current:
            token = "".join(current)
            if len(token) >= 3:
                out.append(token)
            current = []
    if current:
        token = "".join(current)
        if len(token) >= 3:
            out.append(token)
    return out


def _item_dict(item: ZoteroItem, *, source: str) -> dict[str, Any]:
    return {
        "key": item.key,
        "title": item.title,
        "year": item.year,
        "creators": item.creators,
        "item_type": item.item_type,
        "source": source,
    }


def _row_hit(row: sqlite3.Row, *, source: str) -> dict[str, Any]:
    last = str(row["lastName"] or "").strip()
    first = str(row["firstName"] or "").strip()
    creators = " ".join(part for part in (first, last) if part)
    return {
        "key": str(row["itemKey"] or "").strip(),
        "title": str(row["title"] or "Untitled").strip() or "Untitled",
        "year": _item_year(row["dateVal"]),
        "creators": creators,
        "item_type": str(row["typeName"] or "").strip(),
        "source": source,
    }


def _truncate(text: str, limit: int = _MAX_BODY_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def format_tool_payload(data: Any) -> str:
    if isinstance(data, str):
        return _truncate(data)
    try:
        return _truncate(json.dumps(data, ensure_ascii=False, indent=2))
    except TypeError:
        return _truncate(str(data))

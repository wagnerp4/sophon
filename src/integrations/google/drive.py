from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from integrations.google.oauth import active_email, build_service

_FOLDER_MIME = "application/vnd.google-apps.folder"
_EXPORT_MIME = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}
_MAX_BODY_CHARS = 12_000
_PAGE_SIZE = 100


@dataclass
class DriveEntry:
    id: str
    name: str
    mime_type: str
    is_folder: bool
    web_view_link: str = ""
    modified_time: str = ""
    size: str = ""


def list_children(parent_id: str = "root", *, email: str | None = None) -> list[DriveEntry]:
    pid = (parent_id or "root").strip() or "root"
    service = build_service("drive", "v3", email=email)
    query = f"'{pid}' in parents and trashed=false"
    fields = (
        "nextPageToken, files(id, name, mimeType, webViewLink, modifiedTime, size)"
    )
    entries: list[DriveEntry] = []
    page_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {
            "q": query,
            "spaces": "drive",
            "fields": fields,
            "pageSize": _PAGE_SIZE,
            "orderBy": "folder,name_natural",
        }
        if page_token:
            kwargs["pageToken"] = page_token
        response = service.files().list(**kwargs).execute()
        for row in response.get("files") or []:
            mime = str(row.get("mimeType") or "")
            entries.append(
                DriveEntry(
                    id=str(row.get("id") or ""),
                    name=str(row.get("name") or "(unnamed)"),
                    mime_type=mime,
                    is_folder=mime == _FOLDER_MIME,
                    web_view_link=str(row.get("webViewLink") or ""),
                    modified_time=str(row.get("modifiedTime") or ""),
                    size=str(row.get("size") or ""),
                )
            )
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    entries.sort(key=lambda e: (not e.is_folder, e.name.lower()))
    return entries


def get_file(file_id: str, *, email: str | None = None) -> DriveEntry:
    fid = (file_id or "").strip()
    if not fid:
        raise ValueError("file_id is required")
    service = build_service("drive", "v3", email=email)
    row = (
        service.files()
        .get(fileId=fid, fields="id, name, mimeType, webViewLink, modifiedTime, size")
        .execute()
    )
    mime = str(row.get("mimeType") or "")
    return DriveEntry(
        id=str(row.get("id") or ""),
        name=str(row.get("name") or "(unnamed)"),
        mime_type=mime,
        is_folder=mime == _FOLDER_MIME,
        web_view_link=str(row.get("webViewLink") or ""),
        modified_time=str(row.get("modifiedTime") or ""),
        size=str(row.get("size") or ""),
    )


def read_file_text(file_id: str, *, email: str | None = None) -> dict[str, Any]:
    entry = get_file(file_id, email=email)
    if entry.is_folder:
        children = list_children(entry.id, email=email)
        return {
            "id": entry.id,
            "name": entry.name,
            "mime_type": entry.mime_type,
            "is_folder": True,
            "web_view_link": entry.web_view_link,
            "children": [
                {"id": c.id, "name": c.name, "mime_type": c.mime_type, "is_folder": c.is_folder}
                for c in children
            ],
            "account": email or active_email(),
        }
    service = build_service("drive", "v3", email=email)
    export_mime = _EXPORT_MIME.get(entry.mime_type)
    text = ""
    note = ""
    try:
        if export_mime:
            raw = service.files().export(fileId=entry.id, mimeType=export_mime).execute()
            if isinstance(raw, bytes):
                text = raw.decode("utf-8", errors="replace")
            else:
                text = str(raw)
        elif entry.mime_type.startswith("text/") or entry.mime_type in (
            "application/json",
            "application/xml",
        ):
            raw = service.files().get_media(fileId=entry.id).execute()
            if isinstance(raw, bytes):
                text = raw.decode("utf-8", errors="replace")
            else:
                text = str(raw)
        else:
            note = (
                "Binary or unsupported Google type. Open the webViewLink in a browser. "
                "sophon does not download binary Drive files into the editor."
            )
    except Exception as exc:
        note = f"read failed: {exc}"
    if len(text) > _MAX_BODY_CHARS:
        text = text[:_MAX_BODY_CHARS] + "\n\n… (truncated)"
    return {
        "id": entry.id,
        "name": entry.name,
        "mime_type": entry.mime_type,
        "is_folder": False,
        "web_view_link": entry.web_view_link,
        "modified_time": entry.modified_time,
        "text": text,
        "note": note,
        "account": email or active_email(),
    }


def format_tree(parent_id: str = "root", *, depth: int = 2, email: str | None = None) -> dict[str, Any]:
    levels = max(1, min(int(depth), 6))

    def _walk(pid: str, remaining: int) -> list[dict[str, Any]]:
        rows = list_children(pid, email=email)
        out: list[dict[str, Any]] = []
        for row in rows:
            item: dict[str, Any] = {
                "id": row.id,
                "name": row.name,
                "mime_type": row.mime_type,
                "is_folder": row.is_folder,
            }
            if row.is_folder and remaining > 1:
                item["children"] = _walk(row.id, remaining - 1)
            out.append(item)
        return out

    return {
        "account": email or active_email(),
        "parent_id": parent_id or "root",
        "depth": levels,
        "children": _walk(parent_id or "root", levels),
    }


def entry_to_markdown(record: dict[str, Any]) -> str:
    name = str(record.get("name") or "Drive item")
    lines = [f"# Drive · {name}", ""]
    lines.append(f"- id: `{record.get('id')}`")
    lines.append(f"- mime: `{record.get('mime_type')}`")
    if record.get("account"):
        lines.append(f"- account: `{record.get('account')}`")
    if record.get("web_view_link"):
        lines.append(f"- link: {record.get('web_view_link')}")
    lines.append("")
    if record.get("is_folder"):
        children = record.get("children") or []
        if not children:
            lines.append("_Empty folder._")
        else:
            for child in children:
                kind = "folder" if child.get("is_folder") else "file"
                lines.append(f"- ({kind}) {child.get('name')} `{child.get('id')}`")
        return "\n".join(lines)
    note = str(record.get("note") or "").strip()
    text = str(record.get("text") or "").strip()
    if note:
        lines.append(f"_{note}_")
        lines.append("")
    if text:
        lines.append(text)
    elif not note:
        lines.append("_No text content._")
    return "\n".join(lines)

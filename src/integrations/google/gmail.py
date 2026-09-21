from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

from integrations.google.oauth import active_email, build_service

_MAX_SEARCH = 25
_MAX_BODY_CHARS = 12_000


@dataclass
class GmailMessage:
    id: str
    thread_id: str
    subject: str
    from_addr: str
    date: str
    snippet: str
    labels: list[str]


def search_messages(query: str, *, limit: int = 20, email: str | None = None) -> list[GmailMessage]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    n = max(1, min(int(limit), _MAX_SEARCH))
    service = build_service("gmail", "v1", email=email)
    response = (
        service.users()
        .messages()
        .list(userId="me", q=q, maxResults=n)
        .execute()
    )
    rows = response.get("messages") or []
    out: list[GmailMessage] = []
    for row in rows:
        mid = str(row.get("id") or "")
        if not mid:
            continue
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=mid, format="metadata", metadataHeaders=["From", "Subject", "Date"])
            .execute()
        )
        out.append(_parse_metadata(msg))
    return out


def read_message(message_id: str, *, email: str | None = None) -> dict[str, Any]:
    mid = (message_id or "").strip()
    if not mid:
        raise ValueError("message_id is required")
    service = build_service("gmail", "v1", email=email)
    msg = service.users().messages().get(userId="me", id=mid, format="full").execute()
    meta = _parse_metadata(msg)
    body = _extract_body(msg.get("payload") or {})
    if len(body) > _MAX_BODY_CHARS:
        body = body[:_MAX_BODY_CHARS] + "\n\n… (truncated)"
    return {
        "id": meta.id,
        "thread_id": meta.thread_id,
        "subject": meta.subject,
        "from": meta.from_addr,
        "date": meta.date,
        "snippet": meta.snippet,
        "labels": meta.labels,
        "body": body,
        "account": email or active_email(),
    }


def format_search_results(messages: list[GmailMessage], *, query: str) -> dict[str, Any]:
    return {
        "query": query,
        "account": active_email(),
        "count": len(messages),
        "messages": [
            {
                "id": m.id,
                "subject": m.subject,
                "from": m.from_addr,
                "date": m.date,
                "snippet": m.snippet,
            }
            for m in messages
        ],
    }


def messages_to_markdown(messages: list[GmailMessage], *, query: str) -> str:
    lines = [f"# Gmail search: {query}", ""]
    account = active_email()
    if account:
        lines.append(f"Account: `{account}`")
        lines.append("")
    if not messages:
        lines.append("_No messages._")
        return "\n".join(lines)
    for msg in messages:
        lines.append(f"## {msg.subject or '(no subject)'}")
        lines.append(f"- id: `{msg.id}`")
        lines.append(f"- from: {msg.from_addr}")
        lines.append(f"- date: {msg.date}")
        if msg.snippet:
            lines.append(f"- snippet: {msg.snippet}")
        lines.append("")
    return "\n".join(lines)


def message_to_markdown(record: dict[str, Any]) -> str:
    lines = [
        f"# {record.get('subject') or '(no subject)'}",
        "",
        f"- id: `{record.get('id')}`",
        f"- from: {record.get('from')}",
        f"- date: {record.get('date')}",
    ]
    if record.get("account"):
        lines.append(f"- account: `{record.get('account')}`")
    lines.append("")
    body = str(record.get("body") or "").strip()
    if body:
        lines.append(body)
    else:
        lines.append(str(record.get("snippet") or "_Empty body._"))
    return "\n".join(lines)


def _parse_metadata(msg: dict[str, Any]) -> GmailMessage:
    headers = {
        str(h.get("name") or "").lower(): str(h.get("value") or "")
        for h in (msg.get("payload") or {}).get("headers") or []
    }
    date_raw = headers.get("date", "")
    date = date_raw
    try:
        date = parsedate_to_datetime(date_raw).isoformat()
    except Exception:
        pass
    return GmailMessage(
        id=str(msg.get("id") or ""),
        thread_id=str(msg.get("threadId") or ""),
        subject=headers.get("subject", "") or "(no subject)",
        from_addr=headers.get("from", ""),
        date=date,
        snippet=str(msg.get("snippet") or ""),
        labels=[str(x) for x in (msg.get("labelIds") or [])],
    )


def _extract_body(payload: dict[str, Any]) -> str:
    mime = str(payload.get("mimeType") or "")
    body = payload.get("body") or {}
    data = body.get("data")
    if data and mime.startswith("text/"):
        text = _decode_b64(str(data))
        if mime == "text/html":
            return _html_to_text(text)
        return text
    parts = payload.get("parts") or []
    plain_bits: list[str] = []
    html_bits: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_mime = str(part.get("mimeType") or "")
        if part_mime.startswith("multipart/"):
            nested = _extract_body(part)
            if nested:
                plain_bits.append(nested)
            continue
        part_body = part.get("body") or {}
        part_data = part_body.get("data")
        if not part_data:
            continue
        decoded = _decode_b64(str(part_data))
        if part_mime == "text/plain":
            plain_bits.append(decoded)
        elif part_mime == "text/html":
            html_bits.append(_html_to_text(decoded))
    if plain_bits:
        return "\n\n".join(plain_bits)
    if html_bits:
        return "\n\n".join(html_bits)
    return ""


def _decode_b64(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = re.sub(r"(?s)<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

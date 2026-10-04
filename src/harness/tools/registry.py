from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ListedTool:
    name: str
    origin: str
    connector: str
    schema: dict[str, Any] | None
    shadowed_by: str = ""


def schema_name(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(tool.get("name") or "")


def note_injected(state: object, name: str) -> None:
    token = str(name or "").strip()
    if not token:
        return
    seen = getattr(state, "seen_injected", None)
    if not isinstance(seen, list):
        seen = []
        setattr(state, "seen_injected", seen)
    if token not in seen:
        seen.append(token)


def _stub_schema(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Injected tool forwarded by the backend. No local executor.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
        },
    }


def build_inventory(
    nexus_tools: list[dict[str, Any]],
    *,
    project_root: object,
    seen_injected: list[str] | None = None,
) -> list[ListedTool]:
    from integrations.mcp.host import injected_names, retrace_tools

    rows: list[ListedTool] = []
    names: set[str] = set()
    for tool in nexus_tools:
        name = schema_name(tool)
        if not name:
            continue
        names.add(name)
        rows.append(ListedTool(name=name, origin="nexus", connector="nexus", schema=tool))
    for item in retrace_tools(project_root):
        shadow = "nexus" if item.name in names else ""
        rows.append(
            ListedTool(
                name=item.name,
                origin="mcp",
                connector=item.connector,
                schema=None if shadow else item.schema,
                shadowed_by=shadow,
            )
        )
        if not shadow:
            names.add(item.name)
    for name in injected_names(project_root, seen_injected or []):
        shadow = "nexus" if name in names else ""
        rows.append(
            ListedTool(
                name=name,
                origin="injected",
                connector="injected",
                schema=None if shadow else _stub_schema(name),
                shadowed_by=shadow,
            )
        )
        if not shadow:
            names.add(name)
    return rows


def model_schemas(rows: list[ListedTool]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.shadowed_by or row.schema is None:
            continue
        out.append(row.schema)
    return out


def count_origins(rows: list[ListedTool]) -> tuple[int, int, int]:
    nexus = sum(1 for row in rows if row.origin == "nexus")
    mcp = sum(1 for row in rows if row.origin == "mcp")
    injected = sum(1 for row in rows if row.origin == "injected")
    return nexus, mcp, injected


def format_inventory(rows: list[ListedTool], origin: str | None = None) -> list[str]:
    selected = rows if not origin or origin == "all" else [row for row in rows if row.origin == origin]
    nexus, mcp, injected = count_origins(rows)
    total = nexus + mcp + injected
    lines = [
        f"NexusTools: {nexus} · MCP: {mcp} · Injected: {injected} · total {total} ({nexus}n+{mcp}m+{injected}i)"
    ]
    for row in selected:
        shadow = f" shadowed_by={row.shadowed_by}" if row.shadowed_by else ""
        lines.append(f"  {row.origin} {row.connector} {row.name}{shadow}")
    return lines
